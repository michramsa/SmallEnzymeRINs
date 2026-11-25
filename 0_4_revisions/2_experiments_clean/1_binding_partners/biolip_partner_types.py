"""
biolip_partner_types.py
=======================
For a list of PDB IDs, build a per-enzyme table of functional residues
(binding + catalytic, matching biolip_residue_labels.py) and the TYPE of
binding partner each residue contacts.

Pipeline
--------
1. Read PDB IDs from a text file (comma and/or whitespace separated).
2. Pull the matching rows from BioLiP.txt (cols 01, 02, 05, 06, 07, 08, 10).
3. Look up every unique ligand ID (col 05) in the RCSB Chemical Component
   Dictionary via the Data API (cached to JSON so each code is fetched once).
4. Classify each ligand ID into a partner type (see classify_ligand).
5. For each functional residue, collect the partner types of every BioLiP
   row (ligand) whose binding-residue list contains that residue.

Outputs
-------
<OUT_CSV>  (one row per PDB ID)
    pdb_id, found_in_biolip, n_functional_residues, residue_partners
    residue_partners looks like:
        [('A:H111', 'metal_ion'), ('A:E112', 'cofactor+metal_ion'), ('A:D50', 'none')]
    - A residue contacting several ligand types gets them joined with '+'.
    - 'none' = catalytic residue with no ligand contact recorded in BioLiP.

<LONG_CSV>  (one row per residue x partner type; easier to analyse)
    pdb_id, chain, residue, role (binding|catalytic|dual), partner_type,
    ligand_ids

Partner types
-------------
peptide, DNA, RNA, cofactor, metal_ion, nonmetal_ion, saccharide,
small_molecule, unknown (CCD lookup failed), none.

Caveats (read these)
--------------------
* Catalytic residues (BioLiP col 10) are listed on every row of a protein,
  independent of which ligand that row is about. So a catalytic residue only
  gets a partner if it ALSO appears in a row's binding-residue list (col 08).
* BioLiP's placeholder IDs for polymer ligands ('peptide', 'dna', 'rna') are
  recalled from memory of the database, not from the readme. They are matched
  case-insensitively; check against your own file (cut -f5 | sort | uniq -c).
* 'cofactor' is a curated list (COFACTOR_IDS below), not something the PDB
  records. Edit it to suit your definition. Cofactor-list codes take
  priority over the CCD-derived type; metal ions are kept as their own class.

Inputs (edit in main())
-----------------------
PDB_ID_FILE, BIOLIP_TXT, OUT_CSV, LONG_CSV, CCD_CACHE_JSON,
N_WORKERS, REQUEST_TIMEOUT, COFACTOR_IDS
"""

import re
import json
import time
import logging
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# BioLiP.txt column indices (0-based) we actually need
_COL_IDX = {
    0: "pdb_id",
    1: "receptor_chain",
    4: "ligand_id",
    5: "ligand_chain",
    6: "ligand_serial",
    7: "binding_residues_pdb",
    9: "catalytic_residues_pdb",
}

CCD_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{}"

# Placeholder ligand IDs BioLiP uses for polymer partners (lower-case match)
POLYMER_PLACEHOLDERS = {"peptide": "peptide", "dna": "DNA", "rna": "RNA"}

# Default cofactor CCD codes -- EDIT FREELY. Deliberately excludes ATP/ADP/GTP
# etc. (ambiguous: substrate vs cofactor) and plain metal ions.
DEFAULT_COFACTOR_IDS = {
    "NAD", "NAP", "NDP", "NAI", "NAJ",          # NAD(H) / NADP(H)
    "FAD", "FMN", "FDA",                        # flavins
    "HEM", "HEC", "HEA", "HEB", "HDD",          # hemes
    "COA", "ACO",                               # coenzyme A / acetyl-CoA
    "PLP", "PMP",                               # pyridoxal phosphate
    "TPP", "TDP",                               # thiamine pyrophosphate
    "SAM",                                      # S-adenosylmethionine
    "SF4", "FES", "F3S", "CLF",                 # Fe-S clusters
    "PQQ", "TPQ", "LLP",                        # quinones / modified lysine PLP
    "COB", "B12", "CNC",                        # cobalamins
    "THF", "FOL", "MTE",                        # folate / molybdopterin
    "BTN", "LIP", "LPA",                        # biotin / lipoate
    "F43", "MGD",                               # F430 / Mo-bisMGD
}

_METALS = {
    "Li", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",
    "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu",
    "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
    "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "U", "Th",
}

_RESIDUE_TOKEN_RE = re.compile(r"^([A-Za-z])(-?\d+[A-Za-z]?)$")
_SINGLE_ATOM_RE = re.compile(r"^[A-Za-z]{1,2}$")


# ---------------------------------------------------------------------------
# PDB ID file
# ---------------------------------------------------------------------------

def read_pdb_ids(path: Path) -> list[str]:
    """Return unique lower-case PDB IDs (order preserved) from a text file."""
    text = Path(path).read_text()
    ids = [t.strip().lower() for t in re.split(r"[,\s]+", text) if t.strip()]
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    log.info("Read %d unique PDB IDs from %s", len(out), path)
    return out


# ---------------------------------------------------------------------------
# BioLiP loading
# ---------------------------------------------------------------------------

def load_biolip_subset(biolip_txt: Path, pdb_ids: set[str]) -> pd.DataFrame:
    """Chunked read of BioLiP.txt keeping only needed columns / PDB IDs."""
    usecols = sorted(_COL_IDX)
    names = [_COL_IDX[i] for i in usecols]
    parts = []
    reader = pd.read_csv(
        biolip_txt, sep="\t", header=None, usecols=usecols, names=names,
        dtype=str, chunksize=500_000, low_memory=False,
    )
    for chunk in reader:
        chunk["pdb_id"] = chunk["pdb_id"].str.strip().str.lower()
        parts.append(chunk[chunk["pdb_id"].isin(pdb_ids)])
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=names)
    df["receptor_chain"] = df["receptor_chain"].str.strip()
    df["ligand_id"] = df["ligand_id"].str.strip()
    log.info("BioLiP rows kept: %d (%d PDB IDs)", len(df), df["pdb_id"].nunique())
    return df


def parse_residues(field) -> list[str]:
    """'N180 L181 A182' -> ['N180','L181','A182']; null/placeholder -> []."""
    if not isinstance(field, str):
        return []
    s = field.strip()
    if s in ("", "-", "nan", "NaN"):
        return []
    out = []
    for tok in s.split():
        if _RESIDUE_TOKEN_RE.match(tok):
            out.append(tok)
        else:
            log.warning("Unrecognised residue token skipped: %r", tok)
    return out


def parse_catalytic(field) -> list[str]:
    """Catalytic field separates sites with ';' and residues with spaces."""
    if not isinstance(field, str):
        return []
    out = []
    for site in field.split(";"):
        out.extend(parse_residues(site))
    return out


# ---------------------------------------------------------------------------
# CCD lookup (RCSB Data API) with JSON cache
# ---------------------------------------------------------------------------

def _fetch_one(code: str, timeout: float, retries: int = 3) -> dict | None:
    """Fetch name/type/formula for one CCD code. None on failure / 404."""
    url = CCD_URL.format(code)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                cc = json.load(resp).get("chem_comp", {})
            return {
                "name": cc.get("name", ""),
                "type": cc.get("type", ""),
                "formula": cc.get("formula", ""),
            }
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None          # not a CCD code
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def fetch_ccd_records(codes: list[str], cache_json: Path,
                      n_workers: int, timeout: float) -> dict[str, dict | None]:
    """Return {code: record or None}; reuses and updates the JSON cache."""
    cache: dict[str, dict | None] = {}
    if cache_json.exists():
        cache = json.loads(cache_json.read_text())
    todo = [c for c in codes if c not in cache]
    log.info("CCD codes: %d unique, %d cached, %d to fetch",
             len(codes), len(codes) - len(todo), len(todo))

    if todo:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for code, rec in zip(todo, ex.map(lambda c: _fetch_one(c, timeout), todo)):
                cache[code] = rec
        cache_json.parent.mkdir(parents=True, exist_ok=True)
        cache_json.write_text(json.dumps(cache, indent=1))
        log.info("Cache updated: %s", cache_json)
    return {c: cache.get(c) for c in codes}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_ligand(code: str, rec: dict | None, cofactor_ids: set[str]) -> str:
    """Map a BioLiP ligand ID (+ its CCD record) to a partner type."""
    low = code.lower()
    if low in POLYMER_PLACEHOLDERS:
        return POLYMER_PLACEHOLDERS[low]
    if code.upper() in cofactor_ids:
        return "cofactor"
    if rec is None:
        return "unknown"

    ccd_type = (rec.get("type") or "").lower()
    formula = (rec.get("formula") or "").strip()

    # single-atom entry -> ion
    if len(formula.split()) == 1 and _SINGLE_ATOM_RE.match(formula):
        return "metal_ion" if formula.capitalize() in _METALS else "nonmetal_ion"
    if "peptide" in ccd_type:
        return "peptide"
    if "rna linking" in ccd_type:
        return "RNA"
    if "dna linking" in ccd_type:
        return "DNA"
    if "saccharide" in ccd_type:
        return "saccharide"
    return "small_molecule"


# ---------------------------------------------------------------------------
# Per-PDB residue/partner table
# ---------------------------------------------------------------------------

def build_tables(biolip_df: pd.DataFrame, pdb_ids: list[str],
                 type_of: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide_rows, long_rows = [], []
    grouped = {k: g for k, g in biolip_df.groupby("pdb_id")}

    for pdb in pdb_ids:
        g = grouped.get(pdb)
        if g is None or g.empty:
            wide_rows.append({"pdb_id": pdb, "found_in_biolip": False,
                              "n_functional_residues": 0, "residue_partners": "[]"})
            continue

        # key: (chain, residue) -> info
        info: dict[tuple[str, str], dict] = {}

        def _entry(chain, res):
            return info.setdefault((chain, res), {
                "binding": False, "catalytic": False,
                "types": set(), "ligands": set()})

        for _, row in g.iterrows():
            chain, lig = row["receptor_chain"], row["ligand_id"]
            ptype = type_of.get(lig, "unknown")
            for res in parse_residues(row["binding_residues_pdb"]):
                e = _entry(chain, res)
                e["binding"] = True
                e["types"].add(ptype)
                e["ligands"].add(lig)
            for res in parse_catalytic(row["catalytic_residues_pdb"]):
                _entry(chain, res)["catalytic"] = True

        pairs = []
        for (chain, res), e in sorted(
                info.items(), key=lambda kv: (kv[0][0], _sort_key(kv[0][1]))):
            role = ("dual" if e["binding"] and e["catalytic"]
                    else "catalytic" if e["catalytic"] else "binding")
            types = sorted(e["types"]) if e["types"] else ["none"]
            label = f"{chain}:{res}"
            pairs.append((label, "+".join(types)))
            for t in types:
                long_rows.append({
                    "pdb_id": pdb, "chain": chain, "residue": res, "role": role,
                    "partner_type": t,
                    "ligand_ids": ";".join(sorted(e["ligands"])),
                })

        wide_rows.append({"pdb_id": pdb, "found_in_biolip": True,
                          "n_functional_residues": len(pairs),
                          "residue_partners": repr(pairs)})

    return pd.DataFrame(wide_rows), pd.DataFrame(long_rows)


def _sort_key(res: str):
    m = re.match(r"^[A-Za-z](-?\d+)([A-Za-z]?)$", res)
    return (int(m.group(1)), m.group(2)) if m else (0, res)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(pdb_id_file: Path, biolip_txt: Path, out_csv: Path, long_csv: Path,
        ccd_cache_json: Path, n_workers: int, request_timeout: float,
        cofactor_ids: set[str]) -> pd.DataFrame:
    pdb_ids = read_pdb_ids(pdb_id_file)
    biolip_df = load_biolip_subset(biolip_txt, set(pdb_ids))

    codes = sorted(c for c in set(biolip_df["ligand_id"].dropna())
                   if c.lower() not in POLYMER_PLACEHOLDERS)
    # only hit the API for codes the cofactor list doesn't already settle
    need_api = [c for c in codes if c.upper() not in cofactor_ids]
    records = fetch_ccd_records(need_api, ccd_cache_json, n_workers, request_timeout)

    type_of = {c: classify_ligand(c, records.get(c), cofactor_ids)
               for c in set(biolip_df["ligand_id"].dropna())}

    wide_df, long_df = build_tables(biolip_df, pdb_ids, type_of)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(out_csv, index=False)
    long_df.to_csv(long_csv, index=False)
    log.info("Wrote %s (%d rows) and %s (%d rows)",
             out_csv, len(wide_df), long_csv, len(long_df))

    n_unknown = sum(1 for t in type_of.values() if t == "unknown")
    if n_unknown:
        bad = sorted(c for c, t in type_of.items() if t == "unknown")
        log.warning("%d ligand IDs unresolved (typed 'unknown'): %s",
                    n_unknown, ", ".join(bad[:30]) + (" ..." if len(bad) > 30 else ""))
    print("\nPartner-type counts (residue x type):")
    if not long_df.empty:
        print(long_df["partner_type"].value_counts().to_string())
    return wide_df


def main():
    # ── Edit these ───────────────────────────────────────────────────────────
    PDB_ID_FILE     = Path("/home/epi2melabs/AcrNetworks-clean-v2/pdbids.txt")
    BIOLIP_TXT      = Path("/home/epi2melabs/AcrNetworks-clean-v2/BioLiP.txt")
    OUT_CSV         = Path("biolip_partner_types/partner_types_per_pdb.csv")
    LONG_CSV        = Path("biolip_partner_types/partner_types_per_residue.csv")
    CCD_CACHE_JSON  = Path("biolip_partner_types/ccd_cache.json")
    N_WORKERS       = 8       # parallel API requests
    REQUEST_TIMEOUT = 20.0    # seconds per request
    COFACTOR_IDS    = set(DEFAULT_COFACTOR_IDS)   # add/remove codes here
    # ─────────────────────────────────────────────────────────────────────────

    run(PDB_ID_FILE, BIOLIP_TXT, OUT_CSV, LONG_CSV, CCD_CACHE_JSON,
        N_WORKERS, REQUEST_TIMEOUT, COFACTOR_IDS)


if __name__ == "__main__":
    main()
