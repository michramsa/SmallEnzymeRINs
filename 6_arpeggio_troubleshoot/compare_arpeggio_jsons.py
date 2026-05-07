"""
compare_arpeggio_jsons.py
--------------------------
Semantically compares two Arpeggio JSON outputs:
  - CIF method : produced via clean_cif.py  (mmCIF-native path)
  - PDB method : produced via clean_pdb.py  (legacy PDB path)

Edit the parameters in the `if __name__ == "__main__":` block at the
bottom of this file, then run:
    python compare_arpeggio_jsons.py

Outputs (written to OUT_DIR):
  - contacts_only_in_cif.csv     contacts found by CIF method but not PDB
  - contacts_only_in_pdb.csv     contacts found by PDB method but not CIF
  - flag_differences.csv         contacts in both, but with differing flags
  - summary.txt                  copy of the printed summary
"""

# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *


# ===========================================================================
# JSON loading
# ===========================================================================

def load_json(path: str) -> list:
    """Load an Arpeggio JSON file and return the contacts as a flat list."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    # Some Arpeggio versions wrap contacts under a key
    for key in ("contacts", "results", "data"):
        if key in data:
            return data[key]

    raise ValueError(
        f"Unrecognised JSON structure in {path}. "
        f"Top-level keys: {list(data.keys())}"
    )


# ===========================================================================
# Contact key construction
# ===========================================================================

def atom_id(atom: dict) -> str:
    """
    Build a canonical string identifier for one atom endpoint.

    Arpeggio's JSON contacts each have two atom entries.
    The key names vary slightly by Arpeggio version; this function
    tries the most common conventions in order.

    Format returned:  CHAIN/RESNUM[ICODE]/RESNAME/ATOMNAME
    e.g.              A/42/LEU/CA
    """
    # --- Arpeggio mmCIF output (confirmed schema from test output) ---
    if "auth_asym_id" in atom:
        chain   = atom.get("auth_asym_id", "?")
        seq     = str(atom.get("auth_seq_id", "?"))
        ins     = (atom.get("pdbx_PDB_ins_code") or "").strip()
        resname = atom.get("label_comp_id", "?")
        aname   = atom.get("auth_atom_id", "?")

    # --- Fallback: older Arpeggio flat key style ---
    elif "chain" in atom or "chain_id" in atom:
        chain   = atom.get("chain", atom.get("chain_id", "?"))
        seq     = str(atom.get("res_id", atom.get("seq_id", "?")))
        ins     = (atom.get("ins_code") or "").strip()
        resname = atom.get("res_name", atom.get("comp_id", "?"))
        aname   = atom.get("atom_name", atom.get("atom_id", "?"))

    else:
        raise KeyError(
            f"Cannot parse atom dict — unrecognised keys: {list(atom.keys())}"
        )

    return f"{chain}/{seq}{ins}/{resname}/{aname}"


def contact_key(contact: dict) -> tuple:
    """
    Return an order-normalised tuple key for a contact so that
    (atomA->atomB) and (atomB->atomA) hash identically.

    Tries bgn/end keys first, then atom1/atom2, then positional fallback.
    """
    if "bgn" in contact and "end" in contact:
        a = atom_id(contact["bgn"])
        b = atom_id(contact["end"])

    elif "atom1" in contact and "atom2" in contact:
        a = atom_id(contact["atom1"])
        b = atom_id(contact["atom2"])

    else:
        atom_vals = [v for v in contact.values() if isinstance(v, dict)]
        if len(atom_vals) >= 2:
            a = atom_id(atom_vals[0])
            b = atom_id(atom_vals[1])
        else:
            raise ValueError(
                f"Cannot parse contact — no atom pair found. "
                f"Keys present: {list(contact.keys())}"
            )

    return tuple(sorted([a, b]))


def get_flags(contact: dict) -> dict:
    """
    Extract all interaction fields from a contact dict,
    excluding the atom endpoint dicts and the numeric distance.

    In this version of Arpeggio the flag fields are:
      - contact              : list of strings e.g. ['proximal'],
                               ['hbond', 'weak_hbond']
      - interacting_entities : string e.g. 'INTRA_SELECTION', 'INTER'
      - type                 : string e.g. 'atom-atom'
    """
    skip = {"bgn", "end", "atom1", "atom2", "distance", "dist"}
    return {k: v for k, v in contact.items() if k not in skip}


# ===========================================================================
# Analysis helpers
# ===========================================================================

def flag_type_summary(keys: set, lookup: dict) -> dict:
    """
    For a set of contact keys, tally how often each contact type appears.

    Handles Arpeggio's actual flag formats:
      - contact field is a list of strings: each string is counted separately
        e.g. ['hbond', 'weak_hbond'] increments both 'hbond' and 'weak_hbond'
      - interacting_entities / type are strings: counted as-is
      - legacy boolean flags (True/1) are also handled for compatibility
    """
    tally = defaultdict(int)
    for key in keys:
        for field, val in get_flags(lookup[key]).items():
            if isinstance(val, list):
                # e.g. contact: ['proximal'] or ['hbond', 'vdw_clash']
                for item in val:
                    tally[str(item)] += 1
            elif isinstance(val, str):
                tally[f"{field}={val}"] += 1
            elif val is True or val == 1:
                tally[field] += 1
    return dict(sorted(tally.items()))


def contacts_to_rows(keys: set, lookup: dict) -> list:
    """
    Convert a set of contact keys into a list of flat dicts suitable
    for writing to CSV.
    """
    rows = []
    for key in sorted(keys):
        row = {"contact_key": str(key)}
        row.update(get_flags(lookup[key]))
        rows.append(row)
    return rows


def write_csv(rows: list, path: str):
    """Write a list of dicts to CSV.  Writes a placeholder if empty."""
    if not rows:
        with open(path, "w") as f:
            f.write("(no entries)\n")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ===========================================================================
# MSE / MET cross-check
# ===========================================================================

def mse_met_report(only_cif: set, only_pdb: set) -> list:
    """
    Identify contacts that are likely the same physical interaction but
    appear in different sets purely because clean_cif.py retains MSE
    while clean_pdb.py converts MSE->MET.

    Returns a list of (cif_key, pdb_key) matched pairs.
    """
    matched = []
    for cif_key in only_cif:
        if not any("MSE" in part for part in cif_key):
            continue
        # Build what this key would look like with MET substituted for MSE
        met_key = tuple(sorted(
            part.replace("MSE", "MET") if "MSE" in part else part
            for part in cif_key
        ))
        if met_key in only_pdb:
            matched.append((cif_key, met_key))
    return matched


# ===========================================================================
# Main comparison
# ===========================================================================

def compare(cif_json_path: str,
            pdb_json_path: str,
            out_dir: str = "."):

    os.makedirs(out_dir, exist_ok=True)
    lines = []  # mirror all output so we can write summary.txt

    def log(msg=""):
        print(msg)
        lines.append(msg)

    # ---- Load ----
    log(f"Loading CIF-method JSON : {cif_json_path}")
    cif_contacts = load_json(cif_json_path)
    log(f"Loading PDB-method JSON : {pdb_json_path}")
    pdb_contacts = load_json(pdb_json_path)

    log()
    log("=" * 62)
    log(f"  Total contacts  |  CIF: {len(cif_contacts):>6}  |  PDB: {len(pdb_contacts):>6}")
    log("=" * 62)

    # ---- Build lookup dicts keyed by canonical contact tuple ----
    cif_lookup: dict = {}
    for c in cif_contacts:
        cif_lookup[contact_key(c)] = c

    pdb_lookup: dict = {}
    for c in pdb_contacts:
        pdb_lookup[contact_key(c)] = c

    cif_keys = set(cif_lookup.keys())
    pdb_keys = set(pdb_lookup.keys())

    only_cif = cif_keys - pdb_keys
    only_pdb = pdb_keys - cif_keys
    in_both  = cif_keys & pdb_keys

    log()
    log("Contact overlap")
    log(f"  In both            : {len(in_both):>6}  "
        f"({100 * len(in_both) / max(len(cif_keys), 1):.1f}% of CIF, "
        f"{100 * len(in_both) / max(len(pdb_keys), 1):.1f}% of PDB)")
    log(f"  Only in CIF method : {len(only_cif):>6}")
    log(f"  Only in PDB method : {len(only_pdb):>6}")

    # ---- Flag differences among shared contacts ----
    flag_diff_rows = []
    for key in sorted(in_both):
        cf = get_flags(cif_lookup[key])
        pf = get_flags(pdb_lookup[key])
        if cf != pf:
            differing = [
                k for k in set(list(cf.keys()) + list(pf.keys()))
                if cf.get(k) != pf.get(k)
            ]
            flag_diff_rows.append({
                "contact_key"    : str(key),
                "differing_flags": ", ".join(sorted(differing)),
                "cif_values"     : str({k: cf.get(k) for k in differing}),
                "pdb_values"     : str({k: pf.get(k) for k in differing}),
            })

    log(f"  Shared, flags differ:{len(flag_diff_rows):>5}")

    # ---- Flag-type breakdown for unique contacts ----
    log()
    log("Flag breakdown — contacts ONLY in CIF method:")
    cif_summary = flag_type_summary(only_cif, cif_lookup)
    if cif_summary:
        for flag, count in cif_summary.items():
            log(f"    {flag:<42} {count:>5}")
    else:
        log("    (none)")

    log()
    log("Flag breakdown — contacts ONLY in PDB method:")
    pdb_summary = flag_type_summary(only_pdb, pdb_lookup)
    if pdb_summary:
        for flag, count in pdb_summary.items():
            log(f"    {flag:<42} {count:>5}")
    else:
        log("    (none)")

    # ---- MSE / MET cross-check ----
    # clean_cif.py retains MSE; clean_pdb.py converts MSE->MET.
    # A contact may appear in different sets purely because of this name change.
    mse_matches = mse_met_report(only_cif, only_pdb)
    log()
    log("MSE / MET residue check")
    log(f"  CIF-only contacts involving MSE : "
        f"{sum(1 for k in only_cif if any('MSE' in p for p in k))}")
    log(f"  PDB-only contacts involving MET : "
        f"{sum(1 for k in only_pdb if any('MET' in p for p in k))}")
    log(f"  Likely same-contact MSE<->MET pairs : {len(mse_matches)}")
    if mse_matches:
        log("  First few matched pairs:")
        for cif_k, pdb_k in mse_matches[:5]:
            log(f"    CIF key : {cif_k}")
            log(f"    PDB key : {pdb_k}")

    # ---- Source-code note ----
    # clean_cif.py passes polypeptide_residues to CleanSelect but
    # accept_residue() never uses it, so the HETATM->ATOM reclassification
    # that clean_pdb.py applies to polypeptide members is absent.
    # Contacts involving residues at that boundary may differ as a result.
    log()
    log("Note (from source code inspection of clean_cif.py)")
    log("  CleanSelect.accept_residue() does not filter on polypeptide_residues,")
    log("  so non-standard residues are NOT reclassified HETATM->ATOM the way")
    log("  clean_pdb.py does. Contacts involving such residues may appear in")
    log("  only one method's output.")

    # ---- Write output files ----
    p1 = os.path.join(out_dir, "3ERK_contacts_only_in_cif.csv")
    p2 = os.path.join(out_dir, "3ERK_contacts_only_in_pdb.csv")
    p3 = os.path.join(out_dir, "3ERK_flag_differences.csv")
    p4 = os.path.join(out_dir, "3ERK_summary.txt")

    write_csv(contacts_to_rows(only_cif, cif_lookup), p1)
    write_csv(contacts_to_rows(only_pdb, pdb_lookup), p2)
    write_csv(flag_diff_rows, p3)

    with open(p4, "w") as f:
        f.write("\n".join(lines))

    log()
    log("Output files written:")
    log(f"  {p1}")
    log(f"  {p2}")
    log(f"  {p3}")
    log(f"  {p4}")
    log()
    log("Done.")


# ===========================================================================
# EDIT PARAMETERS HERE, THEN RUN
# ===========================================================================

if __name__ == "__main__":

    # Path to the Arpeggio JSON produced by the CIF-native cleaning method
    CIF_JSON = "/home/epi2melabs/AcrNetworks-clean-v2/6_arpeggio_troubleshoot/arp_results/JSONs/3ERK_clean.json"

    # Path to the Arpeggio JSON produced by the legacy PDB cleaning method
    PDB_JSON = "/home/epi2melabs/AcrNetworks-clean-v2/6_arpeggio_troubleshoot/arp_results/pdb_JSONs/3ERK_clean_PDB.json"

    # Directory where output CSVs and summary.txt will be written
    OUT_DIR = "/home/epi2melabs/AcrNetworks-clean-v2/6_arpeggio_troubleshoot/arpeggio_comparison"

    # ----------------------------------------------------------------------
    compare(
        cif_json_path=CIF_JSON,
        pdb_json_path=PDB_JSON,
        out_dir=OUT_DIR,
    )
