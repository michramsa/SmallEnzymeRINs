"""
0_diag_ins_code_audit_ver1.py
==============================
Diagnostic script to audit the pdbx_PDB_ins_code field across a folder
of Arpeggio JSONs. Reports:

  1. What values pdbx_PDB_ins_code actually takes across all JSONs
     (expected: None, "?", "", or a letter like "A")
  2. Whether any proteins have node ID collisions under the CURRENT
     scheme (auth_asym_id + auth_seq_id only — no insertion code)
  3. Whether those collisions involve standard amino acids (i.e., would
     actually affect RIN construction)
  4. A per-protein count table: CIF residues vs JSON-visible residues
     vs RIN nodes, so you can see where the gaps are

Outputs
-------
  ins_code_audit_summary.csv   — one row per JSON file
  ins_code_collisions.csv      — one row per collision (if any)
  ins_code_value_counts.csv    — frequency table of raw ins_code values

Edit the paths in main() before running.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imports import *


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
}


# ---------------------------------------------------------------------------
# CIF helpers (mirrors 0_6_biolip_lookup_ver1.py approach)
# ---------------------------------------------------------------------------
def get_standard_residues_from_cif(cif_path: Path) -> tuple[set, str | None]:
    """
    Return the set of (chain, auth_seq_id, ins_code) tuples for every
    standard amino-acid ATOM residue in the CIF.

    Tries gemmi first; falls back to BioPython MMCIFParser.

    Returns
    -------
    (residues, error_msg)
        residues  : set of (chain_id, seq_id_str, ins_code_str) tuples
        error_msg : None on success, description string on failure
    """
    # ── gemmi path ───────────────────────────────────────────────────────────
    try:
        import gemmi
        st = gemmi.read_structure(str(cif_path))
        residues = set()
        for model in st:
            for chain in model:
                for res in chain:
                    if res.entity_type == gemmi.EntityType.Polymer:
                        if res.name in STANDARD_AA:
                            ins = res.seqid.icode.strip() if res.seqid.icode else ""
                            residues.add((
                                chain.name,
                                str(res.seqid.num),
                                ins,
                            ))
        if residues:
            return residues, None
    except Exception:
        pass  # fall through to BioPython

    # ── BioPython fallback ───────────────────────────────────────────────────
    try:
        from Bio.PDB import MMCIFParser
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(cif_path.stem, str(cif_path))
        residues = set()
        for model in structure:
            for chain in model:
                for res in chain:
                    if (res.get_resname().strip() in STANDARD_AA
                            and res.get_id()[0] == " "):
                        het, seq_num, ins = res.get_id()
                        residues.add((
                            chain.id,
                            str(seq_num),
                            ins.strip(),
                        ))
        return residues, None
    except Exception as exc:
        return set(), str(exc)


# ---------------------------------------------------------------------------
# JSON audit helpers
# ---------------------------------------------------------------------------
def audit_single_json(
    json_path: Path,
    cif_dir: Path | None = None,
) -> dict:
    """
    Audit one Arpeggio JSON file.

    Returns a dict with keys:
        stem, n_interactions_total, n_interactions_std_aa,
        n_unique_residues_json,       # residues visible in JSON (std AA only)
        n_unique_residues_json_noins, # same but node-ID without ins_code
        n_collisions,                 # residues that collapse under current scheme
        collision_details,            # list of collision dicts
        ins_code_values,              # set of raw ins_code strings seen
        n_cif_residues,               # from CIF (None if CIF not found)
        cif_error,
    """
    stem = json_path.stem
    result = {
        "stem":                        stem,
        "n_interactions_total":        0,
        "n_interactions_std_aa":       0,
        "n_unique_residues_json":      0,
        "n_unique_residues_json_noins": 0,
        "n_collisions":                0,
        "collision_details":           [],
        "ins_code_values":             set(),
        "n_cif_residues":              None,
        "cif_error":                   None,
    }

    # ── load JSON ────────────────────────────────────────────────────────────
    try:
        with open(json_path) as f:
            interactions = json.load(f)
    except Exception as exc:
        result["cif_error"] = f"JSON load error: {exc}"
        return result

    result["n_interactions_total"] = len(interactions)

    # ── scan interactions ─────────────────────────────────────────────────────
    # full_id  : (chain, seq_id, ins_code)  — the "truth" identity
    # short_id : (chain, seq_id)            — what the current code uses
    full_residues: dict[tuple, dict] = {}    # full_id → attrs
    short_to_fulls: dict[tuple, set] = defaultdict(set)   # short_id → {full_ids}

    for interaction in interactions:
        for side in ("bgn", "end"):
            atom = interaction[side]
            comp = atom.get("label_comp_id", "")
            if comp not in STANDARD_AA:
                continue

            result["n_interactions_std_aa"] += 1

            chain  = atom.get("auth_asym_id", "")
            seq_id = str(atom.get("auth_seq_id", ""))
            raw_ins = atom.get("pdbx_PDB_ins_code", None)

            # normalise: None, "?", and "" all mean "no insertion code"
            if raw_ins is None or str(raw_ins).strip() in ("?", ""):
                ins = ""
            else:
                ins = str(raw_ins).strip()

            result["ins_code_values"].add(
                repr(atom.get("pdbx_PDB_ins_code", None))
            )

            full_id  = (chain, seq_id, ins)
            short_id = (chain, seq_id)

            full_residues[full_id] = {
                "chain":   chain,
                "seq_id":  seq_id,
                "ins_code": ins,
                "comp_id": comp,
            }
            short_to_fulls[short_id].add(full_id)

    result["n_interactions_std_aa"] //= 2  # counted once per side
    result["n_unique_residues_json"]      = len(full_residues)
    result["n_unique_residues_json_noins"] = len(short_to_fulls)

    # ── detect collisions ─────────────────────────────────────────────────────
    collisions = []
    for short_id, fulls in short_to_fulls.items():
        if len(fulls) > 1:
            chain, seq_id = short_id
            collisions.append({
                "stem":    stem,
                "chain":   chain,
                "seq_id":  seq_id,
                "n_collapsed_to_1": len(fulls),
                "full_ids": "; ".join(
                    f"{f[0]}_{f[1]}_{f[2] or '(none)'}" for f in sorted(fulls)
                ),
                "comp_ids": "; ".join(
                    full_residues[f]["comp_id"] for f in sorted(fulls)
                ),
            })

    result["n_collisions"]       = len(collisions)
    result["collision_details"]  = collisions

    # ── CIF count (optional) ─────────────────────────────────────────────────
    if cif_dir is not None:
        cif_path = cif_dir / f"{stem}.cif"
        if cif_path.exists():
            cif_residues, err = get_standard_residues_from_cif(cif_path)
            result["n_cif_residues"] = len(cif_residues)
            result["cif_error"]      = err
        else:
            result["cif_error"] = "CIF not found"

    return result


# ---------------------------------------------------------------------------
# Main audit loop
# ---------------------------------------------------------------------------
def run_audit(
    json_dir:  Path,
    out_dir:   Path,
    cif_dir:   Path | None = None,
    max_files: int | None = None,
):
    """
    Audit all JSONs in json_dir and write three CSVs to out_dir.

    Parameters
    ----------
    json_dir  : folder containing Arpeggio .json files
    out_dir   : folder for output CSVs
    cif_dir   : optional folder containing <stem>.cif files for residue counts
    max_files : optional cap for testing (e.g. set to 20 to run on a subset)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if max_files is not None:
        json_files = json_files[:max_files]

    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return

    print(f"Auditing {len(json_files)} JSON file(s) from {json_dir} ...")
    if cif_dir:
        print(f"CIF directory: {cif_dir}")
    else:
        print("No CIF directory provided — CIF residue counts will be skipped.")

    summary_rows    = []
    collision_rows  = []
    all_ins_values  = defaultdict(int)   # raw ins_code repr → count

    for json_path in tqdm(json_files, desc="Auditing JSONs"):
        res = audit_single_json(json_path, cif_dir=cif_dir)

        # accumulate insertion-code value counts
        for v in res["ins_code_values"]:
            all_ins_values[v] += 1

        summary_rows.append({
            "stem":                         res["stem"],
            "n_interactions_total":         res["n_interactions_total"],
            "n_interactions_std_aa_pairs":  res["n_interactions_std_aa"],
            "n_unique_residues_json":       res["n_unique_residues_json"],
            "n_unique_residues_json_noins": res["n_unique_residues_json_noins"],
            "n_collisions":                 res["n_collisions"],
            "n_cif_residues":               res["n_cif_residues"],
            "cif_vs_json_gap": (
                res["n_cif_residues"] - res["n_unique_residues_json"]
                if res["n_cif_residues"] is not None else None
            ),
            "cif_error": res["cif_error"],
            "ins_code_values_seen":         "|".join(sorted(res["ins_code_values"])),
        })

        collision_rows.extend(res["collision_details"])

    # ── write summary CSV ────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "ins_code_audit_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary CSV written: {summary_path}")

    # ── write collision CSV ───────────────────────────────────────────────────
    if collision_rows:
        coll_df = pd.DataFrame(collision_rows)
        coll_path = out_dir / "ins_code_collisions.csv"
        coll_df.to_csv(coll_path, index=False)
        print(f"Collision CSV written: {coll_path}  ({len(collision_rows)} collision(s))")
    else:
        print("No node-ID collisions detected — no collision CSV written.")

    # ── write ins_code value-counts CSV ─────────────────────────────────────
    vc_df = pd.DataFrame(
        sorted(all_ins_values.items(), key=lambda x: -x[1]),
        columns=["ins_code_repr", "n_json_files_with_this_value"],
    )
    vc_path = out_dir / "ins_code_value_counts.csv"
    vc_df.to_csv(vc_path, index=False)
    print(f"Insertion-code value counts CSV: {vc_path}")

    # ── console summary ───────────────────────────────────────────────────────
    n_with_collisions = (summary_df["n_collisions"] > 0).sum()
    n_total           = len(summary_df)

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"  JSON files audited         : {n_total:,}")
    print(f"  Files with collisions      : {n_with_collisions:,}  "
          f"({'%.1f' % (100 * n_with_collisions / n_total if n_total else 0)}%)")
    print(f"\n  Unique ins_code values seen (across all files):")
    for val, count in sorted(all_ins_values.items(), key=lambda x: -x[1]):
        print(f"    {val:<20} in {count:,} file(s)")

    if "n_cif_residues" in summary_df.columns:
        valid = summary_df["n_cif_residues"].notna()
        if valid.any():
            gaps = summary_df.loc[valid, "cif_vs_json_gap"]
            print(f"\n  CIF vs JSON residue gap (where CIF available):")
            print(f"    Mean gap  : {gaps.mean():.1f}")
            print(f"    Max gap   : {gaps.max():.0f}")
            print(f"    Min gap   : {gaps.min():.0f}")
            print(f"    Files with gap > 0 : {(gaps > 0).sum():,}")
            print(f"    Files with gap = 0 : {(gaps == 0).sum():,}")
            print(f"    Files with gap < 0 : {(gaps < 0).sum():,}  "
                  f"(negative = JSON has MORE residues than CIF — investigate!)")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    # ── Edit these paths ─────────────────────────────────────────────────────
    json_dir = DATA_DIR_3 / "test" / "JSONs"
    cif_dir  = DATA_DIR_3 / "test" / "CIFs_cleaned"
    out_dir  = FINAL_DIR / "audits" / "test"

    # Set to an integer to test on a small subset first (e.g. 20),
    # or None to run on all files.
    max_files = 20

    run_audit(
        json_dir  = json_dir,
        out_dir   = out_dir,
        cif_dir   = cif_dir,
        max_files = None,
    )


if __name__ == "__main__":
    main()
