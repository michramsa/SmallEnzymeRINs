"""
0_6_biolip_lookup_ver1.py
=========================
Check a test-set CSV of single-chain enzymes against the raw BioLiP
annotation file and produce:

  1. biolip_matched_rows.csv
       All BioLiP rows whose PDB ID appears in the test set AND whose
       receptor chain is the amino-acid chain identified from the CIF.

  2. biolip_per_pdb_summary.csv
       One row per test-set entry summarising:
         - whether any BioLiP match was found
         - the matched receptor chain(s)
         - number of BioLiP annotation rows
         - unique EC numbers
         - unique ligand IDs
         - whether catalytic-site residues are annotated
         - whether binding-site residues are annotated

  3. Console output flagging every test-set PDB ID not found in BioLiP.

Chain identification
--------------------
For each entry the script parses the cleaned CIF (via gemmi then
BioPython fallback) to find every chain that contains at least one
standard amino-acid ATOM residue.  Only those chains are used when
filtering BioLiP receptor-chain column (col 02).  This excludes DNA,
RNA, and water chains automatically.

Inputs (edit in main())
-----------------------
  TEST_CSV   : CSV with columns pdb_id, stem, n_chains, source_file, …
               Only rows where n_chains == 1 are processed.
  CIF_DIR    : Directory that holds <stem>_clean.cif files.
               (source_file column is used as fallback if CIF_DIR is None)
  BIOLIP_TXT : DATA_DIR_3 / "BioLiP.txt"  (tab-separated)
  OUT_DIR    : Directory for the two output CSVs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imports import *

# ---------------------------------------------------------------------------
# Standard amino acids — matches 0_4_enzyme_distribution_ver4.py
# ---------------------------------------------------------------------------
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL",   # selenocysteine / pyrrolysine
}

# ---------------------------------------------------------------------------
# BioLiP column names (21 tab-separated columns)
# ---------------------------------------------------------------------------
BIOLIP_COLS = [
    "pdb_id",               # 01
    "receptor_chain",       # 02
    "resolution",           # 03
    "binding_site_code",    # 04
    "ligand_id",            # 05
    "ligand_chain",         # 06
    "ligand_serial",        # 07
    "binding_residues_pdb", # 08
    "binding_residues_seq", # 09
    "catalytic_residues_pdb",  # 10
    "catalytic_residues_seq",  # 11
    "ec_number",            # 12
    "go_terms",             # 13
    "affinity_literature",  # 14
    "affinity_moad",        # 15
    "affinity_pdbbind",     # 16
    "affinity_bindingdb",   # 17
    "uniprot_id",           # 18
    "pubmed_id",            # 19
    "ligand_seq_nums",      # 20
    "receptor_sequence",    # 21
]


# ===========================================================================
# CIF helpers
# ===========================================================================

def get_protein_chains_from_cif(cif_path: Path) -> tuple[set[str], str | None]:
    """
    Return the set of chain IDs that contain at least one standard amino-acid
    ATOM residue in *cif_path*.

    Tries gemmi first (fast); falls back to BioPython.

    Returns
    -------
    (chains, error_msg)
        chains    : set of chain ID strings; empty set on failure.
        error_msg : None on success, description string on failure.
    """
    # --- gemmi (preferred) ---
    try:
        import gemmi  # type: ignore
        st = gemmi.read_structure(str(cif_path))
        chains: set[str] = set()
        for model in st:
            for chain in model:
                for res in chain:
                    if (
                        res.entity_type == gemmi.EntityType.Polymer
                        and res.name in STANDARD_AA
                    ):
                        chains.add(chain.name)
                        break   # one hit per chain is enough
        if chains:
            return chains, None
        return set(), "gemmi: 0 standard-AA chains found"
    except ImportError:
        pass
    except Exception as exc:
        return set(), f"gemmi: {exc}"

    # --- BioPython fallback ---
    try:
        from Bio.PDB import MMCIFParser
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(cif_path.stem, str(cif_path))
        chains = set()
        for model in structure:
            for chain in model:
                for res in chain:
                    if (
                        res.get_resname().strip() in STANDARD_AA
                        and res.get_id()[0] == " "   # ATOM, not HETATM
                    ):
                        chains.add(chain.id)
                        break
        if chains:
            return chains, None
        return set(), "BioPython: 0 standard-AA chains found"
    except Exception as exc:
        return set(), f"BioPython: {exc}"


# ===========================================================================
# BioLiP loader
# ===========================================================================

def load_biolip(biolip_txt: Path) -> pd.DataFrame:
    """
    Load the raw BioLiP tab-separated annotation file.

    PDB IDs are stored in lower-case (as BioLiP delivers them).
    A 'pdb_id_upper' convenience column is added for case-insensitive joins.

    Returns
    -------
    pd.DataFrame with BIOLIP_COLS columns.
    """
    print(f"Loading BioLiP from: {biolip_txt}")
    df = pd.read_csv(
        biolip_txt,
        sep="\t",
        header=None,
        names=BIOLIP_COLS,
        dtype=str,
        low_memory=False,
    )
    df["pdb_id"]       = df["pdb_id"].str.strip().str.lower()
    df["pdb_id_upper"] = df["pdb_id"].str.upper()
    df["receptor_chain"] = df["receptor_chain"].str.strip()
    print(f"  BioLiP rows loaded : {len(df):,}")
    print(f"  Unique PDB IDs     : {df['pdb_id'].nunique():,}")
    return df


# ===========================================================================
# Main lookup logic
# ===========================================================================

def lookup_biolip(
    test_csv:   Path,
    cif_dir:    Path | None,
    biolip_txt: Path,
    out_dir:    Path,
    n_chains_filter: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Core lookup routine.

    Parameters
    ----------
    test_csv        : Path to the test-set CSV.
    cif_dir         : Directory holding <stem>.cif files. When None, the
                      script falls back to source_file column in test_csv.
    biolip_txt      : Path to the raw BioLiP .txt annotation file.
    out_dir         : Output directory for both CSVs.
    n_chains_filter : Only process rows where n_chains == this value (default 1).

    Returns
    -------
    (matched_df, summary_df)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load test CSV ────────────────────────────────────────────────────
    test_df = pd.read_csv(test_csv, dtype=str)
    print(f"\nTest CSV rows (all)      : {len(test_df):,}")

    # filter to single-chain entries
    test_df["n_chains_int"] = pd.to_numeric(test_df["n_chains"], errors="coerce")
    test_df = test_df[test_df["n_chains_int"] == n_chains_filter].copy()
    print(f"Test CSV rows (n_chains={n_chains_filter}): {len(test_df):,}")

    test_df["pdb_id_upper"] = test_df["pdb_id"].str.strip().str.upper()
    test_ids_upper = set(test_df["pdb_id_upper"])

    # ── 2. Load BioLiP ──────────────────────────────────────────────────────
    biolip_df = load_biolip(biolip_txt)

    # quick overlap check before chain filtering
    biolip_ids_upper = set(biolip_df["pdb_id_upper"])
    overlap_ids      = test_ids_upper & biolip_ids_upper
    missing_ids      = test_ids_upper - biolip_ids_upper
    print(f"\nTest-set IDs           : {len(test_ids_upper):,}")
    print(f"Found in BioLiP (any)  : {len(overlap_ids):,}")
    print(f"NOT found in BioLiP    : {len(missing_ids):,}")

    # ── 3. Identify protein chains per CIF ──────────────────────────────────
    print("\nIdentifying protein chains from CIF files …")
    pdb_to_protein_chains: dict[str, set[str]] = {}
    cif_errors: dict[str, str] = {}

    for _, row in test_df.iterrows():
        pdb_upper = row["pdb_id_upper"]

        # resolve CIF path
        cif_path: Path | None = None
        if cif_dir is not None:
            stem     = str(row.get("stem", row["pdb_id"])).strip()
            candidate = cif_dir / f"{stem}.cif"
            if candidate.exists():
                cif_path = candidate

        # fallback: source_file column
        if cif_path is None and "source_file" in row and pd.notna(row["source_file"]):
            candidate = Path(str(row["source_file"]).strip())
            if candidate.exists():
                cif_path = candidate

        if cif_path is None:
            cif_errors[pdb_upper] = "CIF file not found"
            pdb_to_protein_chains[pdb_upper] = set()
            continue

        chains, err = get_protein_chains_from_cif(cif_path)
        if err:
            cif_errors[pdb_upper] = err
            pdb_to_protein_chains[pdb_upper] = set()
        else:
            pdb_to_protein_chains[pdb_upper] = chains

    n_chain_ok  = sum(1 for c in pdb_to_protein_chains.values() if c)
    n_chain_err = sum(1 for c in pdb_to_protein_chains.values() if not c)
    print(f"  Protein chains resolved : {n_chain_ok:,} / {len(test_df):,}")
    if n_chain_err:
        print(f"  Chain-resolution errors : {n_chain_err:,}")
        for pid, err in cif_errors.items():
            print(f"    {pid}: {err}")

    # ── 4. Filter BioLiP to test-set PDB IDs + matching protein chains ──────
    biolip_subset = biolip_df[biolip_df["pdb_id_upper"].isin(test_ids_upper)].copy()

    def _chain_matches(row_bl: pd.Series) -> bool:
        pdb_upper   = row_bl["pdb_id_upper"]
        rec_chain   = row_bl["receptor_chain"]
        prot_chains = pdb_to_protein_chains.get(pdb_upper, set())
        # if CIF parsing failed, accept any receptor chain (don't silently drop)
        if not prot_chains:
            return True
        return rec_chain in prot_chains

    matched_mask = biolip_subset.apply(_chain_matches, axis=1)
    matched_df   = biolip_subset[matched_mask].copy()

    print(f"\nBioLiP rows after PDB-ID filter            : {len(biolip_subset):,}")
    print(f"BioLiP rows after protein-chain filter     : {len(matched_df):,}")
    print(f"Unique PDB IDs in matched rows             : {matched_df['pdb_id_upper'].nunique():,}")

    # ── 5. Build per-PDB summary ─────────────────────────────────────────────
    summary_records = []
    for _, row in test_df.iterrows():
        pdb_upper  = row["pdb_id_upper"]
        pdb_id_raw = str(row["pdb_id"]).strip()
        stem       = str(row.get("stem", pdb_id_raw)).strip()

        hits = matched_df[matched_df["pdb_id_upper"] == pdb_upper]
        found = len(hits) > 0

        # unique matched receptor chains
        matched_chains = sorted(hits["receptor_chain"].dropna().unique()) if found else []

        # protein chains from CIF
        prot_chains = sorted(pdb_to_protein_chains.get(pdb_upper, set()))

        # EC numbers (may be comma-separated per row; flatten)
        ec_vals = set()
        for ec_raw in hits["ec_number"].dropna():
            for ec in str(ec_raw).split(","):
                ec = ec.strip()
                if ec and ec not in ("", "-", "nan"):
                    ec_vals.add(ec)

        # unique ligand IDs
        lig_ids = sorted(
            hits["ligand_id"].dropna().unique()
        ) if found else []

        # catalytic residues annotated?
        has_catalytic = (
            hits["catalytic_residues_pdb"]
            .dropna()
            .apply(lambda v: str(v).strip() not in ("", "-", "nan"))
            .any()
        ) if found else False

        # binding residues annotated?
        has_binding = (
            hits["binding_residues_pdb"]
            .dropna()
            .apply(lambda v: str(v).strip() not in ("", "-", "nan"))
            .any()
        ) if found else False

        summary_records.append({
            "pdb_id":              pdb_id_raw,
            "pdb_id_upper":        pdb_upper,
            "stem":                stem,
            "n_chains":            row["n_chains"],
            "protein_chains_cif":  ";".join(prot_chains) if prot_chains else "",
            "cif_error":           cif_errors.get(pdb_upper, ""),
            "found_in_biolip":     found,
            "matched_receptor_chains": ";".join(matched_chains),
            "n_biolip_rows":       len(hits),
            "ec_numbers":          ";".join(sorted(ec_vals)),
            "ligand_ids":          ";".join(lig_ids),
            "has_catalytic_residues": has_catalytic,
            "has_binding_residues":   has_binding,
        })

    summary_df = pd.DataFrame(summary_records)

    # ── 6. Console report: missing IDs ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST-SET ENTRIES NOT FOUND IN BioLiP")
    print("=" * 60)
    not_found = summary_df[~summary_df["found_in_biolip"]]
    if not_found.empty:
        print("  All test-set entries have at least one BioLiP match.")
    else:
        print(f"  {len(not_found):,} entry/entries with no BioLiP match:\n")
        for _, r in not_found.iterrows():
            cif_note = f"  [CIF error: {r['cif_error']}]" if r["cif_error"] else ""
            in_biolip_raw = r["pdb_id_upper"] in biolip_ids_upper
            reason = "(PDB ID absent from BioLiP)" if not in_biolip_raw \
                     else "(PDB ID present but no protein-chain match)"
            print(f"    {r['pdb_id_upper']}  {reason}{cif_note}")

    # ── 7. Summary stats ─────────────────────────────────────────────────────
    n_found = summary_df["found_in_biolip"].sum()
    n_total = len(summary_df)
    print(f"\nSummary")
    print(f"  Test entries processed          : {n_total:,}")
    print(f"  Entries with BioLiP match       : {n_found:,} ({100*n_found/n_total:.1f}%)")
    print(f"  Entries WITHOUT BioLiP match    : {n_total - n_found:,}")
    print(f"  Total matched BioLiP rows       : {len(matched_df):,}")
    print(f"  Entries with catalytic residues : "
          f"{summary_df['has_catalytic_residues'].sum():,}")
    print(f"  Entries with binding residues   : "
          f"{summary_df['has_binding_residues'].sum():,}")

    # ── 8. Save outputs ───────────────────────────────────────────────────────
    matched_path = out_dir / "biolip_matched_rows.csv"
    summary_path = out_dir / "biolip_per_pdb_summary.csv"

    matched_df.to_csv(matched_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\nOutputs written to: {out_dir}")
    print(f"  {matched_path.name}   ({len(matched_df):,} rows)")
    print(f"  {summary_path.name}  ({len(summary_df):,} rows)")

    return matched_df, summary_df


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    TEST_CSV   = FIGS_DIR / "enzyme_dist_no_mc" / "no_mc_size_dist_enzyme_test_set.csv"
    CIF_DIR    = DATA_DIR_3 / "test" / "CIFs_cleaned"
    BIOLIP_TXT = FINAL_DIR / "BioLiP.txt"
    OUT_DIR    = DATA_DIR_3 / "test" / "biolip_lookup"

    # Only process entries where n_chains equals this value
    N_CHAINS_FILTER = 1
    # ─────────────────────────────────────────────────────────────────────────

    lookup_biolip(
        test_csv        = TEST_CSV,
        cif_dir         = CIF_DIR,
        biolip_txt      = BIOLIP_TXT,
        out_dir         = OUT_DIR,
        n_chains_filter = N_CHAINS_FILTER,
    )


if __name__ == "__main__":
    main()
