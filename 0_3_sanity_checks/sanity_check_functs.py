import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from imports import *
 
 
def overlap(
    reference_csv = "reference_set.csv",
    test_csv      = "test_set.csv",
):
    ref_df  = pd.read_csv(reference_csv)
    test_df = pd.read_csv(test_csv)
 
    ref_ids  = set(ref_df["pdb_id"].str.strip().str.lower().unique())
    test_ids = set(test_df["pdb_id"].str.strip().str.lower().unique())
 
    print(f"Reference PDB IDs ({len(ref_ids)}): {sorted(ref_ids)}")
    print(f"Test PDB IDs      ({len(test_ids)}): {sorted(test_ids)}")
 
    overlap = ref_ids & test_ids
    if overlap:
        print(f"\n⚠  WARNING: {len(overlap)} PDB ID(s) appear in BOTH files:")
        for pid in sorted(overlap):
            print(f"   {pid.upper()}")
    else:
        print("\n  No overlap — all PDB IDs are unique to their respective file.")
 
def _load_pdb_ids(csv_path: Path) -> set:
    """Return a set of lowercase, stripped PDB IDs from a CSV's 'pdb_id' column."""
    df = pd.read_csv(csv_path)
    if "pdb_id" not in df.columns:
        raise ValueError(f"'pdb_id' column not found in {csv_path}. "
                         f"Columns present: {list(df.columns)}")
    return set(df["pdb_id"].str.strip().str.lower().unique())
 
 
def _find_file(pdb_id: str, source_dir: Path, suffix: str) -> Path | None:
    """
    Look for <PDB_ID><suffix> in source_dir, trying both upper- and lowercase
    stems (e.g. 1ABC.cif and 1abc.cif).  Returns the Path if found, else None.
    """
    for stem in (pdb_id.upper(), pdb_id.lower()):
        candidate = source_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None
 
 
def _copy_set(
    pdb_ids: set,
    cif_source_dir: Path,
    pdb_source_dir: Path,
    output_dir: Path,
    set_label: str,
) -> dict:
    """
    Copy .cif and .pdb files for every PDB ID in *pdb_ids* into *output_dir*.
 
    Returns a summary dict with counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
 
    copied_cif  = 0
    copied_pdb  = 0
    missing_cif = []
    missing_pdb = []
 
    for pid in sorted(pdb_ids):
        # --- .cif ---
        src_cif = _find_file(pid, cif_source_dir, ".cif")
        if src_cif:
            shutil.copy2(src_cif, output_dir / src_cif.name)
            copied_cif += 1
        else:
            missing_cif.append(pid.upper())
 
        # --- .pdb ---
        src_pdb = _find_file(pid, pdb_source_dir, ".pdb")
        if src_pdb:
            shutil.copy2(src_pdb, output_dir / src_pdb.name)
            copied_pdb += 1
        else:
            missing_pdb.append(pid.upper())
 
    return {
        "set":         set_label,
        "n_ids":       len(pdb_ids),
        "copied_cif":  copied_cif,
        "copied_pdb":  copied_pdb,
        "missing_cif": missing_cif,
        "missing_pdb": missing_pdb,
    }
 
 
def _print_summary(summary: dict) -> None:
    label = summary["set"]
    print(f"\n{'='*55}")
    print(f"  {label} set  ({summary['n_ids']} PDB IDs)")
    print(f"{'='*55}")
    print(f"  .cif files copied : {summary['copied_cif']}")
    print(f"  .pdb files copied : {summary['copied_pdb']}")
 
    if summary["missing_cif"]:
        print(f"  .cif not found    : {len(summary['missing_cif'])} ID(s)  "
              f"[silently skipped]")
 
    if summary["missing_pdb"]:
        print(f"  .pdb not found    : {len(summary['missing_pdb'])} ID(s)  "
              f"[silently skipped]")
 
def split_copy(
    reference_csv   = "reference_set.csv",
    test_csv        = "test_set.csv",
    cif_source_dir  = "raw_cif_files",
    pdb_source_dir  = "raw_pdb_files",
    output_ref_dir  = "output_reference_set",
    output_test_dir = "output_test_set",
):
    """
    Parameters
    ----------
    reference_csv   : str or Path
        CSV containing the reference-set PDB IDs (must have a 'pdb_id' column).
    test_csv        : str or Path
        CSV containing the test-set PDB IDs (must have a 'pdb_id' column).
    cif_source_dir  : str or Path
        Folder that holds all source .cif files (named <PDB_ID>.cif).
    pdb_source_dir  : str or Path
        Folder that holds all source .pdb files (named <PDB_ID>.pdb).
    output_ref_dir  : str or Path
        Destination folder for reference-set raw files (created if absent).
    output_test_dir : str or Path
        Destination folder for test-set raw files (created if absent).
    """
    reference_csv   = Path(reference_csv)
    test_csv        = Path(test_csv)
    cif_source_dir  = Path(cif_source_dir)
    pdb_source_dir  = Path(pdb_source_dir)
    output_ref_dir  = Path(output_ref_dir)
    output_test_dir = Path(output_test_dir)
 
    # ------------------------------------------------------------------
    # 1. Load PDB ID sets
    # ------------------------------------------------------------------
    ref_ids  = _load_pdb_ids(reference_csv)
    test_ids = _load_pdb_ids(test_csv)
 
    print(f"Reference PDB IDs : {len(ref_ids)}")
    print(f"Test PDB IDs      : {len(test_ids)}")
 
    # ------------------------------------------------------------------
    # 2. Overlap check (mirrors sanity_check_functs.py)
    # ------------------------------------------------------------------
    overlap = ref_ids & test_ids
    if overlap:
        print(f"\n⚠  WARNING: {len(overlap)} PDB ID(s) appear in BOTH CSVs:")
        for pid in sorted(overlap):
            print(f"   {pid.upper()}")
        print("   These IDs will be copied into BOTH output folders.")
    else:
        print("\n✓  No overlap — all PDB IDs are unique to their respective set.")
 
    # ------------------------------------------------------------------
    # 3. Copy files for each set
    # ------------------------------------------------------------------
    ref_summary  = _copy_set(ref_ids,  cif_source_dir, pdb_source_dir,
                             output_ref_dir,  "Reference")
    test_summary = _copy_set(test_ids, cif_source_dir, pdb_source_dir,
                             output_test_dir, "Test")
 
    # ------------------------------------------------------------------
    # 4. Print summaries
    # ------------------------------------------------------------------
    _print_summary(ref_summary)
    _print_summary(test_summary)
 
    print(f"\n{'='*55}")
    print(f"  Output folders")
    print(f"{'='*55}")
    print(f"  Reference : {output_ref_dir.resolve()}")
    print(f"  Test      : {output_test_dir.resolve()}")
    print()

if __name__ == "__main__":

    # ensure no overlaps in PDB_IDs
    # overlap(
    #     reference_csv = DATA_DIR_3 / "ref_standardized.csv",
    #     test_csv      = DATA_DIR_3 / "test_standardized.csv",
    # )

    # clean data up into two sets, test and reference
    split_copy(
        reference_csv   = DATA_DIR_3 / "ref_standardized.csv",
        test_csv        = DATA_DIR_3 / "test_standardized.csv",
        cif_source_dir  = DATA_DIR_2 / "enzymes_CIF_unflagged" / "CIFs",
        pdb_source_dir  = DATA_DIR_2 / "old_PISCES" / "PDBs",
        output_ref_dir  = DATA_DIR_3 / "ref_set_raw",
        output_test_dir = DATA_DIR_3 / "test_set_raw",
    )
 