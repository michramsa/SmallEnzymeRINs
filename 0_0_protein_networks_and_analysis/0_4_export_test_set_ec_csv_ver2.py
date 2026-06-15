"""
0_4_export_test_set_ec_csv_ver1.py

Exports a CSV containing one row per test-set protein with its PDB ID and
priority EC classification. Mirrors the filtering logic used in
0_4_enzyme_distribution_ver2.py (updated 6/1/26 block):

  1. Load unique PDB IDs from test_standardized.csv.
  2. Filter resolved_prio_ec.csv to those IDs.
  3. Deduplicate on pdb_id (keep="first") so each protein appears once.
  4. Write pdb_id + prio_ec_number (and optionally all columns) to CSV.

Output columns (default):
    pdb_id, prio_ec_number, ec_class_name

ec_class_name is derived from the first digit of prio_ec_number using the
same EC_CLASSES mapping as 0_4_enzyme_distribution_ver2.py
(e.g. "Oxidoreductases", "Transferases", ...).

Set EXPORT_ALL_COLUMNS = True in main() to include every column from
resolved_prio_ec.csv in addition to ec_class_name.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *   # provides pd, Path, DATA_DIR_2, etc.


# ---------------------------------------------------------------------------
# EC class lookup  (mirrors 0_4_enzyme_distribution_ver2.py)
# ---------------------------------------------------------------------------

EC_CLASSES = {
    1: "Oxidoreductases",
    2: "Transferases",
    3: "Hydrolases",
    4: "Lyases",
    5: "Isomerases",
    6: "Ligases",
    7: "Translocases",
}


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def export_test_set_ec(
    test_standardized_csv: Path,
    resolved_prio_ec_csv: Path,
    output_csv: Path,
    export_all_columns: bool = False,
) -> pd.DataFrame:
    """
    Filter resolved_prio_ec.csv to the PDB IDs present in test_standardized.csv
    and save the result as a CSV.

    Parameters
    ----------
    test_standardized_csv : Path
        Residue-level CSV produced by the fig1-recreation pipeline.
        Must contain a 'pdb_id' column.
    resolved_prio_ec_csv : Path
        Master priority-EC table.  Must contain 'pdb_id' and 'prio_ec_number'.
    output_csv : Path
        Destination path for the exported CSV.
    export_all_columns : bool
        If True, retain all columns from resolved_prio_ec.csv (plus ec_class_name).
        If False (default), export only ['pdb_id', 'prio_ec_number', 'ec_class_name'].

    Returns
    -------
    pd.DataFrame
        The filtered, deduplicated DataFrame that was written to disk.
    """
    # ── 1. collect unique test-set PDB IDs ───────────────────────────────
    df_test_std = pd.read_csv(test_standardized_csv, usecols=["pdb_id"])
    test_pdb_ids = set(df_test_std["pdb_id"].str.upper().unique())
    print(f"Unique PDB IDs in test set: {len(test_pdb_ids)}")

    # ── 2. load the priority-EC table ────────────────────────────────────
    df_prio = pd.read_csv(resolved_prio_ec_csv)
    prio_ids = set(df_prio["pdb_id"].str.upper().unique())

    # ── 3. diagnose unmatched IDs ─────────────────────────────────────────
    unmatched = test_pdb_ids - prio_ids
    if unmatched:
        print(
            f"WARNING: {len(unmatched)} test-set PDB ID(s) not found "
            f"in resolved_prio_ec.csv:"
        )
        for uid in sorted(unmatched):
            print(f"  {uid}")
    else:
        print("All test-set PDB IDs matched in resolved_prio_ec.csv.")

    # ── 4. filter + deduplicate ───────────────────────────────────────────
    df_out = (
        df_prio[df_prio["pdb_id"].str.upper().isin(test_pdb_ids)]
        .drop_duplicates(subset="pdb_id", keep="first")
        .reset_index(drop=True)
    )
    print(f"Proteins matched and written to CSV: {len(df_out)}")

    # ── 5. derive ec_class_name from first digit of prio_ec_number ────────
    ec_first_digit = (
        df_out["prio_ec_number"]
        .astype(str)
        .str.strip()
        .str.split(".")
        .str[0]
    )
    ec_first_digit = pd.to_numeric(ec_first_digit, errors="coerce")
    df_out["ec_class_name"] = ec_first_digit.map(EC_CLASSES)

    n_unmapped = df_out["ec_class_name"].isna().sum()
    if n_unmapped > 0:
        print(
            f"NOTE: {n_unmapped} row(s) could not be mapped to an EC class name "
            f"(partial or missing EC number); ec_class_name will be NaN for those rows."
        )

    # ── 6. optionally trim to key columns ────────────────────────────────
    if not export_all_columns:
        cols_to_keep = ["pdb_id", "prio_ec_number", "ec_class_name"]
        missing = [c for c in cols_to_keep if c not in df_out.columns]
        if missing:
            raise ValueError(
                f"Expected columns not found in resolved_prio_ec.csv: {missing}\n"
                f"Available columns: {df_out.columns.tolist()}"
            )
        df_out = df_out[cols_to_keep]

    # ── 7. write CSV ──────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"CSV saved to: {output_csv}")

    return df_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Input paths ───────────────────────────────────────────────────────
    TEST_STANDARDIZED_CSV = (
        DATA_DIR_2
        / "figures"
        / "controlled_fig1_recreation_ver0.7"
        / "test_standardized.csv"
    )

    RESOLVED_PRIO_EC_CSV = DATA_DIR_2 / "resolved_prio_ec.csv"

    # ── Output path ───────────────────────────────────────────────────────
    OUTPUT_CSV = (
        DATA_DIR_2
        / "figures"
        / "controlled_fig1_recreation_ver0.7"
        / "test_set_ec_classifications.csv"
    )

    # ── Options ───────────────────────────────────────────────────────────
    # False  → export only pdb_id, prio_ec_number, ec_class_name
    # True   → export all columns from resolved_prio_ec.csv + ec_class_name
    EXPORT_ALL_COLUMNS = False

    # ── Run ───────────────────────────────────────────────────────────────
    df_result = export_test_set_ec(
        test_standardized_csv=TEST_STANDARDIZED_CSV,
        resolved_prio_ec_csv=RESOLVED_PRIO_EC_CSV,
        output_csv=OUTPUT_CSV,
        export_all_columns=EXPORT_ALL_COLUMNS,
    )

    print("\nPreview of exported CSV:")
    print(df_result.head(10).to_string(index=False))
