# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

from glob import glob

# Directory containing RIN graphml files (e.g. "1A4V_clean.graphml")
RIN_DIR = UF_ENZYMES_DIR / "RINs"

# Output path for the final merged CSV
OUTPUT_CSV = ENZYMES_DIR / "info_CSVs" / "consolidated_data.csv"

# note to self - use with resolved_prio_ec.csv
def get_pdb_ids_from_rin_dir(rin_dir: Path) -> list[str]:
    """
    Scans the RIN directory for files matching the pattern *_clean.graphml
    and extracts the PDB ID (the part before '_clean.graphml').

    Returns a list of uppercase PDB IDs, e.g. ['1A4V', '2LW5', ...]
    """
    pattern = str(rin_dir / "*_clean.graphml")
    files = glob(pattern)

    pdb_ids = []
    for f in files:
        stem = Path(f).stem          # e.g. "1A4V_clean"
        pdb_id = stem.replace("_clean", "").upper()
        pdb_ids.append(pdb_id)

    print(f"[RIN DIR] Found {len(pdb_ids)} PDB IDs.")
    return pdb_ids


# note to self - use with cif_inspection.csv
def extract_enzyme_features(filepath: str | Path, pdb_ids: list[str]) -> pd.DataFrame:
    """
    Extracts selected columns from the enzyme feature CSV for PDB IDs in pdb_ids.

    Expected columns in source file:
        pdb_id, chain, ec_number, prio_ec_number, prio_ec_source,
        oligomeric_state, complex_type, n_chains_asymmetric_unit,
        n_chains_biological_assembly, original_header,
        n_functional_residues, functional_residues

    Columns retained:
        pdb_id, chain, original_header, prio_ec_number,
        complex_type, functional_residues

    Parameters
    ----------
    filepath : path to the source CSV file
    pdb_ids  : list of PDB IDs to keep (matched case-insensitively)

    Returns
    -------
    pd.DataFrame with the filtered and selected columns
    """
    COLS_TO_KEEP = [
        "pdb_id",
        "chain",
        "original_header",
        "prio_ec_number",
        "complex_type",
        "functional_residues",
    ]

    df = pd.read_csv(filepath, dtype=str)

    # Normalise pdb_id to uppercase for consistent matching
    df["pdb_id"] = df["pdb_id"].str.upper()
    pdb_ids_upper = [p.upper() for p in pdb_ids]

    # Filter to only the PDB IDs we care about
    df = df[df["pdb_id"].isin(pdb_ids_upper)]

    # Keep only the columns we need
    df = df[COLS_TO_KEEP].reset_index(drop=True)

    print(f"[enzyme_features] {len(df)} rows retained from {Path(filepath).name}")
    return df

def merge_all(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Left-joins all extracted DataFrames on pdb_id.
    The first DataFrame in the list is used as the base.

    If a source has one row per (pdb_id, chain), the merge
    key is automatically extended to (pdb_id, chain) when
    both DataFrames contain a 'chain' column.
    """
    if not dataframes:
        raise ValueError("No DataFrames provided to merge.")

    merged = dataframes[0]

    for df in dataframes[1:]:
        # Use (pdb_id, chain) as key when both frames have chain
        shared_keys = ["pdb_id"]
        if "chain" in merged.columns and "chain" in df.columns:
            shared_keys.append("chain")

        merged = merged.merge(df, on=shared_keys, how="left")

    print(f"[merge_all] Final shape: {merged.shape}")
    return merged

def extract_cif_metadata(filepath: str | Path, pdb_ids: list[str]) -> pd.DataFrame:
    """
    Extracts selected columns from the CIF metadata CSV for PDB IDs in pdb_ids.

    Expected columns in source file:
        relative_path, file, experimental_method, method_category, resolution,
        total_atoms, has_altlocs, altloc_atom_count, altloc_residue_count,
        altloc_labels, error

    Columns retained:
        pdb_id, experimental_method, resolution

    Parameters
    ----------
    filepath : path to the source CSV file
    pdb_ids  : list of PDB IDs to keep (matched case-insensitively)

    Returns
    -------
    pd.DataFrame with the filtered and selected columns
    """
    COLS_TO_KEEP = ["file", "experimental_method", "resolution"]

    df = pd.read_csv(filepath, dtype=str)

    # Derive pdb_id from the filename stem (e.g. "11BA.cif" -> "11BA")
    df["pdb_id"] = df["file"].apply(lambda x: Path(x).stem.upper())
    pdb_ids_upper = [p.upper() for p in pdb_ids]

    df = df[df["pdb_id"].isin(pdb_ids_upper)]
    df = df[["pdb_id", "experimental_method", "resolution"]].reset_index(drop=True)

    print(f"[cif_metadata] {len(df)} rows retained from {Path(filepath).name}")
    return df

def filter_by_resolution(filepath: str | Path, max_resolution: float = 3.0) -> pd.DataFrame:
    """
    Filters the consolidated CSV by resolution threshold.

    Rows with a resolution value above max_resolution are removed.
    Rows with no resolution value (NaN/empty) are kept regardless.

    Parameters
    ----------
    filepath       : path to the consolidated CSV file
    max_resolution : maximum allowed resolution in angstroms (default 3.0)

    Returns
    -------
    pd.DataFrame with low-resolution and no-resolution entries removed
    """
    df = pd.read_csv(filepath, dtype=str)

    # Convert resolution to numeric — non-numeric/empty entries become NaN
    df["resolution"] = pd.to_numeric(df["resolution"], errors="coerce")

    # Keep rows where resolution is NaN OR within the threshold
    mask = df["resolution"].isna() | (df["resolution"] <= max_resolution)
    df = df[mask].reset_index(drop=True)

    print(f"[filter_by_resolution] {len(df)} rows retained after resolution filter.")
    return df

def main():
    # 5/5/26
    # created consolidated_data.csv within info_CSVs
    # pdb_ids = get_pdb_ids_from_rin_dir(RIN_DIR)
    # enzyme_feats = ENZYMES_DIR / "info_CSVs" / "resolved_prio_ec.csv"
    # cif_feats = ENZYMES_DIR / "info_CSVs" / "cif_inspection.csv"
    # df_enzyme = extract_enzyme_features(enzyme_feats, pdb_ids)
    # print(df_enzyme)
    # df_cif = extract_cif_metadata(cif_feats, pdb_ids)
    # print(df_cif)
    # final_df = merge_all([df_enzyme, df_cif])
    # final_df.to_csv(OUTPUT_CSV, index=False)
    # print(f"[main] Saved consolidated CSV to {OUTPUT_CSV}")

    # 5/5/26 - filter by the resolution
    consolidated_csv = ENZYMES_DIR / "info_CSVs" / "consolidated_data.csv"
    df_resolution = filter_by_resolution(consolidated_csv)
    output = ENZYMES_DIR / "info_CSVs" / "filtered_consolidated.csv"
    df_resolution.to_csv(output, index=False)

if __name__ == "__main__":
    main()
