# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *


def filter_xray_csv(input_csv_path, output_csv_path):
    """
    Reads the input CSV file, filters rows where method == 'XRAY',
    and writes the filtered results to a new CSV file.

    Args:
        input_csv_path (str or Path): Path to the original CSV file.
        output_csv_path (str or Path): Path where the filtered CSV will be saved.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    input_csv_path = Path(input_csv_path)
    output_csv_path = Path(output_csv_path)

    df = pd.read_csv(input_csv_path)

    # filtering for only xray - run 1
    # filtered_df = df[df["method"] == "XRAY"].reset_index(drop=True)

    # excluded - run 2
    filtered_df = df[df["species"] != "EXCLUDE"].reset_index(drop=True)

    filtered_df.to_csv(output_csv_path, index=False)

    print(f"[filter_xray_csv] Total entries      : {len(df)}")
    print(f"[filter_xray_csv] Filtered entries kept  : {len(filtered_df)}")
    print(f"[filter_xray_csv] Saved to            : {output_csv_path}")

    return filtered_df


def diagnose_paths(
    source_base_dir,
    filtered_csv_path,
    cif_folder_name="CIFs",
    pdb_folder_name="PDBs_cleaned",
    graphml_folder_name="GraphMLs",
):
    """
    Prints a diagnostic report showing:
      - Whether each source folder exists and how many files are in it
      - The exact file paths the script will look for, for the first 5 CSV entries
      - Whether those files actually exist on disk

    Run this if copy_filtered_files() copies nothing or reports missing files.

    Args:
        source_base_dir     (str or Path): Directory containing the three source folders.
        filtered_csv_path   (str or Path): Path to the filtered CSV.
        cif_folder_name     (str): Name of the source CIF folder.
        pdb_folder_name     (str): Name of the source PDB folder.
        graphml_folder_name (str): Name of the source GraphML folder.
    """
    source_base_dir   = Path(source_base_dir)
    filtered_csv_path = Path(filtered_csv_path)

    print("=" * 60)
    print("PATH DIAGNOSTICS")
    print("=" * 60)
    print(f"source_base_dir : {source_base_dir.resolve()}")
    print(f"  exists?       : {source_base_dir.exists()}\n")

    folder_specs = [
        (cif_folder_name,     "_clean.cif"),
        (pdb_folder_name,     "_clean.pdb"),
        (graphml_folder_name, "_clean.graphml"),
    ]

    for folder_name, suffix in folder_specs:
        folder = source_base_dir / folder_name
        print(f"  Folder : {folder.resolve()}")
        print(f"  exists?: {folder.exists()}")
        if folder.exists():
            files = sorted(folder.iterdir())
            print(f"  # files: {len(files)}")
            print(f"  Sample filenames: {[f.name for f in files[:5]]}")
        print()

    print(f"filtered_csv_path : {filtered_csv_path.resolve()}")
    print(f"  exists?         : {filtered_csv_path.exists()}\n")

    if not filtered_csv_path.exists():
        print("  ERROR: filtered CSV not found — run filter_xray_csv() first.")
        return

    df = pd.read_csv(filtered_csv_path)
    print(f"CSV rows: {len(df)}")
    print("Checking first 5 IDs:\n")

    for raw_id in df["id"].head(5):
        pdb_id = str(raw_id)[:4]
        print(f"  raw_id={raw_id!r}  ->  pdb_id used={pdb_id!r}")
        for folder_name, suffix in folder_specs:
            expected = source_base_dir / folder_name / (pdb_id + suffix)
            print(f"    {'EXISTS' if expected.exists() else 'MISSING'} : {expected.resolve()}")
        print()

    print("=" * 60)
    print("END DIAGNOSTICS")
    print("=" * 60)


def copy_filtered_files(
    filtered_csv_path,
    source_base_dir,
    cif_folder_name="CIFs",
    pdb_folder_name="PDBs_cleaned",
    graphml_folder_name="GraphMLs",
    cif_out_folder_name="CIFs_filtered",
    pdb_out_folder_name="PDBs_cleaned_filtered",
    graphml_out_folder_name="GraphMLs_filtered",
):
    """
    Reads the filtered CSV, iterates through the PDB IDs (first 4 characters
    of the 'id' column), and copies the matching CIF, PDB, and GraphML files
    into three new output folders.

    Expected filename convention (using id '5D8VA' as an example):
        CIFs         -> <cif_folder_name>/5D8V_clean.cif
        PDBs_cleaned -> <pdb_folder_name>/5D8V_clean.pdb
        GraphMLs     -> <graphml_folder_name>/5D8V_clean.graphml

    Args:
        filtered_csv_path       (str or Path): Path to the filtered CSV produced by filter_xray_csv().
        source_base_dir         (str or Path): Directory that contains the three source folders.
        cif_folder_name         (str): Name of the source CIF folder.
        pdb_folder_name         (str): Name of the source PDB folder.
        graphml_folder_name     (str): Name of the source GraphML folder.
        cif_out_folder_name     (str): Name for the output CIF folder.
        pdb_out_folder_name     (str): Name for the output PDB folder.
        graphml_out_folder_name (str): Name for the output GraphML folder.
    """
    filtered_csv_path = Path(filtered_csv_path)
    source_base_dir   = Path(source_base_dir)

    # Define source folders
    cif_src     = source_base_dir / cif_folder_name
    pdb_src     = source_base_dir / pdb_folder_name
    graphml_src = source_base_dir / graphml_folder_name

    # Define and create output folders
    cif_dst     = source_base_dir / cif_out_folder_name
    pdb_dst     = source_base_dir / pdb_out_folder_name
    graphml_dst = source_base_dir / graphml_out_folder_name

    for folder in (cif_dst, pdb_dst, graphml_dst):
        folder.mkdir(parents=True, exist_ok=True)
        print(f"[copy_filtered_files] Output folder ready: {folder}")

    df = pd.read_csv(filtered_csv_path)

    # File extension / suffix mappings
    file_specs = [
        (cif_src,     cif_dst,     "_clean.cif",     "CIF"),
        (pdb_src,     pdb_dst,     "_clean.pdb",     "PDB"),
        (graphml_src, graphml_dst, "_clean.graphml", "GraphML"),
    ]

    copied_counts = {"CIF": 0, "PDB": 0, "GraphML": 0}
    missing = []

    for raw_id in df["id"]:
        # The PDB ID is the first 4 characters of the 'id' column
        pdb_id = str(raw_id)[:4]

        for src_dir, dst_dir, suffix, label in file_specs:
            filename = pdb_id + suffix
            src_file = src_dir / filename
            dst_file = dst_dir / filename

            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                copied_counts[label] += 1
            else:
                missing.append((pdb_id, label, str(src_file)))

    # Summary
    print("\n[copy_filtered_files] Copy summary:")
    for label, count in copied_counts.items():
        print(f"  {label:>8} files copied: {count}")

    if missing:
        print(f"\n[copy_filtered_files] WARNING — {len(missing)} file(s) not found:")
        for pdb_id, label, path in missing:
            print(f"  [{label}] {pdb_id} -> {path}")
    else:
        print("[copy_filtered_files] All files found and copied successfully.")


if __name__ == "__main__":
    ### rerun on 5/7/2026, i had exclude tags (likely for membrane proteins)
    INPUT_CSV       = DATA_DIR / "old_PISCES" / "PISCES_xray.csv"
    OUTPUT_CSV      = DATA_DIR / "old_PISCES" / "PISCES_xray_v2.csv"
    SOURCE_BASE_DIR = DATA_DIR / "old_PISCES"

    # Optional: change source/destination folder names if yours differ
    # run 1 was CIFs, PDBs_cleaned, GraphMLs
    # run 2 was the output from run 1, so CIFs_filtered, and so on
    CIF_FOLDER_NAME     = "CIFs_filtered"
    PDB_FOLDER_NAME     = "PDBs_cleaned_filtered"
    GRAPHML_FOLDER_NAME = "GraphMLs_filtered"

    CIF_OUT_FOLDER_NAME     = "CIFs_filtered_v2"
    PDB_OUT_FOLDER_NAME     = "PDBs_cleaned_filtered_v2"
    GRAPHML_OUT_FOLDER_NAME = "GraphMLs_filtered_v2"
    # ------------------------------------------------------------------ #

    # Step 1 – filter CSV
    filter_xray_csv(
        input_csv_path=INPUT_CSV,
        output_csv_path=OUTPUT_CSV,
    )

    # Step 2 – diagnose before copying (comment out once working)
    diagnose_paths(
        source_base_dir=SOURCE_BASE_DIR,
        filtered_csv_path=OUTPUT_CSV,
        cif_folder_name=CIF_FOLDER_NAME,
        pdb_folder_name=PDB_FOLDER_NAME,
        graphml_folder_name=GRAPHML_FOLDER_NAME,
    )

    # Step 3 – copy matching files into the three new folders
    copy_filtered_files(
        filtered_csv_path=OUTPUT_CSV,
        source_base_dir=SOURCE_BASE_DIR,
        cif_folder_name=CIF_FOLDER_NAME,
        pdb_folder_name=PDB_FOLDER_NAME,
        graphml_folder_name=GRAPHML_FOLDER_NAME,
        cif_out_folder_name=CIF_OUT_FOLDER_NAME,
        pdb_out_folder_name=PDB_OUT_FOLDER_NAME,
        graphml_out_folder_name=GRAPHML_OUT_FOLDER_NAME,
    )


