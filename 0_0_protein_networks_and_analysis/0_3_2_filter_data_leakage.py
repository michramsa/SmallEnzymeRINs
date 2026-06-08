"""
filter_data_leakage.py

Identifies PDB IDs flagged as data leakage from a TSV search results file
and moves the corresponding RIN (.graphml) and cleaned CIF (.cif) files
out of their working directories into a dedicated `data_leakage` directory.

The TSV is expected to have a header row with at least a `query` column,
where each query entry is a PDB ID concatenated with a chain ID
(e.g. `1m4dA` -> PDB ID `1m4d`, `7ah2AAA` -> PDB ID `7ah2`).
PDB IDs are always 4 characters, so we extract the first 4 characters.

Source folders (under UF_ENZYMES_DIR):
    - RINs/<PDB_ID>_clean.graphml
    - CIFs_cleaned/<PDB_ID>_clean.cif

Destination folders (under UF_ENZYMES_DIR / "data_leakage"):
    - RINs/<PDB_ID>_clean.graphml
    - CIFs_cleaned/<PDB_ID>_clean.cif
"""

# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *


def load_leakage_pdb_ids(tsv_path, query_col="query", pdb_id_length=4):
    """
    Read the leakage TSV and return a set of unique PDB IDs (lowercased)
    extracted from the `query` column.

    Parameters
    ----------
    tsv_path : Path or str
        Path to the TSV file containing leakage hits.
    query_col : str
        Name of the column holding the query identifier
        (PDB ID + chain ID, e.g. `1m4dA`).
    pdb_id_length : int
        Number of leading characters that make up the PDB ID. Standard
        PDB IDs are 4 characters long.

    Returns
    -------
    set[str]
        Set of unique, lowercase PDB IDs.
    """
    # The TSV in this project has a quirk: the *header* row uses ' | ' as
    # a visual separator (no tabs at all), while the *data* rows are
    # tab-delimited. If we just call pd.read_csv(sep='\t') the header
    # collapses into a single column. So we parse the header manually and
    # read the data rows skipping that first line.
    tsv_path = Path(tsv_path)
    with open(tsv_path, "r") as fh:
        header_line = fh.readline().strip()

    if "\t" in header_line:
        # Header is properly tab-delimited; let pandas handle it normally
        df = pd.read_csv(tsv_path, sep="\t")
        df.columns = [c.strip().strip("|").strip() for c in df.columns]
    else:
        # Header uses ' | ' as a separator; split manually
        column_names = [c.strip() for c in header_line.split("|")]
        df = pd.read_csv(
            tsv_path,
            sep="\t",
            header=None,
            names=column_names,
            skiprows=1,
        )

    if query_col not in df.columns:
        raise KeyError(
            f"Column '{query_col}' not found in TSV. "
            f"Available columns: {list(df.columns)}"
        )

    # Extract the PDB ID (first N characters), lowercase for consistent matching
    pdb_ids = (
        df[query_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str[:pdb_id_length]
        .str.lower()
        .unique()
    )

    return set(pdb_ids)


def move_matching_files(pdb_ids, source_dir, dest_dir, file_suffix):
    """
    Move files from `source_dir` to `dest_dir` whose filename matches
    `<PDB_ID><file_suffix>` for any PDB ID in `pdb_ids`.

    Matching is case-insensitive on the PDB ID portion of the filename
    so we work whether the on-disk files are upper- or lower-case.

    Parameters
    ----------
    pdb_ids : set[str]
        Set of PDB IDs (lowercase) to move.
    source_dir : Path
        Directory currently containing the files.
    dest_dir : Path
        Directory to move matching files into. Created if missing.
    file_suffix : str
        Filename suffix following the PDB ID, e.g. `_clean.graphml`.

    Returns
    -------
    dict
        Summary with counts and lists of moved / missing / skipped files.
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    moved = []
    missing = []
    already_in_dest = []

    # Build a case-insensitive lookup of files in source_dir by PDB ID stem.
    # Using a dict allows us to find the actual filename regardless of case.
    on_disk = {}
    for f in source_dir.iterdir():
        if f.is_file() and f.name.lower().endswith(file_suffix.lower()):
            stem_pdb = f.name[: -len(file_suffix)].lower()
            on_disk[stem_pdb] = f

    for pdb_id in sorted(pdb_ids):
        src_file = on_disk.get(pdb_id)
        if src_file is None:
            missing.append(pdb_id)
            continue

        dst_file = dest_dir / src_file.name
        if dst_file.exists():
            # Already moved in a previous run; nothing to do
            already_in_dest.append(src_file.name)
            continue

        shutil.move(str(src_file), str(dst_file))
        moved.append(src_file.name)

    return {
        "moved": moved,
        "missing": missing,
        "already_in_dest": already_in_dest,
        "source_dir": source_dir,
        "dest_dir": dest_dir,
    }


def report(summary, label):
    """Print a short summary of a move operation."""
    print(f"\n--- {label} ---")
    print(f"Source: {summary['source_dir']}")
    print(f"Dest:   {summary['dest_dir']}")
    print(f"Moved:           {len(summary['moved'])}")
    print(f"Already in dest: {len(summary['already_in_dest'])}")
    print(f"Not found:       {len(summary['missing'])}")
    if summary["missing"]:
        # Show only a handful so the log stays readable
        preview = ", ".join(summary["missing"][:10])
        more = "" if len(summary["missing"]) <= 10 else f" ... (+{len(summary['missing']) - 10} more)"
        print(f"  Missing PDB IDs: {preview}{more}")


def main(
    leakage_tsv,
    rins_source,
    cifs_source,
    rins_dest,
    cifs_dest,
    query_col="query",
    pdb_id_length=4,
    rin_suffix="_clean.graphml",
    cif_suffix="_clean.cif",
):
    """
    Top-level orchestrator. Loads leakage PDB IDs from the TSV and moves
    matching RIN and CIF files into the `data_leakage` subdirectories.
    """
    print(f"Loading leakage hits from: {leakage_tsv}")
    pdb_ids = load_leakage_pdb_ids(
        leakage_tsv,
        query_col=query_col,
        pdb_id_length=pdb_id_length,
    )
    print(f"Unique PDB IDs flagged as data leakage: {len(pdb_ids)}")

    rin_summary = move_matching_files(
        pdb_ids=pdb_ids,
        source_dir=rins_source,
        dest_dir=rins_dest,
        file_suffix=rin_suffix,
    )
    report(rin_summary, "RINs (.graphml)")

    cif_summary = move_matching_files(
        pdb_ids=pdb_ids,
        source_dir=cifs_source,
        dest_dir=cifs_dest,
        file_suffix=cif_suffix,
    )
    report(cif_summary, "CIFs_cleaned (.cif)")

    print("\nDone.")
    return {"rin": rin_summary, "cif": cif_summary}


if __name__ == "__main__":

    ## keep in mind - this TSV is filtering out any overlaps with the reference set/"old_PISCES"!!
    leakage_tsv = ENZYMES_DIR / "mmseq" / "runs" / "050726_leakage_search.tsv"

    # Source directories (where files currently live)
    rins_source = DATA_DIR_2 / "RINs"
    cifs_source = DATA_DIR_2 / "enzymes_CIF_unflagged" / "CIFs_cleaned"

    # Destination directories (under data_leakage/)
    data_leakage_dir = DATA_DIR_2 / "519_data_leakage"
    rins_dest = data_leakage_dir / "519_RINs"
    cifs_dest = data_leakage_dir / "519_CIFs_cleaned"

    # TSV / filename conventions
    query_col = "query"
    pdb_id_length = 4
    rin_suffix = "_clean.graphml"
    cif_suffix = "_clean.cif"

    main(
        leakage_tsv=leakage_tsv,
        rins_source=rins_source,
        cifs_source=cifs_source,
        rins_dest=rins_dest,
        cifs_dest=cifs_dest,
        query_col=query_col,
        pdb_id_length=pdb_id_length,
        rin_suffix=rin_suffix,
        cif_suffix=cif_suffix,
    )
