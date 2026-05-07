# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *


def sort_cif_by_hetatm_flags(enzymes_cif_dir, hetatm_inspection_dir, output_parent_dir=None):
    """
    Sorts CIF files into two folders based on whether their PDB ID appears
    in the hetatm_inspection folder (as a *_flagged_hetatm.csv file).

    Args:
        enzymes_cif_dir (str or Path):
            Path to the folder containing CIF files named "<PDB_ID>.cif".
        hetatm_inspection_dir (str or Path):
            Path to the folder containing CSV files named "<PDB_ID>_flagged_hetatm.csv".
        output_parent_dir (str or Path, optional):
            Directory where the two output folders will be created.
            Defaults to the parent of enzymes_cif_dir.

    Returns:
        dict: Summary with keys "flagged", "unflagged", and "missing_cif",
              each mapping to a list of PDB IDs.
    """
    enzymes_cif_dir     = Path(enzymes_cif_dir)
    hetatm_inspection_dir = Path(hetatm_inspection_dir)

    # Default output location: same parent as the CIF folder
    if output_parent_dir is None:
        output_parent_dir = enzymes_cif_dir.parent
    else:
        output_parent_dir = Path(output_parent_dir)

    flagged_dir   = output_parent_dir / "enzymes_CIF_flagged"
    unflagged_dir = output_parent_dir / "enzymes_CIF_unflagged"
    flagged_dir.mkdir(parents=True, exist_ok=True)
    unflagged_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect PDB IDs that HAVE a flagged_hetatm CSV ──────────────────────
    flagged_pdb_ids = set()
    for csv_file in hetatm_inspection_dir.glob("*_flagged_hetatm.csv"):
        # Strip the "_flagged_hetatm" suffix to recover the PDB ID
        pdb_id = csv_file.stem.replace("_flagged_hetatm", "")
        flagged_pdb_ids.add(pdb_id.upper())

    # ── Walk every CIF file and route it to the correct output folder ────────
    summary = {"flagged": [], "unflagged": [], "missing_cif": []}

    for cif_file in enzymes_cif_dir.glob("*.cif"):
        pdb_id = cif_file.stem.upper()          # e.g. "1A4V"
        dest_dir = flagged_dir if pdb_id in flagged_pdb_ids else unflagged_dir
        shutil.copy2(cif_file, dest_dir / cif_file.name)

        key = "flagged" if pdb_id in flagged_pdb_ids else "unflagged"
        summary[key].append(pdb_id)

    # ── Optionally report flagged CSVs whose CIF is absent ──────────────────
    cif_pdb_ids = {f.stem.upper() for f in enzymes_cif_dir.glob("*.cif")}
    summary["missing_cif"] = sorted(flagged_pdb_ids - cif_pdb_ids)

    # ── Print a short summary ────────────────────────────────────────────────
    print(f"Output folders created in: {output_parent_dir.resolve()}")
    print(f"  enzymes_CIF_flagged   : {len(summary['flagged'])} CIF files")
    print(f"  enzymes_CIF_unflagged : {len(summary['unflagged'])} CIF files")
    if summary["missing_cif"]:
        print(f"  WARNING – {len(summary['missing_cif'])} flagged PDB ID(s) had no matching CIF: "
              f"{summary['missing_cif']}")

    return summary


# ── Configure your paths here ────────────────────────────────────────────────
if __name__ == "__main__":

    ENZYMES_CIF_DIR       = DATA_DIR / "enzymes" / "enzymes_CIFs"
    HETATM_INSPECTION_DIR = DATA_DIR / "enzymes" / "hetatm_inspection"

    # Optional: set a custom parent directory for the two output folders.
    # Leave as None to place them next to ENZYMES_CIF_DIR.
    OUTPUT_PARENT_DIR = DATA_DIR / "enzymes" / "sorted_enzymes_CIFs"

    results = sort_cif_by_hetatm_flags(
        enzymes_cif_dir       = ENZYMES_CIF_DIR,
        hetatm_inspection_dir = HETATM_INSPECTION_DIR,
        output_parent_dir     = OUTPUT_PARENT_DIR,
    )
