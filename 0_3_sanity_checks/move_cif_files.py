# move_cif_files.py
# ============================================================
# Given a directory of already-moved JSON files and a source
# directory of CIF files, locate the CIF file for each JSON
# (matched by PDB ID) and move it — along with any companion
# files sharing the same stem (.break_residues, .breaks) —
# into a designated output directory.
#
# Files are MOVED (not deleted) so nothing is lost.
#
# A dry-run mode (enabled by default) prints what would be
# moved without touching the filesystem.  Set dry_run=False
# in main() to execute.
#
# Outputs
#   <cif_moved_dir>/              – destination for moved CIF files
#   <cif_moved_dir>/move_log.csv  – record of every move with status
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *


# ---------------------------------------------------------------------------
# Companion-file suffixes to move alongside the primary CIF
# ---------------------------------------------------------------------------

COMPANION_SUFFIXES = {".break_residues", ".breaks"}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def get_pdb_ids_from_json_dir(json_dir: Path) -> list[str]:
    """
    Iterate through *json_dir* and extract PDB IDs from JSON filenames.

    Expects filenames of the form  ``PDBID_clean.json``  (uppercase stem
    before the first underscore).  Only files with a ``.json`` extension
    are considered.

    Returns a sorted list of upper-cased PDB IDs.
    """
    pdb_ids = []
    for f in sorted(json_dir.iterdir()):
        if f.suffix.lower() != ".json":
            continue
        # stem examples: "1ABC_clean"  →  PDB ID = "1ABC"
        pdb_id = f.stem.split("_")[0].upper()
        pdb_ids.append(pdb_id)

    print(f"  Found {len(pdb_ids)} JSON file(s) in: {json_dir}")
    return pdb_ids


def build_cif_file_map(
    pdb_ids: list[str],
    cif_dir: Path,
) -> list[dict]:
    """
    For each PDB ID, locate the primary CIF file and any companion files
    (.break_residues, .breaks) in *cif_dir*.

    CIF files are expected to follow the pattern ``<stem>.cif`` where the
    stem starts with the PDB ID (case-insensitive match attempted first on
    an exact-case match, then lower-case, then upper-case).

    Returns a list of dicts, one per file (primary + companions), each with:
        pdb_id   – the PDB ID this file belongs to
        file_type – "cif" | "break_residues" | "breaks"
        src      – full Path to the source file
        found    – bool, whether the file exists
    """
    # Build a lookup of all files in cif_dir keyed by lower-cased stem
    cif_dir_files: dict[str, Path] = {}
    for f in cif_dir.iterdir():
        if f.is_file():
            cif_dir_files[f.name.lower()] = f

    file_map = []

    for pdb_id in pdb_ids:
        # ── Resolve the primary CIF ──────────────────────────────────────
        # Try common stem patterns: PDBID_clean.cif, pdbid_clean.cif, PDBID.cif
        candidate_stems = [
            f"{pdb_id}_clean",
            f"{pdb_id.lower()}_clean",
            pdb_id,
            pdb_id.lower(),
        ]

        cif_path: Path | None = None
        matched_stem: str | None = None

        for stem in candidate_stems:
            candidate_name = f"{stem}.cif"
            if candidate_name.lower() in cif_dir_files:
                cif_path     = cif_dir_files[candidate_name.lower()]
                matched_stem = stem
                break

        file_map.append({
            "pdb_id":    pdb_id,
            "file_type": "cif",
            "src":       cif_path if cif_path else cif_dir / f"{pdb_id}_clean.cif",
            "found":     cif_path is not None,
        })

        # ── Resolve companion files ──────────────────────────────────────
        if matched_stem is not None:
            for suffix in COMPANION_SUFFIXES:
                companion_name = f"{matched_stem}{suffix}"
                companion_path = cif_dir_files.get(companion_name.lower())
                file_map.append({
                    "pdb_id":    pdb_id,
                    "file_type": suffix.lstrip("."),
                    "src":       companion_path if companion_path
                                 else cif_dir / companion_name,
                    "found":     companion_path is not None,
                })

    return file_map


def report_cif(file_map: list[dict], cif_moved_dir: Path, dry_run: bool) -> None:
    """Print a pre-move summary to stdout."""
    found   = [f for f in file_map if f["found"]]
    missing = [f for f in file_map if not f["found"]]

    mode_label = "DRY RUN — no files will be moved" if dry_run else "LIVE RUN"
    print(f"\n{'=' * 55}")
    print(f"  CIF Move Plan  [{mode_label}]")
    print(f"{'=' * 55}")
    print(f"  Destination      : {cif_moved_dir}")
    print(f"  Files to move    : {len(found)}")
    if missing:
        print(f"  Not found        : {len(missing)}  (listed below)")

    if found:
        print("\n  Files queued for move:")
        for f in sorted(found, key=lambda x: (x["pdb_id"], x["file_type"])):
            print(f"    [{f['file_type']:<16}]  {f['src'].name}")

    if missing:
        print("\n  Files NOT found in cif_dir:")
        for f in sorted(missing, key=lambda x: (x["pdb_id"], x["file_type"])):
            print(f"    {f['pdb_id']}  ({f['file_type']})  expected: {f['src'].name}")

    print()


def move_cif_files(
    file_map:      list[dict],
    cif_moved_dir: Path,
    dry_run:       bool,
) -> pd.DataFrame:
    """
    Move files listed in *file_map* to *cif_moved_dir*.

    Skips entries where ``found=False``.

    Returns a log DataFrame with columns:
        pdb_id, file_type, filename, src, dst, status
    """
    LOG_COLS = ["pdb_id", "file_type", "filename", "src", "dst", "status"]

    if not file_map:
        return pd.DataFrame(columns=LOG_COLS)

    cif_moved_dir.mkdir(parents=True, exist_ok=True)

    log_rows = []
    for f in file_map:
        dst    = cif_moved_dir / f["src"].name
        status = "skipped_missing"

        if f["found"]:
            if dry_run:
                status = "dry_run"
            else:
                try:
                    shutil.move(str(f["src"]), str(dst))
                    status = "moved"
                except Exception as exc:
                    status = f"error: {exc}"

        log_rows.append({
            "pdb_id":    f["pdb_id"],
            "file_type": f["file_type"],
            "filename":  f["src"].name,
            "src":       str(f["src"]),
            "dst":       str(dst),
            "status":    status,
        })

    return pd.DataFrame(log_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    # ── Input ───────────────────────────────────────────────────────────────
    # Directory containing the already-moved JSON files (used to derive PDB IDs)
    # json_dir      = DATA_DIR_3 / "ref" / "JSONs" / "moved",
    json_dir      = DATA_DIR_3 / "ref" / "ref_JSONs_removed",

    # Directory containing the source CIF files (and companion files)
    # cif_dir       = DATA_DIR_3 / "ref" / "CIFs_cleaned" / "structures",
    cif_dir       = DATA_DIR_3 / "ref" / "CIFs_cleaned" / "structures",

    # ── Output ──────────────────────────────────────────────────────────────
    # Directory where matching CIF (and companion) files will be moved
    cif_moved_dir = DATA_DIR_3 / "ref" / "CIFs_cleaned" / "moved",

    # ── Safety ──────────────────────────────────────────────────────────────
    # Set to False to perform the actual move. True by default so the first
    # run is always a safe preview.
    dry_run       = False,
):
    json_dir      = Path(json_dir)
    cif_dir       = Path(cif_dir)
    cif_moved_dir = Path(cif_moved_dir)

    # ── 1. Collect PDB IDs from the JSON directory ───────────────────────────
    print(f"\nScanning JSON directory: {json_dir}")
    pdb_ids = get_pdb_ids_from_json_dir(json_dir)

    if not pdb_ids:
        print("  No JSON files found — nothing to do.")
        return

    # ── 2. Resolve CIF file paths ────────────────────────────────────────────
    print(f"\nResolving CIF files in: {cif_dir}")
    file_map = build_cif_file_map(pdb_ids, cif_dir)

    # ── 3. Print plan ────────────────────────────────────────────────────────
    report_cif(file_map, cif_moved_dir, dry_run)

    # ── 4. Move (or simulate) ────────────────────────────────────────────────
    log_df = move_cif_files(file_map, cif_moved_dir, dry_run)

    # ── 5. Save log ──────────────────────────────────────────────────────────
    cif_moved_dir.mkdir(parents=True, exist_ok=True)
    log_path = cif_moved_dir / "cif_move_log.csv"
    log_df.to_csv(log_path, index=False)

    # ── 6. Summary ───────────────────────────────────────────────────────────
    n_moved   = (log_df["status"] == ("dry_run" if dry_run else "moved")).sum()
    n_skipped = (log_df["status"] == "skipped_missing").sum()
    n_errors  = log_df["status"].str.startswith("error").sum()

    print(f"{'DRY RUN — ' if dry_run else ''}Results")
    print(f"  {'Would move' if dry_run else 'Moved'}  : {n_moved}")
    print(f"  Skipped (not found) : {n_skipped}")
    if n_errors:
        print(f"  Errors              : {n_errors}")
    print(f"  Log saved to        : {log_path}\n")


if __name__ == "__main__":
    main()
