# test_remove_multichains.py
# ============================================================
# Scan the JSON directory and move any PDBID_clean.json files
# whose PDB ID does NOT appear in the biolip_per_pdb_summary
# CSV into a separate "moved" subdirectory.
#
# ~430 JSONs exist but only ~233 have a BioLiP match; the
# remainder (~197) are moved out so only matched structures
# remain in the working directory.
#
# Files are MOVED (not deleted) so nothing is lost — review
# the moved directory and delete permanently only when
# satisfied.
#
# A dry-run mode (enabled by default) prints what would be
# moved without touching the filesystem.  Set dry_run=False
# to execute.
#
# Outputs
#   <moved_dir>/              – destination for moved JSON files
#   <moved_dir>/move_log.csv  – record of every move with reason
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_biolip_ids(summary_csv: Path) -> set:
    """
    Read the BioLiP per-PDB summary CSV and return the set of PDB IDs
    that have a matched entry (i.e. are present in the CSV at all).

    Required column: pdb_id
    """
    df = pd.read_csv(summary_csv)

    if "pdb_id" not in df.columns:
        raise ValueError(
            "Summary CSV is missing required column 'pdb_id'."
        )

    ids = set(df["pdb_id"].str.upper().str.strip().unique())
    print(f"  {len(ids)} PDB ID(s) found in BioLiP summary CSV.")
    return ids


def build_file_map(
    biolip_ids: set,
    json_dir: Path,
) -> list:
    """
    Scan json_dir for *_clean.json files and flag any whose PDB ID is
    not present in biolip_ids.

    Returns a list of dicts with keys:
        pdb_id, reason, src, found
    'found' is always True here (the file exists by definition of the
    directory scan), kept for compatibility with move_files/report.
    """
    file_map = []
    for json_file in sorted(json_dir.glob("*_clean.json")):
        # Extract PDB ID from filename: PDBID_clean.json
        pid = json_file.stem.replace("_clean", "").upper()
        if pid not in biolip_ids:
            file_map.append({
                "pdb_id": pid,
                "reason": "not_in_biolip",
                "src":    json_file,
                "found":  True,
            })
    return file_map


def report(file_map: list, moved_dir: Path, dry_run: bool) -> None:
    """Print a pre-move summary to stdout."""
    mode_label = "DRY RUN — no files will be moved" if dry_run else "LIVE RUN"
    print(f"\n{'=' * 50}")
    print(f"  Move plan  [{mode_label}]")
    print(f"{'=' * 50}")
    print(f"  Destination   : {moved_dir}")
    print(f"  Files to move : {len(file_map)}")

    if file_map:
        print("\n  Files queued for move:")
        for f in file_map:
            print(f"    {f['src'].name:<30}  reason={f['reason']}")

    print()


def move_files(
    file_map: list,
    moved_dir: Path,
    dry_run: bool,
) -> pd.DataFrame:
    """
    Move files from json_dir to moved_dir.

    Returns a log DataFrame recording the outcome of each attempted move.
    """
    LOG_COLS = ["pdb_id", "reason", "filename", "src", "dst", "status"]
    if not file_map:
        return pd.DataFrame(columns=LOG_COLS)

    moved_dir.mkdir(parents=True, exist_ok=True)

    log_rows = []
    for f in file_map:
        dst    = moved_dir / f["src"].name
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
            "pdb_id":   f["pdb_id"],
            "reason":   f["reason"],
            "filename": f["src"].name,
            "src":      str(f["src"]),
            "dst":      str(dst),
            "status":   status,
        })

    return pd.DataFrame(log_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    # ── Input ───────────────────────────────────────────────────────────────
    # CSV produced by 0_6_biolip_lookup — only PDB IDs present here are kept
    summary_csv = FINAL_DIR / "biolip_lookup" / "biolip_per_pdb_summary.csv",

    # Directory containing the PDBID_clean.json test files
    json_dir    = DATA_DIR_3 / "test" / "JSONs",

    # ── Output ──────────────────────────────────────────────────────────────
    # Subdirectory where unmatched JSON files will be placed
    moved_dir   = DATA_DIR_3 / "test" / "JSONs" / "moved",

    # ── Safety ──────────────────────────────────────────────────────────────
    # Set to False to perform the actual move. True by default so the first
    # run is always a safe preview.
    dry_run     = True,
):
    summary_csv = Path(summary_csv)
    json_dir    = Path(json_dir)
    moved_dir   = Path(moved_dir)

    # ── 1. Load matched PDB IDs from BioLiP summary CSV ─────────────────────
    print(f"Reading summary CSV: {summary_csv}")
    biolip_ids = load_biolip_ids(summary_csv)

    # ── 2. Scan JSON dir and find files not in the CSV ───────────────────────
    print(f"Scanning JSON dir : {json_dir}")
    file_map = build_file_map(biolip_ids, json_dir)
    print(f"  {len(file_map)} JSON file(s) not matched in BioLiP summary.")

    # ── 3. Print plan ────────────────────────────────────────────────────────
    report(file_map, moved_dir, dry_run)

    # ── 4. Move (or simulate) ────────────────────────────────────────────────
    log_df = move_files(file_map, moved_dir, dry_run)

    # ── 5. Save log ──────────────────────────────────────────────────────────
    log_path = moved_dir / "move_log.csv"
    moved_dir.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)

    # ── 6. Print outcome summary ─────────────────────────────────────────────
    moved_count   = (log_df["status"] == "moved").sum()
    dry_count     = (log_df["status"] == "dry_run").sum()
    skipped_count = (log_df["status"] == "skipped_missing").sum()
    error_count   = log_df["status"].str.startswith("error").sum()

    if dry_run:
        print(f"Dry run complete — {dry_count} file(s) would be moved.")
        print("Set dry_run=False in main() to execute.")
    else:
        print(f"Move complete — {moved_count} moved, "
              f"{skipped_count} not found, {error_count} errors.")

    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main(dry_run=False)
