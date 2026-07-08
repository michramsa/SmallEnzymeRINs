# ref_remove_multichains.py
# ============================================================
# Read the reference-set size distribution CSV and move any
# PDBID_clean.json files where n_chains != 1 out of the
# reference JSON directory into a separate "moved"
# subdirectory.
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

def load_removal_targets(summary_csv: Path) -> pd.DataFrame:
    """
    Read the reference-set size distribution CSV and return rows where
    n_chains != 1.

    Required columns: pdb_id, n_chains

    Returns a DataFrame of flagged rows with a 'reason' column indicating
    the actual chain count.
    """
    df = pd.read_csv(summary_csv)

    required = {"pdb_id", "n_chains"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Summary CSV is missing expected columns: {missing}."
        )

    df["pdb_id"]   = df["pdb_id"].str.upper().str.strip()
    df["n_chains"] = pd.to_numeric(df["n_chains"], errors="coerce")

    print(f"  n_chains value counts:\n"
          f"{df['n_chains'].value_counts(dropna=False).sort_index().to_string()}")

    targets = df[df["n_chains"] != 1].copy().reset_index(drop=True)
    targets["reason"] = targets["n_chains"].apply(
        lambda n: f"n_chains={int(n)}" if pd.notna(n) else "n_chains=NaN"
    )
    return targets


def build_file_map(
    targets: pd.DataFrame,
    json_dir: Path,
) -> list:
    """
    For each target PDB ID, resolve the expected PDBID_clean.json path.

    Returns a list of dicts with keys:
        pdb_id, reason, src, found
    where `found` is False if the file does not exist in json_dir.
    """
    file_map = []
    for _, row in targets.iterrows():
        pid      = row["pdb_id"]
        filename = f"{pid}_clean.json"
        src      = json_dir / filename
        file_map.append({
            "pdb_id": pid,
            "reason": row["reason"],
            "src":    src,
            "found":  src.exists(),
        })
    return file_map


def report(file_map: list, moved_dir: Path, dry_run: bool) -> None:
    """Print a pre-move summary to stdout."""
    found   = [f for f in file_map if f["found"]]
    missing = [f for f in file_map if not f["found"]]

    mode_label = "DRY RUN — no files will be moved" if dry_run else "LIVE RUN"
    print(f"\n{'=' * 50}")
    print(f"  Move plan  [{mode_label}]")
    print(f"{'=' * 50}")
    print(f"  Destination   : {moved_dir}")
    print(f"  Files to move : {len(found)}")
    if missing:
        print(f"  Not in dir    : {len(missing)}  (listed below)")

    if found:
        print("\n  Files queued for move:")
        for f in sorted(found, key=lambda x: x["pdb_id"]):
            print(f"    {f['src'].name:<30}  reason={f['reason']}")

    if missing:
        print("\n  PDB IDs in CSV but NOT found in json_dir:")
        for f in sorted(missing, key=lambda x: x["pdb_id"]):
            print(f"    {f['pdb_id']}  (expected: {f['src'].name})")

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
    # Reference-set size distribution CSV with pdb_id and n_chains columns
    summary_csv = FINAL_DIR / "figures" / "enzyme_dist_no_mc" / "no_mc_size_dist_ref_set.csv",

    # Directory containing the PDBID_clean.json reference files
    json_dir    = DATA_DIR_3 / "ref" / "JSONs" / "structures",

    # ── Output ──────────────────────────────────────────────────────────────
    # Subdirectory where multi-chain JSON files will be placed
    moved_dir   = DATA_DIR_3 / "ref" / "JSONs" / "moved",

    # ── Safety ──────────────────────────────────────────────────────────────
    # Set to False to perform the actual move. True by default so the first
    # run is always a safe preview.
    dry_run     = True,
):
    summary_csv = Path(summary_csv)
    json_dir    = Path(json_dir)
    moved_dir   = Path(moved_dir)

    # ── 1. Load multi-chain targets from size distribution CSV ───────────────
    print(f"Reading summary CSV: {summary_csv}")
    targets = load_removal_targets(summary_csv)
    print(f"  {len(targets)} PDB ID(s) flagged for move (n_chains != 1).")

    # ── 2. Resolve file paths ────────────────────────────────────────────────
    file_map = build_file_map(targets, json_dir)

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
