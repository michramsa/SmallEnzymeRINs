# move_SASA_files.py
# ============================================================
# For each *_centrality.csv found in a Centrality directory:
#   1. Extract the PDB ID from the filename.
#   2. Locate the matching *_sasa.csv in one of two SASA source
#      directories (test or reference).
#   3. Validate residue consensus: every row joined on
#      (Chain, Residue, Residue number) must agree between the
#      two files before any values are written.
#   4. Append RSA-theoretical and RSA-empirical to the Centrality
#      CSV in-place (only for rows that matched; unmatched rows
#      receive NaN).
#   5. Move the SASA file to a designated output directory.
#
# A dry-run mode (enabled by default) prints what would happen
# without touching the filesystem.  Set dry_run=False in main()
# to execute.
#
# Outputs
#   <sasa_moved_dir>/                 – destination for moved SASA files
#   <sasa_moved_dir>/sasa_move_log.csv  – per-file move log
#   <centrality_dir>/                 – centrality CSVs are edited in-place
#   <centrality_dir>/sasa_merge_log.csv – per-protein merge quality report
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns written into the centrality CSV
RSA_COLS = ["RSA-theoretical", "RSA-empirical"]

# Join keys
#   Centrality side : Chain, Residue, Residue number
#   SASA side       : Chain, Residue, Residue ID
CENT_JOIN_KEYS = ["Chain", "Residue", "Residue number"]
SASA_JOIN_KEYS = ["Chain", "Residue", "Residue ID"]


# ---------------------------------------------------------------------------
# Helper: locate SASA file for a given PDB ID
# ---------------------------------------------------------------------------

def find_sasa_file(pdb_id: str, sasa_dirs: list[Path]) -> Path | None:
    """
    Search each directory in *sasa_dirs* for a file whose name matches
    ``<pdb_id>_sasa.csv`` (case-insensitive on the PDB ID portion).

    Returns the first match found, or None if no file exists in any directory.
    """
    for sdir in sasa_dirs:
        if not sdir.is_dir():
            continue
        # Try exact-case first, then lower-cased PDB ID
        for variant in [pdb_id, pdb_id.lower(), pdb_id.upper()]:
            candidate = sdir / f"{variant}_sasa.csv"
            if candidate.exists():
                return candidate
    return None


# ---------------------------------------------------------------------------
# Core: merge RSA values into one centrality CSV
# ---------------------------------------------------------------------------

def merge_rsa_into_centrality(
    cent_path: Path,
    sasa_path: Path,
    dry_run:   bool,
) -> dict:
    """
    Read the centrality CSV and the SASA CSV, validate residue consensus,
    then write RSA-theoretical and RSA-empirical back into the centrality CSV.

    Consensus rule
    --------------
    A row is considered matched when (Chain, Residue, Residue number) in the
    centrality CSV agrees with (Chain, Residue, Residue ID) in the SASA CSV.
    Only matching rows receive RSA values; unmatched rows get NaN.

    The SASA file may contain multiple models (Model column); only the
    lowest-numbered model is used to avoid duplicating rows.

    Returns a summary dict with merge statistics.
    """
    pdb_id = cent_path.stem.replace("_centrality", "")

    cent_df = pd.read_csv(cent_path)
    sasa_df = pd.read_csv(sasa_path)

    # ── guard: skip if RSA columns already present ───────────────────────
    already_present = [c for c in RSA_COLS if c in cent_df.columns]
    if already_present:
        return {
            "pdb_id":         pdb_id,
            "status":         "skipped_already_merged",
            "n_centrality":   len(cent_df),
            "n_sasa":         len(sasa_df),
            "n_matched":      None,
            "n_unmatched":    None,
            "note":           f"Columns already exist: {already_present}",
            "unmatched_rows": pd.DataFrame(columns=["pdb_id", "Chain", "Residue", "Residue number"]),
        }

    # ── use only the first (lowest-numbered) model ───────────────────────
    if "Model" in sasa_df.columns:
        min_model = sasa_df["Model"].min()
        sasa_df   = sasa_df[sasa_df["Model"] == min_model].copy()

    # ── normalise join-key dtypes ─────────────────────────────────────────
    cent_df["Residue number"] = cent_df["Residue number"].astype(int)
    sasa_df["Residue ID"]     = sasa_df["Residue ID"].astype(int)

    # Strip whitespace from string join keys
    for col in ["Chain", "Residue"]:
        cent_df[col] = cent_df[col].astype(str).str.strip()
        sasa_df[col] = sasa_df[col].astype(str).str.strip()

    # ── merge ─────────────────────────────────────────────────────────────
    merged = cent_df.merge(
        sasa_df[["Chain", "Residue", "Residue ID"] + RSA_COLS],
        left_on  = CENT_JOIN_KEYS,          # Chain, Residue, Residue number
        right_on = SASA_JOIN_KEYS,          # Chain, Residue, Residue ID
        how      = "left",
    )

    # Drop the redundant "Residue ID" column introduced by the merge
    merged.drop(columns=["Residue ID"], errors="ignore", inplace=True)

    n_matched   = merged[RSA_COLS[0]].notna().sum()
    n_unmatched = merged[RSA_COLS[0]].isna().sum()

    # ── write back ────────────────────────────────────────────────────────
    if not dry_run:
        merged.to_csv(cent_path, index=False)

    # ── collect unmatched residue identities ─────────────────────────────
    unmatched_rows = merged[merged[RSA_COLS[0]].isna()][
        ["Chain", "Residue", "Residue number"]
    ].copy()
    unmatched_rows.insert(0, "pdb_id", pdb_id)

    return {
        "pdb_id":         pdb_id,
        "status":         "dry_run" if dry_run else "merged",
        "n_centrality":   len(cent_df),
        "n_sasa":         len(sasa_df),
        "n_matched":      n_matched,
        "n_unmatched":    n_unmatched,
        "unmatched_rows": unmatched_rows,   # DataFrame, may be empty
    }


# ---------------------------------------------------------------------------
# Core: move a single SASA file
# ---------------------------------------------------------------------------

def move_sasa_file(
    sasa_path:     Path,
    sasa_moved_dir: Path,
    dry_run:       bool,
) -> dict:
    """
    Move *sasa_path* to *sasa_moved_dir*.

    Returns a log dict with keys: filename, src, dst, status.
    """
    dst    = sasa_moved_dir / sasa_path.name
    status = "dry_run"

    if not dry_run:
        try:
            shutil.move(str(sasa_path), str(dst))
            status = "moved"
        except Exception as exc:
            status = f"error: {exc}"

    return {
        "filename": sasa_path.name,
        "src":      str(sasa_path),
        "dst":      str(dst),
        "status":   status,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_plan(
    cent_paths:    list[Path],
    sasa_found:    dict[str, Path],   # pdb_id -> Path
    sasa_missing:  list[str],         # pdb_ids with no SASA file
    sasa_moved_dir: Path,
    dry_run:       bool,
) -> None:
    """Print a pre-run summary to stdout."""
    mode_label = "DRY RUN — no files will be changed" if dry_run else "LIVE RUN"

    print(f"\n{'=' * 60}")
    print(f"  SASA Merge + Move Plan  [{mode_label}]")
    print(f"{'=' * 60}")
    print(f"  Centrality files found  : {len(cent_paths)}")
    print(f"  SASA files matched      : {len(sasa_found)}")
    print(f"  SASA files missing      : {len(sasa_missing)}")
    print(f"  SASA destination        : {sasa_moved_dir}")

    if sasa_found:
        print("\n  Proteins to process:")
        for pdb_id, spath in sorted(sasa_found.items()):
            print(f"    {pdb_id:<10}  SASA: {spath.name}")

    if sasa_missing:
        print("\n  PDB IDs with no SASA file:")
        for pdb_id in sorted(sasa_missing):
            print(f"    {pdb_id}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    # ── Input: Centrality directory ──────────────────────────────────────────
    # Directory containing *_centrality.csv files.  PDB ID is taken from each
    # filename stem (e.g. "1abc_centrality.csv" → PDB ID "1abc").
    centrality_dir = DATA_DIR_3 / "ref" / "Centralities",

    # ── Input: SASA source directories ──────────────────────────────────────
    # One or both directories will be searched for <pdb_id>_sasa.csv files.
    # The first match wins.
    sasa_dir_1 = DATA_DIR_3 / "ref" / "SASAs",
    sasa_dir_2 = DATA_DIR_2 / "old_PISCES" / "SASAs",

    # ── Output: where SASA files are moved ──────────────────────────────────
    sasa_moved_dir = DATA_DIR_3 / "ref" / "SASA",

    # ── Safety ──────────────────────────────────────────────────────────────
    # Set to False to perform the actual merge + move.
    dry_run = True,
):
    centrality_dir  = Path(centrality_dir)
    sasa_dir_1      = Path(sasa_dir_1)
    sasa_dir_2      = Path(sasa_dir_2)
    sasa_moved_dir  = Path(sasa_moved_dir)
    sasa_dirs       = [sasa_dir_1, sasa_dir_2]

    # ── 1. Collect centrality CSVs and extract PDB IDs ───────────────────────
    cent_paths = sorted(centrality_dir.glob("*_centrality.csv"))
    print(f"\nScanning centrality directory: {centrality_dir}")
    print(f"  Found {len(cent_paths)} centrality CSV(s)")

    if not cent_paths:
        print("  No centrality CSVs found — nothing to do.")
        return

    # ── 2. Resolve SASA file for each PDB ID ────────────────────────────────
    sasa_found:   dict[str, Path] = {}
    sasa_missing: list[str]       = []

    for cp in cent_paths:
        pdb_id = cp.stem.replace("_centrality", "")
        sasa_path = find_sasa_file(pdb_id, sasa_dirs)
        if sasa_path is not None:
            sasa_found[pdb_id] = sasa_path
        else:
            sasa_missing.append(pdb_id)

    # ── 3. Print plan ────────────────────────────────────────────────────────
    cent_map = {cp.stem.replace("_centrality", ""): cp for cp in cent_paths}
    report_plan(cent_paths, sasa_found, sasa_missing, sasa_moved_dir, dry_run)

    # ── 4. Process each matched protein ─────────────────────────────────────
    sasa_moved_dir.mkdir(parents=True, exist_ok=True)

    merge_log_rows   = []
    move_log_rows    = []
    mismatch_frames  = []

    for pdb_id, sasa_path in sorted(sasa_found.items()):
        cent_path = cent_map[pdb_id]

        # 4a. Merge RSA columns into centrality CSV
        merge_result = merge_rsa_into_centrality(cent_path, sasa_path, dry_run)

        # Collect unmatched residue rows for the mismatch CSV
        unmatched_rows = merge_result.pop("unmatched_rows")
        if not unmatched_rows.empty:
            mismatch_frames.append(unmatched_rows)

        merge_log_rows.append(merge_result)

        # 4b. Move the SASA file
        move_result = move_sasa_file(sasa_path, sasa_moved_dir, dry_run)
        move_result["pdb_id"] = pdb_id
        move_log_rows.append(move_result)

        # Console feedback per protein
        status_str = merge_result["status"]
        matched    = merge_result["n_matched"]
        unmatched  = merge_result["n_unmatched"]
        print(
            f"  {pdb_id:<10} merge={status_str:<25} "
            f"matched={matched}  unmatched={unmatched}"
        )

    # ── 5. Save logs ─────────────────────────────────────────────────────────
    merge_log_df = pd.DataFrame(merge_log_rows)
    move_log_df  = pd.DataFrame(move_log_rows)

    # Flat table of every residue that failed the join — one row per residue
    mismatch_df = (
        pd.concat(mismatch_frames, ignore_index=True)
        if mismatch_frames
        else pd.DataFrame(columns=["pdb_id", "Chain", "Residue", "Residue number"])
    )

    merge_log_path    = centrality_dir / "sasa_merge_log.csv"
    move_log_path     = sasa_moved_dir  / "sasa_move_log.csv"
    mismatch_log_path = centrality_dir / "sasa_mismatch_residues.csv"

    merge_log_df.to_csv(merge_log_path,    index=False)
    move_log_df.to_csv(move_log_path,      index=False)
    mismatch_df.to_csv(mismatch_log_path,  index=False)

    # ── 6. Summary ───────────────────────────────────────────────────────────
    n_merged  = (merge_log_df["status"] == ("dry_run" if dry_run else "merged")).sum() \
                if not merge_log_df.empty else 0
    n_skipped = (merge_log_df["status"] == "skipped_already_merged").sum() \
                if not merge_log_df.empty else 0
    n_moved   = (move_log_df["status"]  == ("dry_run" if dry_run else "moved")).sum()  \
                if not move_log_df.empty else 0
    n_errors  = move_log_df["status"].str.startswith("error").sum() \
                if not move_log_df.empty else 0

    print(f"\n{'DRY RUN — ' if dry_run else ''}Results")
    print(f"  {'Would merge' if dry_run else 'Merged'} centrality CSVs : {n_merged}")
    print(f"  Skipped (RSA already present) : {n_skipped}")
    print(f"  {'Would move' if dry_run else 'Moved'} SASA files        : {n_moved}")
    print(f"  SASA files not found          : {len(sasa_missing)}")
    print(f"  Proteins with mismatches      : {mismatch_df['pdb_id'].nunique()}")
    print(f"  Total mismatched residues     : {len(mismatch_df)}")
    if n_errors:
        print(f"  Move errors                   : {n_errors}")
    print(f"\n  Merge log       : {merge_log_path}")
    print(f"  Move log        : {move_log_path}")
    print(f"  Mismatch log    : {mismatch_log_path}\n")


if __name__ == "__main__":
    main()
