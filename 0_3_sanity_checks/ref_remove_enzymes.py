# 0_4_ref_remove_enzymes_ver1.py
# ============================================================
# Read the ref_ec_lookup.csv produced by 0_4_ref_enzyme_check_ver1.py
# and move any PDBID_clean.json files identified as enzymes or
# flagged with an API error out of the reference JSON directory
# into a separate removal directory.
#
# Files are MOVED (not deleted) so nothing is lost — review the
# removal directory and delete permanently only when satisfied.
#
# A dry-run mode (enabled by default) prints what would be moved
# without touching the filesystem.  Set dry_run=False to execute.
#
# Outputs
#   <removal_dir>/           – destination for removed JSON files
#   <removal_dir>/removal_log.csv  – record of every move with reason
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_removal_targets(lookup_csv: Path) -> pd.DataFrame:
    """
    Read ref_ec_lookup.csv and return rows where the JSON should be removed:
      - is_enzyme == True   : confirmed enzyme
      - api_error == True   : enzyme status could not be determined

    Returns a DataFrame with columns: pdb_id, ec_number, is_enzyme,
    api_error, reason.
    """
    df = pd.read_csv(lookup_csv)

    required = {"pdb_id", "is_enzyme", "api_error"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"lookup CSV is missing expected columns: {missing}. "
            "Was it produced by 0_4_ref_enzyme_check_ver1.py?"
        )

    df["pdb_id"] = df["pdb_id"].str.upper().str.strip()

    # Assign human-readable removal reason (api_error takes priority in label)
    conditions = [
        df["api_error"],
        df["is_enzyme"] & ~df["api_error"],
    ]
    choices = [
        "api_error",
        "enzyme",
    ]
    df["reason"] = np.select(conditions, choices, default="keep")

    targets = df[df["reason"] != "keep"].copy().reset_index(drop=True)
    return targets


def build_file_map(
    targets: pd.DataFrame,
    json_dir: Path,
) -> list[dict]:
    """
    For each target PDB ID, resolve the expected PDBID_clean.json path.

    Returns a list of dicts with keys:
        pdb_id, reason, ec_number, src, found
    where `found` is False if the file does not exist in json_dir.
    """
    file_map = []
    for _, row in targets.iterrows():
        pid      = row["pdb_id"]
        filename = f"{pid}_clean.json"
        src      = json_dir / filename
        file_map.append({
            "pdb_id":    pid,
            "reason":    row["reason"],
            "ec_number": row.get("ec_number", ""),
            "src":       src,
            "found":     src.exists(),
        })
    return file_map


def report(file_map: list[dict], removal_dir: Path, dry_run: bool) -> None:
    """Print a pre-move summary to stdout."""
    found    = [f for f in file_map if f["found"]]
    missing  = [f for f in file_map if not f["found"]]
    enzymes  = [f for f in found if f["reason"] == "enzyme"]
    errors   = [f for f in found if f["reason"] == "api_error"]

    mode_label = "DRY RUN — no files will be moved" if dry_run else "LIVE RUN"
    print(f"\n{'=' * 50}")
    print(f"  Removal plan  [{mode_label}]")
    print(f"{'=' * 50}")
    print(f"  Destination   : {removal_dir}")
    print(f"  Files to move : {len(found)}")
    print(f"    Enzymes     : {len(enzymes)}")
    print(f"    API errors  : {len(errors)}")
    if missing:
        print(f"  Not in dir    : {len(missing)}  (listed below)")

    if found:
        print("\n  Files queued for move:")
        for f in sorted(found, key=lambda x: x["pdb_id"]):
            ec_note = f"  [{f['ec_number']}]" if f["ec_number"] else ""
            print(f"    {f['src'].name:<30}  reason={f['reason']}{ec_note}")

    if missing:
        print("\n  PDB IDs in lookup CSV but NOT found in json_dir:")
        for f in sorted(missing, key=lambda x: x["pdb_id"]):
            print(f"    {f['pdb_id']}  (expected: {f['src'].name})")

    print()


def move_files(
    file_map: list[dict],
    removal_dir: Path,
    dry_run: bool,
) -> pd.DataFrame:
    """
    Move files from json_dir to removal_dir.

    Returns a log DataFrame recording the outcome of each attempted move.
    """
    removal_dir.mkdir(parents=True, exist_ok=True)

    log_rows = []
    for f in file_map:
        dst    = removal_dir / f["src"].name
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
            "reason":    f["reason"],
            "ec_number": f["ec_number"],
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
    # ── Inputs ──────────────────────────────────────────────────────────────
    # CSV produced by 0_4_ref_enzyme_check_ver1.py
    lookup_csv  = DATA_DIR_3 / "ref" / "ref_enzyme_check" / "ref_ec_lookup.csv",

    # Directory containing the PDBID_clean.json reference files
    json_dir    = DATA_DIR_3 / "ref" / "JSONs" / "structures",

    # ── Output ──────────────────────────────────────────────────────────────
    # Subdirectory where removed JSON files will be placed
    removal_dir = DATA_DIR_3 / "ref" / "ref_JSONs_removed",

    # ── Safety ──────────────────────────────────────────────────────────────
    # Set to False to perform the actual move. True by default so the first
    # run is always a safe preview.
    dry_run     = True,
):
    lookup_csv  = Path(lookup_csv)
    json_dir    = Path(json_dir)
    removal_dir = Path(removal_dir)

    # ── 1. Load removal targets from lookup CSV ──────────────────────────────
    print(f"Reading lookup CSV: {lookup_csv}")
    targets = load_removal_targets(lookup_csv)
    print(f"  {len(targets)} PDB ID(s) flagged for removal "
          f"({(targets['reason'] == 'enzyme').sum()} enzymes, "
          f"{(targets['reason'] == 'api_error').sum()} API errors).")

    # ── 2. Resolve file paths ────────────────────────────────────────────────
    file_map = build_file_map(targets, json_dir)

    # ── 3. Print plan ────────────────────────────────────────────────────────
    report(file_map, removal_dir, dry_run)

    # ── 4. Move (or simulate) ────────────────────────────────────────────────
    log_df = move_files(file_map, removal_dir, dry_run)

    # ── 5. Save log ──────────────────────────────────────────────────────────
    log_path = removal_dir / "removal_log.csv"
    removal_dir.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)

    # ── 6. Print outcome summary ─────────────────────────────────────────────
    moved   = (log_df["status"] == "moved").sum()
    dry     = (log_df["status"] == "dry_run").sum()
    skipped = (log_df["status"] == "skipped_missing").sum()
    errors  = log_df["status"].str.startswith("error").sum()

    if dry_run:
        print(f"Dry run complete — {dry} file(s) would be moved.")
        print("Set dry_run=False in main() to execute.")
    else:
        print(f"Move complete — {moved} moved, {skipped} not found, {errors} errors.")

    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main(dry_run     = False)
