# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_CSV  = DATA_DIR / "enzymes" / "pdb_ids_with_ec.csv"
OUTPUT_CSV = DATA_DIR / "enzymes" / "resolved_prio_ec.csv"
LOG_FILE   = DATA_DIR / "enzymes" / "resolve_prio_ec.log"

RCSB_REST_URL  = "https://data.rcsb.org/rest/v1/core"
SOURCE_PRIORITY = ["UniProt", "PDB Primary Data"]

# Number of parallel threads for RCSB queries.
# Keep this conservative — RCSB rate-limits aggressive clients.
MAX_WORKERS = 8

# ── Logging setup ─────────────────────────────────────────────────────────────
def _setup_logger() -> logging.Logger:
    """
    Configure a logger that writes to both the console and a timestamped log file.
    The log file is placed alongside the input/output CSVs.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("resolve_prio_ec")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # avoid duplicate handlers on re-runs in interactive sessions

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — DEBUG and above
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO and above (keeps terminal readable)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── RCSB helpers ──────────────────────────────────────────────────────────────

def _fetch_entity_ids(pdb_id: str, logger: logging.Logger) -> list[str]:
    """Return all polymer entity IDs for a given PDB entry."""
    url = f"{RCSB_REST_URL}/entry/{pdb_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        ids = data.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
        logger.debug(f"[{pdb_id}] Found {len(ids)} polymer entity IDs: {ids}")
        return ids
    except Exception as exc:
        logger.error(f"[{pdb_id}] Entry fetch ERROR: {exc}")
        return []


def _fetch_provenance_for_entry(
    pdb_id: str,
    logger: logging.Logger,
) -> tuple[str, dict[str, dict[str, str]]]:
    """
    For each chain in a PDB entry, return a mapping:
        chain_id -> { ec_number -> provenance_source }

    Only EC numbers present in rcsb_enzyme_class_combined are included.
    If the same EC appears under multiple sources, the highest-priority source wins.

    Returns (pdb_id, chain_ec_source) so results can be collected from a thread pool.
    """
    pdb_id = pdb_id.strip().upper()
    chain_ec_source: dict[str, dict[str, str]] = {}

    entity_ids = _fetch_entity_ids(pdb_id, logger)
    if not entity_ids:
        logger.warning(f"[{pdb_id}] No entity IDs returned — entry will have no provenance data.")
        return pdb_id, chain_ec_source

    for entity_id in entity_ids:
        url = f"{RCSB_REST_URL}/polymer_entity/{pdb_id}/{entity_id}"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            entity = resp.json()
        except Exception as exc:
            logger.error(f"[{pdb_id}] Entity {entity_id} fetch ERROR: {exc}")
            continue

        # Chain IDs: comma-separated string in entity_poly.pdbx_strand_id
        strand_id_str = entity.get("entity_poly", {}).get("pdbx_strand_id", "")
        chain_ids = [c.strip().upper() for c in strand_id_str.split(",") if c.strip()]

        # Build ec -> best_source map for this entity
        ec_combined = (
            entity.get("rcsb_polymer_entity", {})
                  .get("rcsb_enzyme_class_combined") or []
        )

        ec_to_source: dict[str, str] = {}
        for entry in ec_combined:
            ec  = entry.get("ec", "").strip()
            src = entry.get("provenance_source", "Unknown").strip()
            if not ec:
                continue
            # Keep the highest-priority source seen for this EC
            existing = ec_to_source.get(ec)
            if existing is None:
                ec_to_source[ec] = src
            else:
                existing_rank = SOURCE_PRIORITY.index(existing) if existing in SOURCE_PRIORITY else 999
                new_rank      = SOURCE_PRIORITY.index(src)      if src      in SOURCE_PRIORITY else 999
                if new_rank < existing_rank:
                    ec_to_source[ec] = src

        logger.debug(
            f"[{pdb_id}] Entity {entity_id} → chains {chain_ids}, "
            f"EC map: {ec_to_source}"
        )

        for chain_id in chain_ids:
            if ec_to_source or chain_id not in chain_ec_source:
                chain_ec_source[chain_id] = ec_to_source

        time.sleep(0.03)  # polite inter-entity delay

    return pdb_id, chain_ec_source


# ── Priority selection logic ───────────────────────────────────────────────────

def _select_priority_ec(
    ec_numbers: list[str],
    ec_to_source: dict[str, str],
    logger: logging.Logger,
    pdb_id: str = "",
    chain_id: str = "",
) -> tuple[str, str]:
    """
    Given the EC numbers for a row and the source map from the API, return
    (prio_ec_number, prio_ec_source) according to the project rules:

      1. More than 3 EC numbers → ("-1", "-1")  [handled in caller]
      2. Single EC number       → that EC + its source (or "Unknown")
      3. Multiple EC numbers    → prefer UniProt; if no UniProt present,
                                  fall back to PDB Primary Data; then Unknown.
                                  All EC numbers sharing the winning source
                                  are kept (joined with "; ").
    """
    if not ec_numbers:
        logger.warning(f"[{pdb_id}:{chain_id}] Empty EC list — returning empty strings.")
        return ("", "")

    # Attach source to each EC (fall back to "Unknown" if API had no data for it)
    annotated = [(ec, ec_to_source.get(ec, "Unknown")) for ec in ec_numbers]

    # Single EC — straightforward
    if len(annotated) == 1:
        ec, src = annotated[0]
        if src == "Unknown":
            logger.warning(
                f"[{pdb_id}:{chain_id}] EC {ec} has no provenance data from API — "
                f"source recorded as 'Unknown'."
            )
        logger.debug(f"[{pdb_id}:{chain_id}] Single EC: {ec} (source: {src})")
        return (ec, src)

    # Multiple EC — find best source tier present
    winning_source = None
    for preferred in SOURCE_PRIORITY:
        if any(src == preferred for _, src in annotated):
            winning_source = preferred
            break
    if winning_source is None:
        winning_source = annotated[0][1]  # fallback: whatever the first source is
        logger.warning(
            f"[{pdb_id}:{chain_id}] No preferred source found among {[s for _, s in annotated]}. "
            f"Falling back to '{winning_source}'."
        )

    winning_ecs = sorted(ec for ec, src in annotated if src == winning_source)
    discarded   = [(ec, src) for ec, src in annotated if src != winning_source]

    logger.debug(
        f"[{pdb_id}:{chain_id}] Winning source: '{winning_source}', "
        f"kept ECs: {winning_ecs}, discarded: {discarded}"
    )
    if discarded:
        logger.info(
            f"[{pdb_id}:{chain_id}] Discarded lower-priority ECs: "
            + ", ".join(f"{ec} ({src})" for ec, src in discarded)
        )

    return ("; ".join(winning_ecs), winning_source)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger = _setup_logger()
    logger.info(f"Starting resolve_prio_ec")
    logger.info(f"Input  : {INPUT_CSV}")
    logger.info(f"Output : {OUTPUT_CSV}")
    logger.info(f"Log    : {LOG_FILE}")
    logger.info(f"Workers: {MAX_WORKERS}")

    df = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(df)} rows from input CSV.")

    # Parse ec_number into a list per row
    def parse_ec(val: str) -> list[str]:
        return [e.strip() for e in str(val).split(";") if e.strip()]

    df["_ec_list"] = df["ec_number"].apply(parse_ec)

    # Identify rows with >3 EC numbers — these go to the bottom with -1
    mask_too_many = df["_ec_list"].apply(len) > 3
    df_normal     = df[~mask_too_many].copy()
    df_too_many   = df[mask_too_many].copy()

    logger.info(f"Rows with ≤3 EC numbers (normal)  : {len(df_normal)}")
    logger.info(f"Rows with >3 EC numbers (flagged) : {len(df_too_many)}")
    if len(df_too_many) > 0:
        flagged_ids = df_too_many[["pdb_id", "chain", "ec_number"]].to_string(index=False)
        logger.info(f"Flagged entries:\n{flagged_ids}")

    # ── Parallel RCSB queries ─────────────────────────────────────────────────
    unique_pdb_ids = df_normal["pdb_id"].str.upper().unique().tolist()
    logger.info(f"Querying RCSB for {len(unique_pdb_ids)} unique PDB entries "
                f"using {MAX_WORKERS} parallel workers...")

    pdb_cache: dict[str, dict[str, dict[str, str]]] = {}

    # _fetch_provenance_for_entry already uses internal per-entity sleep(0.03).
    # The inter-entry sleep is removed here because parallelism naturally
    # spreads the load; the per-entity delay is kept as a courtesy to RCSB.
    def _fetch_wrapper(pdb_id: str) -> tuple[str, dict]:
        return _fetch_provenance_for_entry(pdb_id, logger)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_wrapper, pid): pid for pid in unique_pdb_ids}
        with tqdm(total=len(futures), desc="RCSB provenance queries", unit="entry") as bar:
            for future in concurrent.futures.as_completed(futures):
                pdb_id_done = futures[future]
                try:
                    pdb_id_result, chain_map = future.result()
                    pdb_cache[pdb_id_result] = chain_map
                except Exception as exc:
                    logger.error(f"[{pdb_id_done}] Unhandled thread exception: {exc}")
                    pdb_cache[pdb_id_done] = {}
                bar.update(1)

    logger.info("RCSB queries complete.")

    # ── Assign prio_ec_number and prio_ec_source ──────────────────────────────
    prio_ec_numbers = []
    prio_ec_sources = []

    for _, row in df_normal.iterrows():
        pdb_id   = str(row["pdb_id"]).strip().upper()
        chain_id = str(row["chain"]).strip().upper()
        ec_list  = row["_ec_list"]

        ec_to_source = pdb_cache.get(pdb_id, {}).get(chain_id, {})

        if not ec_to_source:
            logger.warning(
                f"[{pdb_id}:{chain_id}] No provenance data found in cache — "
                f"all ECs will be marked 'Unknown'."
            )

        prio_ec, prio_src = _select_priority_ec(
            ec_list, ec_to_source, logger, pdb_id, chain_id
        )
        prio_ec_numbers.append(prio_ec)
        prio_ec_sources.append(prio_src)

    df_normal = df_normal.copy()
    df_normal["prio_ec_number"] = prio_ec_numbers
    df_normal["prio_ec_source"] = prio_ec_sources

    # ── Assign -1 to flagged rows ─────────────────────────────────────────────
    df_too_many = df_too_many.copy()
    df_too_many["prio_ec_number"] = "-1"
    df_too_many["prio_ec_source"] = "-1"

    # ── Reassemble: normal rows first, flagged rows at the bottom ─────────────
    df_out = pd.concat([df_normal, df_too_many], ignore_index=True)
    df_out = df_out.drop(columns=["_ec_list"])

    # Place new columns immediately after ec_number
    ec_idx = df_out.columns.get_loc("ec_number")
    cols   = list(df_out.columns)
    cols.remove("prio_ec_number")
    cols.remove("prio_ec_source")
    cols.insert(ec_idx + 1, "prio_ec_number")
    cols.insert(ec_idx + 2, "prio_ec_source")
    df_out = df_out[cols]

    df_out.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Output written to {OUTPUT_CSV}")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  Total rows        : {len(df_out)}")
    logger.info(f"  Normal rows       : {len(df_normal)}")
    logger.info(f"  Flagged (>3 EC)   : {len(df_too_many)}")

    source_counts = df_normal["prio_ec_source"].value_counts()
    logger.info("Priority source distribution (normal rows):")
    for src, count in source_counts.items():
        logger.info(f"    {src}: {count}")

    unknown_count = (df_normal["prio_ec_source"] == "Unknown").sum()
    if unknown_count > 0:
        logger.warning(
            f"  {unknown_count} row(s) have prio_ec_source = 'Unknown' — "
            f"check the log for entries where the API returned no provenance data."
        )
    logger.info("=" * 60)
    logger.info("Done.")


if __name__ == "__main__":
    main()
