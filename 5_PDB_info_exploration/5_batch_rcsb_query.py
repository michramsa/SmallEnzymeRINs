# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"

QUERY = """
query TestQuery($pdb_id: String!) {
  entry(entry_id: $pdb_id) {
    rcsb_id
    struct {
      title
    }
    assemblies {
      rcsb_id
      pdbx_struct_assembly {
        oligomeric_count
        oligomeric_details
      }
    }
  }
}
"""


def query_single(pdb_id: str) -> list[dict]:
    """
    Query the RCSB API for a single PDB ID.
    Returns a list of row dicts (one per assembly), or a single error row.
    """
    pdb_id = pdb_id.strip().upper()

    try:
        resp = requests.post(
            RCSB_GRAPHQL_URL,
            json={"query": QUERY, "variables": {"pdb_id": pdb_id}},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

    except requests.exceptions.ConnectionError:
        print(f"  [{pdb_id}] ERROR: Could not connect.")
        return [_error_row(pdb_id, "Connection error")]
    except requests.exceptions.Timeout:
        print(f"  [{pdb_id}] ERROR: Request timed out.")
        return [_error_row(pdb_id, "Timeout")]
    except Exception as exc:
        print(f"  [{pdb_id}] ERROR: {exc}")
        return [_error_row(pdb_id, str(exc))]

    entry = (data.get("data") or {}).get("entry")

    if entry is None:
        errors = data.get("errors")
        msg = json.dumps(errors) if errors else "Entry not found"
        print(f"  [{pdb_id}] Not found: {msg}")
        return [_error_row(pdb_id, msg)]

    title = (entry.get("struct") or {}).get("title", "")
    assemblies = entry.get("assemblies") or []

    if not assemblies:
        # Entry exists but has no assembly data
        print(f"  [{pdb_id}] Found (no assemblies listed)")
        return [{
            "pdb_id":              pdb_id,
            "title":               title,
            "assembly_id":         "",
            "oligomeric_details":  "",
            "oligomeric_count":    "",
            "error":               "",
        }]

    rows = []
    for asm in assemblies:
        asm_data = asm.get("pdbx_struct_assembly") or {}
        rows.append({
            "pdb_id":              pdb_id,
            "title":               title,
            "assembly_id":         asm.get("rcsb_id", ""),
            "oligomeric_details":  asm_data.get("oligomeric_details", ""),
            "oligomeric_count":    asm_data.get("oligomeric_count", ""),
            "error":               "",
        })

    print(f"  [{pdb_id}] OK — {len(rows)} assembly row(s)")
    return rows


def _error_row(pdb_id: str, message: str) -> dict:
    return {
        "pdb_id":             pdb_id,
        "title":              "",
        "assembly_id":        "",
        "oligomeric_details": "",
        "oligomeric_count":   "",
        "error":              message,
    }


def batch_query(pdb_ids: list[str], output_csv: str) -> None:
    """
    Query all PDB IDs and write results to a CSV file.
    """
    fieldnames = [
        "pdb_id",
        "title",
        "assembly_id",
        "oligomeric_details",
        "oligomeric_count",
        "error",
    ]

    all_rows = []
    print(f"Querying {len(pdb_ids)} PDB ID(s)...\n")

    for pdb_id in pdb_ids:
        rows = query_single(pdb_id)
        all_rows.extend(rows)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} row(s) written to: {output_csv}")


#### edits from 4/9/2026 - curious about enzyme commission number

RCSB_REST_URL = "https://data.rcsb.org/rest/v1/core"
 
 
def _fetch_entity_ids(pdb_id: str) -> list[str]:
    """
    Fetch the list of polymer entity IDs for a PDB entry via the REST API.
    Returns a list of entity ID strings (e.g. ["1", "2"]), or [] on failure.
    """
    url = f"{RCSB_REST_URL}/entry/{pdb_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  [{pdb_id}] Entry fetch ERROR: {exc}")
        return []
 
    entity_ids = (
        data.get("rcsb_entry_container_identifiers", {})
            .get("polymer_entity_ids", [])
    )
    return [str(e) for e in entity_ids]
 
 
def _fetch_ec_for_entry(pdb_id: str) -> dict[str, str]:
    """
    Query the RCSB REST API for EC numbers for all polymer entities in a PDB entry.
 
    Uses the REST polymer_entity endpoint (not GraphQL) because rcsb_enzyme_class_combined
    is exposed in the REST response but is not available as a GraphQL field.
 
    EC numbers are sourced from rcsb_enzyme_class_combined, which aggregates from
    PDB Primary Data, UniProt, GenBank, KEGG, and author specifications.
 
    Returns a dict mapping chain_id (uppercase) -> EC string (or "").
    Multiple EC numbers are joined with "; " and sorted for determinism.
    """
    pdb_id = pdb_id.strip().upper()
    chain_to_ec: dict[str, str] = {}
 
    entity_ids = _fetch_entity_ids(pdb_id)
    if not entity_ids:
        return chain_to_ec
 
    for entity_id in entity_ids:
        url = f"{RCSB_REST_URL}/polymer_entity/{pdb_id}/{entity_id}"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            entity = resp.json()
        except Exception as exc:
            print(f"  [{pdb_id}] Entity {entity_id} fetch ERROR: {exc}")
            continue
 
        # Chain IDs: comma-separated string in entity_poly.pdbx_strand_id
        strand_id_str = entity.get("entity_poly", {}).get("pdbx_strand_id", "")
        chain_ids = [c.strip().upper() for c in strand_id_str.split(",") if c.strip()]
 
        # EC numbers from rcsb_enzyme_class_combined inside rcsb_polymer_entity
        ec_combined = entity.get("rcsb_polymer_entity", {}).get("rcsb_enzyme_class_combined") or []
        ec_set = {e["ec"] for e in ec_combined if e.get("ec")}
 
        ec_str = "; ".join(sorted(ec_set))
 
        for chain_id in chain_ids:
            # Only overwrite if we have a non-empty EC, or if the chain hasn't been
            # seen yet. This prevents a later entity with a lowercase chain alias
            # (e.g. "a" -> "A") and no EC from clobbering a valid result already set
            # by an earlier entity for the same uppercased chain ID.
            if ec_str or chain_id not in chain_to_ec:
                chain_to_ec[chain_id] = ec_str
 
        time.sleep(0.03)  # small delay between entity calls
 
    return chain_to_ec
 
def _checkpoint_path(output_csv: Path) -> Path:
    """Return the checkpoint file path alongside the output CSV."""
    return output_csv.with_suffix(".checkpoint.json")
 
 
def _load_checkpoint(checkpoint_file: Path) -> dict[str, dict[str, str]]:
    """
    Load previously fetched EC results from a checkpoint file.
 
    Returns a dict:  { "PDBID": { "CHAIN": "ec_str", ... }, ... }
    An empty dict is returned if no checkpoint exists.
    """
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
        print(f"  Resuming from checkpoint: {len(data)} PDB IDs already fetched.")
        return data
    return {}
 
 
def _save_checkpoint(checkpoint_file: Path, completed: dict[str, dict[str, str]]) -> None:
    """Persist completed results to the checkpoint file via a temp file to avoid corruption."""
    tmp = checkpoint_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(completed, f)
    tmp.replace(checkpoint_file)
 
 
def add_ec_numbers(input_csv: str | Path, output_csv: str | Path) -> None:
    """
    Read a BioLiP-derived CSV, query RCSB for EC numbers per chain, and write
    a new CSV with an 'ec_number' column inserted between 'chain' and
    'oligomeric_state'.
 
    Supports stop-and-resume: progress is saved to a checkpoint file
    (<output_csv>.checkpoint.json) after every PDB ID is queried. If the
    process is interrupted and restarted with the same output_csv path,
    already-fetched entries are skipped automatically. The checkpoint file is
    deleted on successful completion.
 
    Parameters
    ----------
    input_csv  : path to the input CSV (must have at least 'pdb_id' and 'chain' columns)
    output_csv : path for the output CSV
    """
    input_csv  = Path(input_csv)
    output_csv = Path(output_csv)
    checkpoint_file = _checkpoint_path(output_csv)
 
    df = pd.read_csv(input_csv, dtype=str)
 
    # Validate expected columns
    required = {"pdb_id", "chain"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
 
    # --- Load checkpoint (if resuming) ---
    # Structure: { "PDBID": { "CHAIN_A": "ec_str", ... }, ... }
    completed: dict[str, dict[str, str]] = _load_checkpoint(checkpoint_file)
 
    unique_pdb_ids = df["pdb_id"].str.upper().unique().tolist()
    remaining = [p for p in unique_pdb_ids if p.upper() not in completed]
 
    print(f"Total unique PDB entries : {len(unique_pdb_ids)}")
    print(f"Already fetched          : {len(completed)}")
    print(f"Remaining to query       : {len(remaining)}\n")
 
    # --- Query remaining entries, checkpointing after each ---
    with tqdm(total=len(remaining), desc="RCSB EC queries", unit="entry", dynamic_ncols=True) as bar:
        for pdb_id in remaining:
            bar.set_postfix({"current": pdb_id, "done": len(completed)}, refresh=True)
            chain_to_ec = _fetch_ec_for_entry(pdb_id)
            # Always write the entry — even if empty — so we don't re-query it on resume
            completed[pdb_id.upper()] = chain_to_ec
            _save_checkpoint(checkpoint_file, completed)
            bar.update(1)
            time.sleep(0.05)  # be polite to the API
 
    # --- Flatten checkpoint into (pdb_id, chain) -> ec_str lookup ---
    ec_map: dict[tuple[str, str], str] = {}
    for pdb_id, chain_dict in completed.items():
        for chain_id, ec_str in chain_dict.items():
            ec_map[(pdb_id.upper(), chain_id.upper())] = ec_str
 
    # --- Map EC numbers onto the dataframe ---
    def lookup_ec(row):
        key = (str(row["pdb_id"]).upper(), str(row["chain"]).upper())
        return ec_map.get(key, "")
 
    df["ec_number"] = df.apply(lookup_ec, axis=1)
 
    # --- Reorder columns: insert ec_number between 'chain' and 'oligomeric_state' ---
    cols = list(df.columns)
    cols.remove("ec_number")
 
    insert_after = "chain"
    if insert_after in cols:
        idx = cols.index(insert_after) + 1
    else:
        idx = len(cols)  # fallback: append at end
 
    cols.insert(idx, "ec_number")
    df = df[cols]
 
    # --- Summary ---
    n_total = len(df)
    n_with_ec = (df["ec_number"] != "").sum()
    print(f"\nEC number coverage: {n_with_ec}/{n_total} chains ({100 * n_with_ec / n_total:.1f}%)")
    print(f"Chains with EC:    {n_with_ec}")
    print(f"Chains without EC: {n_total - n_with_ec}  (non-enzymes or unannotated)")
 
    df.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")
 
    # --- Clean up checkpoint on successful completion ---
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"Checkpoint removed: {checkpoint_file}")

# 4/10/2026 edits for the complex type

def _fetch_assembly_info(pdb_id: str, assembly_id: str = "1") -> dict:
    """
    Fetch rcsb_assembly_info for a given PDB entry and biological assembly ID.
 
    Returns the rcsb_assembly_info dict, or {} on failure.
    The default assembly_id "1" is the primary biological assembly for the
    vast majority of PDB entries.
    """
    pdb_id = pdb_id.strip().upper()
    url = f"{RCSB_REST_URL}/assembly/{pdb_id}/{assembly_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json().get("rcsb_assembly_info", {})
    except Exception as exc:
        print(f"  [{pdb_id}] Assembly fetch ERROR: {exc}")
        return {}
 
 
def _classify_complex(assembly_info: dict) -> str:
    """
    Classify a biological assembly as monomer/homo/hetero/other/unknown using
    pre-computed counts from rcsb_assembly_info:
 
        polymer_entity_instance_count_protein : total protein chain instances
        polymer_entity_count_protein          : number of distinct protein sequences
 
    Classification:
        monomer  — 1 protein chain instance
        homo     — 2–3 chain instances, all from 1 distinct entity (identical sequence)
        hetero   — 2–3 chain instances, from 2+ distinct entities (different sequences)
        other    — 4+ chain instances (tetramers, large complexes, etc.)
        unknown  — assembly info missing or no protein chains found
    """
    if not assembly_info:
        return "unknown"
 
    total_instances = assembly_info.get("polymer_entity_instance_count_protein", 0)
    distinct_entities = assembly_info.get("polymer_entity_count_protein", 0)
 
    if total_instances == 0 or distinct_entities == 0:
        return "unknown"
 
    if total_instances == 1:
        return "monomer"
 
    if total_instances in (2, 3):
        return "homo" if distinct_entities == 1 else "hetero"
 
    return "other"
 
 
def add_complex_type(input_csv: str | Path, output_csv: str | Path) -> None:
    """
    Read a BioLiP-derived CSV and add a 'complex_type' column after 'oligomeric_state'.
 
    Classification is based on the number of distinct protein entities in the
    biological assembly:
 
        monomer  — 1 protein chain
        homo     — 2–3 chains, all sharing the same sequence (same entity)
        hetero   — 2–3 chains, from 2+ distinct sequences (different entities)
        other    — 4+ chains, or any assembly outside dimer/trimer scope
        unknown  — API fetch failed or no protein entities found
 
    Only protein polymer entities (polypeptide(L) / polypeptide(D)) are counted.
    Nucleic acids, carbohydrates, and cofactor chains are ignored.
 
    Supports stop-and-resume via a checkpoint file (<output_csv>.checkpoint.json).
 
    Parameters
    ----------
    input_csv  : path to input CSV (must have at least 'pdb_id' column)
    output_csv : path for the output CSV
    """
    input_csv  = Path(input_csv)
    output_csv = Path(output_csv)
    checkpoint_file = _checkpoint_path(output_csv)
 
    df = pd.read_csv(input_csv, dtype=str)
 
    required = {"pdb_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
 
    # Checkpoint stores: { "PDBID": "classification_string", ... }
    completed: dict[str, str] = _load_checkpoint(checkpoint_file)
 
    unique_pdb_ids = df["pdb_id"].str.upper().unique().tolist()
    remaining = [p for p in unique_pdb_ids if p.upper() not in completed]
 
    print(f"Total unique PDB entries : {len(unique_pdb_ids)}")
    print(f"Already fetched          : {len(completed)}")
    print(f"Remaining to classify    : {len(remaining)}\n")
 
    with tqdm(total=len(remaining), desc="Complex type classification", unit="entry", dynamic_ncols=True) as bar:
        for pdb_id in remaining:
            bar.set_postfix({"current": pdb_id, "done": len(completed)}, refresh=True)
            assembly_info = _fetch_assembly_info(pdb_id)
            classification = _classify_complex(assembly_info)
            completed[pdb_id.upper()] = classification
            _save_checkpoint(checkpoint_file, completed)
            bar.update(1)
            time.sleep(0.05)
 
    # Map classification onto every row by pdb_id
    df["complex_type"] = df["pdb_id"].str.upper().map(completed).fillna("unknown")
 
    # Insert complex_type after oligomeric_state if present, else after chain
    cols = list(df.columns)
    cols.remove("complex_type")
    for anchor in ("oligomeric_state", "chain"):
        if anchor in cols:
            cols.insert(cols.index(anchor) + 1, "complex_type")
            break
    else:
        cols.append("complex_type")
    df = df[cols]
 
    # Summary
    print("\nComplex type distribution:")
    print(df["complex_type"].value_counts().to_string())
 
    df.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")
 
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"Checkpoint removed: {checkpoint_file}")

# edits on 4/14/26
#
# New function: add_structure_metadata
#
# Adds two columns to an existing CSV:
#   - resolution_angstroms : float (X-ray/EM resolution); empty string if unavailable (e.g. NMR)
#   - has_bio_assembly     : True if RCSB lists at least one biological assembly for the entry,
#                            False otherwise (rare, but possible for some NMR or theoretical models)
#
# Note: asu_exists column was intentionally omitted — every deposited PDB entry has an ASU
# by definition, so the column would be True for all rows. Document this assumption in methods.
#
# Resolution source:
#   Uses rcsb_entry_info.resolution_combined from the REST /entry endpoint.
#   This field is preferred over refine.ls_d_res_high because it is populated for
#   X-ray, cryo-EM, and neutron diffraction entries alike. NMR structures have no
#   experimental resolution and will return an empty string.
#
# Biological assembly check:
#   Queries the REST /entry endpoint for rcsb_entry_container_identifiers.assembly_ids.
#   If that list is non-empty, has_bio_assembly = True.
#
# Supports stop-and-resume via checkpoint (same pattern as add_ec_numbers / add_complex_type).
#
# Logging:
#   Errors-only log written to DATA_DIR / "logs" / add_structure_metadata_{timestamp}.log
#   The log file is created at the start of the run and appended to on resume.
#   Each error line records: timestamp, PDB ID, and the error message.
#   A run summary (total processed, error count) is written at the end.


import logging


def _setup_logger(log_dir: Path, func_name: str) -> logging.Logger:
    """
    Create a file-based logger that writes errors (and above) to a timestamped log file.

    Log file is placed in log_dir and named {func_name}_{YYYYMMDD_HHMMSS}.log.
    The logger uses append mode so partial runs accumulate in the same session's log
    when stop-and-resume is used within the same day; a new file is created each
    fresh invocation.

    Parameters
    ----------
    log_dir   : directory where the log file will be written (created if absent)
    func_name : used as both the logger name and the log filename prefix
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path  = log_dir / f"{func_name}_{timestamp}.log"

    logger = logging.getLogger(f"{func_name}_{timestamp}")
    logger.setLevel(logging.ERROR)

    # File handler — errors only, append mode
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    logger.addHandler(fh)

    print(f"  Log file: {log_path}")
    return logger


def _fetch_structure_metadata(pdb_id: str) -> tuple[dict, str | None]:
    """
    Fetch resolution and biological assembly existence for a single PDB entry.

    Uses the REST /entry endpoint, which is already called by _fetch_entity_ids,
    so no additional endpoint is needed.

    Returns
    -------
    result : dict with keys:
                 resolution_angstroms : str  — numeric string (e.g. "1.8") or "" if unavailable
                 has_bio_assembly     : bool — True if >=1 biological assembly is listed
    error  : str | None — error message if the fetch failed, None on success
    """
    pdb_id = pdb_id.strip().upper()
    url    = f"{RCSB_REST_URL}/entry/{pdb_id}"

    result = {
        "resolution_angstroms": "",
        "has_bio_assembly":     False,
    }

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return result, str(exc)  # return defaults + error message

    # --- Resolution ---
    # resolution_combined is populated for X-ray, cryo-EM, and neutron diffraction.
    # NMR entries do not have this field (returns None) — empty string is the correct default.
    resolution = (
        data.get("rcsb_entry_info", {})
            .get("resolution_combined")
    )
    if resolution is not None:
        result["resolution_angstroms"] = str(resolution)

    # --- Biological assembly existence ---
    # assembly_ids is a list of strings e.g. ["1"] or ["1", "2"].
    # Missing or empty list → no biological assembly recorded for this entry.
    assembly_ids = (
        data.get("rcsb_entry_container_identifiers", {})
            .get("assembly_ids") or []
    )
    result["has_bio_assembly"] = len(assembly_ids) > 0

    return result, None  # None = no error


def add_structure_metadata(input_csv: str | Path, output_csv: str | Path) -> None:
    """
    Read an existing CSV and add two structural metadata columns:
        resolution_angstroms, has_bio_assembly

    Columns are inserted after 'n_chains_biological_assembly' if present,
    otherwise appended at the end.

    Supports stop-and-resume via a checkpoint file alongside the output CSV.
    Errors are written to DATA_DIR / "logs" / add_structure_metadata_{timestamp}.log

    Parameters
    ----------
    input_csv  : path to input CSV (must have at least a 'pdb_id' column)
    output_csv : path for the output CSV
    """
    input_csv  = Path(input_csv)
    output_csv = Path(output_csv)
    checkpoint_file = _checkpoint_path(output_csv)

    # --- Set up error logger ---
    log_dir = DATA_DIR / "logs"
    logger  = _setup_logger(log_dir, "add_structure_metadata")

    df = pd.read_csv(input_csv, dtype=str)

    if "pdb_id" not in df.columns:
        raise ValueError("Input CSV is missing required 'pdb_id' column.")

    # Checkpoint stores: { "PDBID": { "resolution_angstroms": ..., "has_bio_assembly": ... } }
    completed: dict[str, dict] = _load_checkpoint(checkpoint_file)

    unique_pdb_ids = df["pdb_id"].str.upper().unique().tolist()
    remaining      = [p for p in unique_pdb_ids if p.upper() not in completed]

    print(f"Total unique PDB entries : {len(unique_pdb_ids)}")
    print(f"Already fetched          : {len(completed)}")
    print(f"Remaining to fetch       : {len(remaining)}\n")

    error_count = 0

    with tqdm(total=len(remaining), desc="Structure metadata", unit="entry", dynamic_ncols=True) as bar:
        for pdb_id in remaining:
            bar.set_postfix({"current": pdb_id, "done": len(completed)}, refresh=True)

            meta, error = _fetch_structure_metadata(pdb_id)

            if error:
                error_count += 1
                logger.error("[%s] %s", pdb_id, error)

            completed[pdb_id.upper()] = meta
            _save_checkpoint(checkpoint_file, completed)
            bar.update(1)
            time.sleep(0.05)  # polite API usage

    # --- Write run summary to log ---
    # Use a direct file write for the summary so it appears even if error_count == 0
    # (logging.ERROR level would suppress an info-level summary line).
    summary = (
        f"\n--- Run summary ---\n"
        f"  Entries processed : {len(remaining)}\n"
        f"  Errors logged     : {error_count}\n"
        f"  Checkpoint file   : {checkpoint_file}\n"
        f"  Output CSV        : {output_csv}\n"
    )
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.stream.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}  INFO  {summary}\n"
            )

    print(summary)

    # --- Map fetched metadata onto every row by pdb_id ---
    new_cols = ["resolution_angstroms", "has_bio_assembly"]

    for col in new_cols:
        df[col] = df["pdb_id"].str.upper().map(
            lambda pid, c=col: completed.get(pid, {}).get(c, "")
        )

    # --- Column insertion: after 'n_chains_biological_assembly' if present ---
    cols = list(df.columns)
    for col in new_cols:
        cols.remove(col)

    anchor = "n_chains_biological_assembly"
    if anchor in cols:
        insert_at = cols.index(anchor) + 1
    else:
        insert_at = len(cols)  # fallback: append at end

    for i, col in enumerate(new_cols):
        cols.insert(insert_at + i, col)

    df = df[cols]

    # --- Console summary ---
    n_total    = len(df)
    n_with_res = (df["resolution_angstroms"] != "").sum()
    n_with_asm = (df["has_bio_assembly"].astype(str).str.lower() == "true").sum()

    print(f"Resolution available : {n_with_res}/{n_total} entries")
    print(f"  (NMR or missing data accounts for the remainder)")
    print(f"Has bio assembly     : {n_with_asm}/{n_total} entries")

    df.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")

    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"Checkpoint removed: {checkpoint_file}")


if __name__ == "__main__":

    # all the Amitai proteins used in the supplemental s1
    PDB_IDS = [
        "13PK", 
        "1AH7", 
        "1AKM",
        "1AL6", 
        "1ALK", 
        "1B3R", 
        "1BG0", 
        "1BML", 
        "1BOL", 
        "1BRM", 
        "1BWD", 
        "1BWP", 
        "1D4A", 
        "1D4C", 
        "1DII", 
        "1DNK", 
        "1EF8", 
        "1GET", 
        "1GPA", 
        "1GRC", 
        "1HFS", 
        "1HXQ", 
        "1LXA", 
        "1MAS", 
        "1MBB", 
        "1MHY", 
        "1MPY", 
        "1OFG", 
        "1PFK", 
        "1RA2", 
        "1SMN", 
        "1ULA", 
        "1UOK", 
        "1UOX", 
        "1VAO", 
        "1VNC", 
        "1XVA", 
        "1ZIO", 
        "2BBK", 
        "2JCW", 
        "2THI", 
        "3ECA"
    ]

    # OUTPUT_CSV = DATA_DIR / "Amitai RCSB query"/ "rcsb_results.csv"

    # batch_query(PDB_IDS, OUTPUT_CSV)

#     add_ec_numbers(
#     input_csv  = DATA_DIR / "FASTA"/ "oligomeric_biolip_func_residues.csv",
#     output_csv = DATA_DIR / "FASTA"/ "oligomeric_biolip_func_residues_with_ec.csv"
# )

    # add_complex_type(
    #     input_csv  = DATA_DIR / "FASTA" / "oligomeric_biolip_func_residues_with_ec.csv",
    #     output_csv = DATA_DIR / "FASTA" / "oligo_bio_func_residues_with_complex_type.csv"
    # )

    # --- 4/14/26: add resolution_angstroms, asu_exists, has_bio_assembly ---
    # Run after add_complex_type has produced the complex_type CSV.
    # Input should be your most recent CSV; output gets the three new columns.
    add_structure_metadata(
        input_csv  = DATA_DIR / "FASTA" / "oligo_bio_func_residues_with_complex_type.csv",
        output_csv = DATA_DIR / "FASTA" / "oligo_bio_func_residues_with_metadata.csv"
    )
