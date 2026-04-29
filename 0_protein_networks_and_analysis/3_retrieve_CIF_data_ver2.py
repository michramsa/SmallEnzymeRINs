# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

# This file takes a CSV of PDB entries
# and performs the following:
# 1. Load the CSV and extract PDB IDs.
# 2. Download mmCIF files in parallel from RCSB, skipping already-downloaded files.
#    (mmCIF is used instead of the legacy PDB format, which is being phased out
#    by the wwPDB and is already unavailable for large structures and for entries
#    with extended PDB IDs. PDBe Arpeggio accepts mmCIF natively.)
# 3. Optionally filter for mutants and write filtered outputs.

# --- Logging setup ---
log_dir = BASE_DIR / "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = log_dir / f"retrieve_PDB_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Loads the input CSV file and returns a DataFrame.
# Input: path to CSV file
# Output: pandas DataFrame
def load_csv(csv_file):
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} entries from {csv_file}")
    return df


# Extracts unique PDB IDs from the 'pdb_id' column of a DataFrame.
# Input: DataFrame with 'pdb_id' column
# Output: list of unique PDB ID strings
def extract_pdb_ids(df):
    pdb_ids = df['pdb_id'].dropna().unique().tolist()
    logger.info(f"Found {len(pdb_ids)} unique PDB IDs")
    return pdb_ids


# Writes a list of PDB IDs to a comma-separated text file suitable for batch_download.sh.
# Input: list of PDB ID strings, output file path, output directory path
# Output: None
def write_pdb_id_list(pdb_id_list, output_file, output_directory):
    logger.info("Creating PDB ID text file")
    os.makedirs(output_directory, exist_ok=True)
    comma_separated = ",".join([pid.upper() for pid in pdb_id_list])
    with open(output_file, 'w') as f:
        f.write(comma_separated)
    logger.info(f"PDB ID list written to {output_file}")


# Runs the RCSB batch download shell script to retrieve mmCIF files.
# Uses the '-c' flag for cif.gz (legacy PDB format is being phased out).
# Input: path to the text file of PDB IDs, path to the output directory for CIF files
# Output: None (CIF files are written to output_directory)
def PDB_script(script_file, output_directory):
    script_path = BASE_DIR / "batch_download.sh"
    script_param = '-f'
    type_param = '-c'  # -c = cif.gz (was -p for pdb.gz in the legacy version)

    result = subprocess.run(
        ['bash', str(script_path), script_param, str(script_file), '-o', str(output_directory), type_param],
        capture_output=True,
        text=True,
        check=True
    )
    logger.info(f"batch_download.sh stdout: {result.stdout}")
    if result.stderr:
        logger.warning(f"batch_download.sh stderr: {result.stderr}")
    logger.info(f"CIF files downloaded to {output_directory}")


# Downloads a single mmCIF file (.cif.gz) from RCSB. Skips if already present.
# Input: pdb_id string, output directory Path, max retries int
# Output: tuple of (pdb_id, status_string)  where status is 'downloaded', 'skipped', or 'failed'
def _download_single_cif(pdb_id, output_directory, max_retries=3):
    pdb_id = pdb_id.upper()
    out_path = Path(output_directory) / f"{pdb_id}.cif.gz"

    if out_path.exists():
        return (pdb_id, "skipped")

    url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(out_path, 'wb') as f:
                    f.write(response.content)
                return (pdb_id, "downloaded")
            elif response.status_code == 404:
                return (pdb_id, "failed:404")
            else:
                if attempt == max_retries:
                    return (pdb_id, f"failed:http{response.status_code}")
        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                return (pdb_id, f"failed:{str(e)}")
            time.sleep(2 ** attempt)  # exponential backoff

    return (pdb_id, "failed:unknown")


# Downloads mmCIF files in parallel from RCSB, skipping any already present on disk.
# Input: list of PDB ID strings, output directory Path, number of worker threads int
# Output: dict mapping pdb_id -> status ('downloaded', 'skipped', 'failed:...')
def download_cifs_parallel(pdb_id_list, output_directory, n_workers=16):
    os.makedirs(output_directory, exist_ok=True)
    total = len(pdb_id_list)
    logger.info(f"Starting parallel download: {total} PDB IDs, {n_workers} workers")
    logger.info(f"Output directory: {output_directory}")

    results = {}
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_download_single_cif, pdb_id, output_directory): pdb_id
            for pdb_id in pdb_id_list
        }
        with tqdm(total=total, desc="Downloading CIFs", unit="file") as pbar:
            for future in concurrent.futures.as_completed(futures):
                pdb_id, status = future.result()
                results[pdb_id] = status

                if status == "downloaded":
                    counts["downloaded"] += 1
                elif status == "skipped":
                    counts["skipped"] += 1
                else:
                    counts["failed"] += 1
                    logger.warning(f"Failed to download {pdb_id}: {status}")

                pbar.update(1)
                pbar.set_postfix(counts)

    logger.info(
        f"Download complete — "
        f"downloaded: {counts['downloaded']}, "
        f"skipped (already present): {counts['skipped']}, "
        f"failed: {counts['failed']}"
    )

    failed_ids = [pid for pid, s in results.items() if s.startswith("failed")]
    if failed_ids:
        logger.warning(f"Failed IDs ({len(failed_ids)}): {failed_ids}")

    return results

# --- Mutant detection ---------------------------------------------------------
#
# Design note:
#   Amitai et al. (2004) excluded "no mutants" from their baseline connectivity
#   dataset by relying on the (now-defunct) PDB-REPRDB database. Since we have
#   to operationalize "mutant" ourselves, we use the RCSB-precomputed integer
#   field entity_poly.rcsb_mutation_count, which is derived from
#   struct_ref_seq_dif records annotated "ENGINEERED MUTATION" (residues in
#   the deposited structure that differ from the canonical UniProt reference
#   and were flagged by curators as engineered, not natural).
#
#   An entry is classified as a mutant if any of its polymer entities has
#   rcsb_mutation_count > 0. We do NOT flag natural sequence variants, splice
#   isoforms, or truncations as mutants under this definition.
#
# Implementation note:
#   We use the RCSB GraphQL endpoint to fetch counts for many entries in a
#   single request (batched) and parallelize across batches.

RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"

_MUTATION_QUERY = """
query MutationCheck($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    polymer_entities {
      rcsb_id
      entity_poly {
        rcsb_mutation_count
      }
    }
  }
}
"""


# Queries the RCSB GraphQL API for rcsb_mutation_count on a batch of PDB IDs.
# Input: list of PDB ID strings (ideally <= 50 per batch), max retries
# Output: dict mapping uppercased pdb_id -> dict with keys:
#     'is_mutant' (bool), 'total_count' (int), 'n_entities' (int),
#     'status' ('ok' | 'missing' | 'error:...')
def _query_mutation_batch(pdb_id_batch, max_retries=3):
    ids = [pid.upper() for pid in pdb_id_batch]
    payload = {"query": _MUTATION_QUERY, "variables": {"ids": ids}}

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                RCSB_GRAPHQL_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()
                if "errors" in data and data["errors"]:
                    last_error = f"graphql_errors:{data['errors']}"
                    break
                return _parse_mutation_response(data, ids)
            else:
                last_error = f"http{response.status_code}"
                if 500 <= response.status_code < 600 and attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                break
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue

    # If we get here, all retries failed for the whole batch. Mark every ID
    # as errored so the caller knows not to trust these results.
    return {pid: {"is_mutant": False, "total_count": 0,
                  "n_entities": 0, "status": f"error:{last_error}"}
            for pid in ids}

# Parses a successful GraphQL response into the per-entry dict format.
# Decision rule: an entry is a mutant if any of its polymer entities has
# entity_poly.rcsb_mutation_count > 0.
# Input: GraphQL JSON response dict, list of requested uppercased IDs
# Output: dict mapping pdb_id -> {is_mutant, total_count, n_entities, status}
def _parse_mutation_response(data, requested_ids):
    out = {pid: {"is_mutant": False, "total_count": 0,
                 "n_entities": 0, "status": "missing"}
           for pid in requested_ids}

    entries = (data.get("data") or {}).get("entries") or []
    for entry in entries:
        if entry is None:
            continue
        pdb_id = (entry.get("rcsb_id") or "").upper()
        if pdb_id not in out:
            continue

        polymer_entities = entry.get("polymer_entities") or []
        total_count = 0
        for pe in polymer_entities:
            if pe is None:
                continue
            ep = pe.get("entity_poly") or {}
            count = ep.get("rcsb_mutation_count")
            if isinstance(count, int) and count > 0:
                total_count += count

        out[pdb_id] = {
            "is_mutant": total_count > 0,
            "total_count": total_count,
            "n_entities": len(polymer_entities),
            "status": "ok",
        }

    return out


# Checks a list of PDB IDs for engineered mutations via RCSB GraphQL.
# Batches requests and parallelizes across batches.
# Input: list of PDB ID strings, batch size, number of worker threads
# Output: dict mapping pdb_id -> {is_mutant, total_count, n_entities, status}
def check_mutations_bulk(pdb_id_list, batch_size=50, n_workers=4):
    unique_ids = sorted({pid.upper() for pid in pdb_id_list if pid})
    total = len(unique_ids)
    logger.info(f"Checking {total} PDB IDs for rcsb_mutation_count "
                f"(batch size {batch_size}, {n_workers} workers)")

    batches = [unique_ids[i:i + batch_size]
               for i in range(0, total, batch_size)]

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_query_mutation_batch, b): b for b in batches}
        with tqdm(total=len(batches), desc="Mutation check", unit="batch") as pbar:
            for future in concurrent.futures.as_completed(futures):
                batch_result = future.result()
                results.update(batch_result)
                pbar.update(1)

    n_mutants = sum(1 for v in results.values() if v["is_mutant"])
    n_missing = sum(1 for v in results.values() if v["status"] == "missing")
    n_errors = sum(1 for v in results.values() if v["status"].startswith("error"))
    logger.info(
        f"Mutation check complete — "
        f"mutants: {n_mutants}, wild-type: {total - n_mutants - n_missing - n_errors}, "
        f"missing from API: {n_missing}, errors: {n_errors}"
    )
    if n_errors:
        err_ids = [pid for pid, v in results.items()
                   if v["status"].startswith("error")]
        logger.warning(f"Errored IDs ({len(err_ids)}): {err_ids[:20]}"
                       + (" ..." if len(err_ids) > 20 else ""))

    return results


# Filters a DataFrame to exclude rows where any polymer entity has a
# non-zero rcsb_mutation_count (RCSB-precomputed engineered-mutation count).
# Adds three columns for audit trail:
#   mutant_status         — 0 (wild-type), 1 (mutant), or -1 (API error / missing)
#   mutation_count        — total engineered mutations summed across entities
#   mutation_check_status — 'ok' | 'missing' | 'error:...'
# To see the actual mutation string for a flagged entry, visit
#   https://www.rcsb.org/structure/{pdb_id}
# Input: DataFrame with 'pdb_id' column
# Output: DataFrame with the three new columns (a copy)
def filter_mutants(df):
    pdb_ids = df['pdb_id'].tolist()
    results = check_mutations_bulk(pdb_ids)

    mutant_status = []
    mutation_counts = []
    check_status = []
    for pid in pdb_ids:
        key = pid.upper() if isinstance(pid, str) else pid
        r = results.get(key, {"is_mutant": False, "total_count": 0,
                              "status": "missing"})
        if r["status"].startswith("error") or r["status"] == "missing":
            mutant_status.append(-1)
        else:
            mutant_status.append(1 if r["is_mutant"] else 0)
        mutation_counts.append(r.get("total_count", 0))
        check_status.append(r["status"])

    df = df.copy()
    df['mutant_status'] = mutant_status
    df['mutation_count'] = mutation_counts
    df['mutation_check_status'] = check_status
    return df

# Filters a DataFrame to rows matching a given list of PDB IDs, then saves to CSV.
# Input: list of PDB ID strings, DataFrame to filter, output file path
# Output: filtered DataFrame
def filter_csv(pdb_id_list, df, output_file_name):
    filtered_df = df[df['pdb_id'].isin(pdb_id_list)].copy()
    filtered_df.to_csv(output_file_name, index=False)
    logger.info(f"Filtered data saved to {output_file_name} ({len(filtered_df)} entries)")
    return filtered_df


if __name__ == '__main__':

    logger.info("Starting PDB data retrieval pipeline (mmCIF format)")

    input_csv = DATA_DIR / "enzymes" / "resolved_prio_ec.csv"
    df = load_csv(input_csv)

    # pdb_ids = extract_pdb_ids(df)

    # pdb_id_list_file = DATA_DIR / "enzymes" / "pdb_ids.txt"
    # cif_output_dir = DATA_DIR / "enzymes" / "enzymes_CIFs_compressed"

    # write_pdb_id_list(pdb_ids, pdb_id_list_file, DATA_DIR)
    # download_cifs_parallel(pdb_ids, cif_output_dir, n_workers=16)

    ### check for mutants (via rcsb_mutation_count)
    df_mutant = filter_mutants(df)
    df_mutant.to_csv(DATA_DIR / "enzymes" / "enzyme_data_info_mutant.csv", index=False)
    logger.info(f"Mutant-annotated CSV saved")

    ### filter to non-mutant entries only
    ### mutant_status == 0  -> wild-type, keep
    ### mutant_status == 1  -> engineered mutant, exclude
    ### mutant_status == -1 -> API error or entry missing; exclude to be safe,
    ###                        but these are logged and worth re-running
    n_err = int((df_mutant['mutant_status'] == -1).sum())
    if n_err:
        logger.warning(f"{n_err} entries had API errors / missing data and are "
                       f"excluded from the filtered set. See mutation_check_status "
                       f"column in enzyme_data_info_mutant.csv")

    df_clean = df_mutant[df_mutant['mutant_status'] == 0]
    non_mutant_ids = df_clean['pdb_id'].tolist()
    filter_csv(non_mutant_ids, df_mutant, DATA_DIR / "enzymes" / "enzymes_no_mutants.csv")

    logger.info("Pipeline complete")
