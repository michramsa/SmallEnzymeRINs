# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc=""):
        """Minimal fallback — prints progress every 500 items."""
        total = total or 0
        for i, item in enumerate(iterable):
            if i % 500 == 0:
                print(f"  {desc}: {i}/{total}", flush=True)
            yield item

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

def get_entry_info(pdb_id: str, session: requests.Session) -> dict:
    """
    Query RCSB entry endpoint for asymmetric unit chain count.
    Returns deposited_polymer_entity_instance_count — the total number
    of polymer chain instances in the asymmetric unit.
    """
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            n_asym = (
                data.get("rcsb_entry_info", {})
                    .get("deposited_polymer_entity_instance_count", None)
            )
            return {"n_chains_asymmetric_unit": n_asym}
        else:
            log.warning(f"{pdb_id}: entry endpoint returned {r.status_code}")
            return {"n_chains_asymmetric_unit": None}
    except Exception as e:
        log.warning(f"{pdb_id}: entry request failed — {e}")
        return {"n_chains_asymmetric_unit": None}


def get_assembly_info(pdb_id: str, session: requests.Session) -> dict:
    """
    Query RCSB assembly/1 endpoint for oligomeric state and chain count
    in the primary biological assembly.

    oligomeric_details comes from the depositor and may be strings like:
        "monomeric", "homodimeric", "homo-4-mer", "heterodimeric", etc.
    A 404 means no assembly record exists — common for some NMR entries
    that are inherently monomeric.
    """
    url = f"https://data.rcsb.org/rest/v1/core/assembly/{pdb_id.lower()}/1"
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            oligo_state = (
                data.get("pdbx_struct_assembly", {})
                    .get("oligomeric_details", None)
            )
            n_bio = (
                data.get("rcsb_assembly_info", {})
                    .get("polymer_entity_instance_count", None)
            )
            return {
                "oligomeric_state": oligo_state,
                "n_chains_biological_assembly": n_bio,
            }
        elif r.status_code == 404:
            log.debug(f"{pdb_id}: no assembly/1 record (404)")
            return {
                "oligomeric_state": "not_found",
                "n_chains_biological_assembly": None,
            }
        else:
            log.warning(f"{pdb_id}: assembly endpoint returned {r.status_code}")
            return {
                "oligomeric_state": None,
                "n_chains_biological_assembly": None,
            }
    except Exception as e:
        log.warning(f"{pdb_id}: assembly request failed — {e}")
        return {
            "oligomeric_state": None,
            "n_chains_biological_assembly": None,
        }


def fetch_pdb_info(pdb_id: str, session: requests.Session, delay: float) -> dict:
    """
    Fetch both entry and assembly info for one PDB ID.
    Returns a flat dict ready to merge into the main dataframe.
    """
    result = {"pdb_id": pdb_id.lower()}
    result.update(get_entry_info(pdb_id, session))
    time.sleep(delay)
    result.update(get_assembly_info(pdb_id, session))
    return result


def load_fasta_headers(fasta_path: str) -> set:
    """
    Parse a FASTA file and return the set of header IDs (e.g. '207lA', '43caB').
    Used only as a sanity-check cross-reference against the original_header column.
    """
    headers = set()
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                headers.add(line.strip().lstrip(">"))
    return headers



def run(input_csv, output_csv, fasta_path, delay, workers):

    # ── Load and validate CSV ─────────────────────────────────────────────────
    log.info(f"Loading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)

    required_cols = {"pdb_id", "chain", "original_header",
                     "n_functional_residues", "functional_residues"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        log.error(f"Input CSV is missing required columns: {missing_cols}")
        sys.exit(1)

    df["pdb_id"] = df["pdb_id"].str.lower().str.strip()
    df["chain"]  = df["chain"].str.strip()

    unique_pdbs = df["pdb_id"].unique().tolist()
    log.info(f"Found {len(df)} rows across {len(unique_pdbs)} unique PDB IDs")

    # ── Optional FASTA cross-reference ───────────────────────────────────────
    if fasta_path:
        log.info(f"Cross-referencing against FASTA: {fasta_path}")
        fasta_headers = load_fasta_headers(fasta_path)
        expected = set(df["original_header"].str.strip())
        missing_from_fasta = expected - fasta_headers
        if missing_from_fasta:
            log.warning(
                f"{len(missing_from_fasta)} original_header values not in FASTA "
                f"(first 10: {sorted(missing_from_fasta)[:10]})"
            )
        else:
            log.info("All original_header values matched in FASTA — good.")

    # ── Query PDB API in parallel ─────────────────────────────────────────────
    log.info(f"Querying PDB API ({workers} workers, {delay}s delay per PDB ID)...")
    api_results = []

    with requests.Session() as session:
        session.headers.update({"User-Agent": "oligomeric-state-fetcher/1.0"})

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fetch_pdb_info, pdb_id, session, delay): pdb_id
                for pdb_id in unique_pdbs
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="PDB API"):
                try:
                    api_results.append(future.result())
                except Exception as e:
                    pdb_id = futures[future]
                    log.error(f"{pdb_id}: unhandled exception — {e}")
                    api_results.append({
                        "pdb_id": pdb_id,
                        "n_chains_asymmetric_unit": None,
                        "oligomeric_state": None,
                        "n_chains_biological_assembly": None,
                    })

    # ── Merge API results back onto original dataframe ────────────────────────
    api_df = pd.DataFrame(api_results)[
        ["pdb_id", "oligomeric_state",
         "n_chains_asymmetric_unit", "n_chains_biological_assembly"]
    ]
    out_df = df.merge(api_df, on="pdb_id", how="left")

    # Insert new columns immediately after 'chain'
    original_cols = list(df.columns)
    new_cols = ["oligomeric_state", "n_chains_asymmetric_unit", "n_chains_biological_assembly"]
    insert_pos = original_cols.index("chain") + 1
    ordered_cols = original_cols[:insert_pos] + new_cols + original_cols[insert_pos:]
    out_df = out_df[ordered_cols]

    log.info("── Oligomeric state summary (unique PDB entries) ─────────────")
    counts = (
        out_df.drop_duplicates("pdb_id")["oligomeric_state"]
              .value_counts(dropna=False)
    )
    for state, count in counts.items():
        log.info(f"  {str(state):<40} {count:>6}")

    n_multi = (
        out_df.drop_duplicates("pdb_id")["n_chains_asymmetric_unit"]
              .apply(lambda x: x > 1 if pd.notna(x) else False)
              .sum()
    )
    log.info(f"  PDB entries with >1 chain in asymmetric unit: {n_multi}")

    n_failed = api_df["oligomeric_state"].isna().sum()
    if n_failed:
        log.warning(f"  {n_failed} PDB IDs returned no oligomeric state — check warnings above.")

    out_path = Path(output_csv)
    out_df.to_csv(out_path, index=False)
    log.info(f"Wrote {len(out_df)} rows to {out_path}")



if __name__ == "__main__":


    INPUT_CSV  = DATA_DIR / "FASTA" / "biolip_functional_residues.csv"           # path to your input CSV
    OUTPUT_CSV = DATA_DIR / "FASTA" / "oligomeric_biolip_func_residues.csv" # path for the output CSV

    FASTA_PATH = None                       # set to FASTA path for cross-ref check,
    # FASTA_PATH = "your_sequences.fasta"   # or leave as None to skip

    DELAY   = 0.05  # seconds between API calls per PDB ID (keep RCSB happy)
    WORKERS = 5     # parallel threads — keep at or below 10

    # =========================================================================

    run(
        input_csv  = INPUT_CSV,
        output_csv = OUTPUT_CSV,
        fasta_path = FASTA_PATH,
        delay      = DELAY,
        workers    = WORKERS,
    )
