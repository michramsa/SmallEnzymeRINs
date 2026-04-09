# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

import threading
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUT_DIR = PROTEIN_DIR / "SIFTs"

# Thread lock to prevent simultaneous writes to the checkpoint file
_checkpoint_lock = threading.Lock()


def load_checkpoint(output_dir):
    """
    Load existing checkpoint if present.
    Returns a dict of {ncbi_id: result} for all already-processed proteins.
    """
    checkpoint_path = Path(output_dir) / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        print(f"Checkpoint found — {len(data)} proteins already processed, skipping.")
        return data
    return {}


def save_to_checkpoint(output_dir, ncbi_id, result):
    """
    Append a single result to the checkpoint file immediately after completion.
    Thread-safe — uses a lock to prevent concurrent write conflicts.
    """
    checkpoint_path = Path(output_dir) / "checkpoint.json"
    with _checkpoint_lock:
        if checkpoint_path.exists():
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
        else:
            data = {}

        # Serialise Path objects before saving
        serialisable = {k: str(v) if isinstance(v, Path) else v
                        for k, v in result.items()}
        data[ncbi_id] = serialisable

        with open(checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)

def get_pdb_files_flat(base_dir):
    """
    Find all PDB files in a flat directory (no subdirectory structure expected).

    Expected layout:
        <base_dir>/*.pdb

    Returns a list of dicts with keys:
        pdb_path, acr_type, structure_name, ncbi_id

    Note: acr_type defaults to "unknown" since there is no subdirectory
    to infer it from.
    """
    pdb_files = []
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Warning: {base_path} does not exist")
        return pdb_files

    for pdb_file in base_path.glob("*.pdb"):
        structure_name = pdb_file.stem
        ncbi_id = structure_name.split(".")[0]
        pdb_files.append({
            "pdb_path":       pdb_file,
            "acr_type":       "unknown",
            "structure_name": structure_name,
            "ncbi_id":        ncbi_id,
        })

    print(f"Found {len(pdb_files)} PDB file(s) in {base_path}")
    return pdb_files

def get_all_pdb_files(base_dir):
    """
    Recursively find all PDB files in the directory structure.

    Expected layout:
        <base_dir>/all_acrs/Acr_PDB/<acr_type>/<accession>.pdb

    Returns a list of dicts with keys:
        pdb_path, acr_type, structure_name, ncbi_id
    """
    pdb_files = []

    # change as needed
    acr_pdb_dir = Path(base_dir) / "all_acrs" / "Acr_PDB"

    if not acr_pdb_dir.exists():
        print(f"Warning: {acr_pdb_dir} does not exist")
        return pdb_files

    for acr_type_dir in acr_pdb_dir.iterdir():
        if acr_type_dir.is_dir():
            acr_type = acr_type_dir.name
            for pdb_file in acr_type_dir.glob("*.pdb"):
                structure_name = pdb_file.stem                 # e.g. "AAR27922.1"
                ncbi_id = structure_name.split(".")[0]          # e.g. "AAR27922"
                pdb_files.append({
                    "pdb_path":       pdb_file,
                    "acr_type":       acr_type,
                    "structure_name": structure_name,
                    "ncbi_id":        ncbi_id,
                })

    return pdb_files

def ncbi_to_uniprot(ncbi_accession):
    """
    Convert an NCBI accession to UniProt, handling both
    RefSeq (WP_, NP_, YP_ prefixes) and GenBank/EMBL accessions.
    """
    refseq_prefixes = ("WP_", "NP_", "YP_", "XP_", "AP_")
    if ncbi_accession.startswith(refseq_prefixes):
        from_db = "RefSeq_Protein"
    else:
        from_db = "EMBL-GenBank-DDBJ_CDS"

    submit_url = "https://rest.uniprot.org/idmapping/run"
    payload = {
        "from": from_db,
        "to":   "UniProtKB",
        "ids":  ncbi_accession
    }

    try:
        response = requests.post(submit_url, data=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"  Network error submitting mapping job for {ncbi_accession}: {e}")
        return None

    if response.status_code != 200:
        print(f"  Failed to submit mapping job for {ncbi_accession} "
              f"(HTTP {response.status_code})")
        return None

    job_id = response.json().get("jobId")
    if not job_id:
        print(f"  No jobId returned for {ncbi_accession}")
        return None

    # Poll for results — UniProt mapping is asynchronous
    result_url = f"https://rest.uniprot.org/idmapping/results/{job_id}"
    for attempt in range(10):
        time.sleep(2)
        try:
            result = requests.get(result_url, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"  Network error polling results for {ncbi_accession}: {e}")
            continue

        if result.status_code == 200:
            results = result.json().get("results", [])
            if results:
                # Handle both response formats — dict or plain string
                to_field = results[0].get("to")
                if isinstance(to_field, dict):
                    uniprot_id = to_field.get("primaryAccession")
                elif isinstance(to_field, str):
                    uniprot_id = to_field
                else:
                    print(f"  Unexpected 'to' format for {ncbi_accession}: {to_field}")
                    return None

                print(f"  Mapped {ncbi_accession} -> {uniprot_id}")
                return uniprot_id

        elif result.status_code == 400:
            print(f"  Mapping job failed for {ncbi_accession}")
            return None

        print(f"  Waiting for mapping result... (attempt {attempt + 1}/10)")

    print(f"  Mapping timed out for {ncbi_accession}")
    return None

def get_sifts_annotations(uniprot_id):
    """
    Retrieve GO term annotations for a UniProt accession via the PDBe/SIFTS API.
    Returns the parsed JSON response dict, or None if no record found.
    """
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/go_terms/{uniprot_id}"
    try:
        response = requests.get(url, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"  Network error querying SIFTS for {uniprot_id}: {e}")
        return None

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        print(f"  No SIFTS record found for UniProt ID: {uniprot_id}")
        return None
    else:
        print(f"  Unexpected status {response.status_code} for {uniprot_id}")
        return None

def extract_go_terms(annotations, uniprot_id):
    """
    Parse GO terms out of a SIFTS API response for a given UniProt ID.
    Returns a list of dicts with keys: go_id, go_name, category, mapped_pdb
    """
    go_terms = []
    try:
        entry   = annotations.get(uniprot_id, {})
        go_data = entry.get("GO", {})

        for go_id, details in go_data.items():
            mappings = details.get("mappings", [{}])
            go_terms.append({
                "go_id":      go_id,
                "go_name":    details.get("name", "Unknown"),
                "category":   details.get("category", "Unknown"),
                "mapped_pdb": mappings[0].get("entry_id", "N/A") if mappings else "N/A"
            })
    except (AttributeError, KeyError, IndexError) as e:
        print(f"  Warning: could not parse GO terms for {uniprot_id} — {e}")

    return go_terms

def process_single_protein(entry, output_dir):
    """
    Runs the full NCBI -> UniProt -> SIFTS pipeline for one protein.
    Saves result to checkpoint immediately on completion.
    Designed to be called in parallel.
    """
    ncbi_id        = entry["ncbi_id"]
    acr_type       = entry["acr_type"]
    structure_name = entry["structure_name"]

    result = {
        "ncbi_id":        ncbi_id,
        "uniprot_id":     "N/A",
        "acr_type":       acr_type,
        "structure_name": structure_name,
        "outcome":        None,
        "go_terms":       [],
        "notes":          ""
    }

    # --- NCBI -> UniProt ---
    uniprot_id = ncbi_to_uniprot(ncbi_id)
    result["uniprot_id"] = uniprot_id or "N/A"

    if uniprot_id is None:
        result["outcome"] = "no_uniprot"
        result["notes"]   = "NCBI accession could not be mapped to UniProt"

    else:
        # --- UniProt -> SIFTS ---
        annotations = get_sifts_annotations(uniprot_id)

        if annotations is None:
            result["outcome"] = "no_sifts"
            result["notes"]   = "UniProt ID found but no SIFTS/PDB record exists"

        else:
            # --- Parse GO terms ---
            go_terms = extract_go_terms(annotations, uniprot_id)
            result["go_terms"] = go_terms

            if go_terms:
                result["outcome"] = "confirmed"
                result["notes"]   = f"{len(go_terms)} GO term(s) found in SIFTS"
            else:
                result["outcome"] = "novel"
                result["notes"]   = "SIFTS record exists but no GO terms returned"

            # Save raw SIFTS JSON response
            raw_dir = Path(output_dir) / "sifts_results"
            raw_dir.mkdir(exist_ok=True)
            with open(raw_dir / f"{ncbi_id}.json", "w") as f:
                json.dump(annotations, f, indent=2)

    # Save to checkpoint immediately
    save_to_checkpoint(output_dir, ncbi_id, result)

    return result

def build_summary_from_results(all_results):
    """Rebuild the summary dict from a list of results."""
    summary = {"confirmed": [], "novel": [], "no_sifts": [], "no_uniprot": []}
    for result in all_results:
        outcome = result.get("outcome")
        if outcome in summary:
            summary[outcome].append(result["ncbi_id"])
    return summary


def write_csv_report(all_results, output_dir):
    """
    Write a flat CSV of all results — one row per GO term,
    or one row per protein if no GO terms were found.
    """
    csv_path = Path(output_dir) / "annotation_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ncbi_id", "uniprot_id", "acr_type", "structure_name",
            "outcome",
            "go_id", "go_name", "go_category", "mapped_pdb",
            "notes"
        ])
        for result in all_results:
            base_row = [
                result["ncbi_id"],
                result["uniprot_id"],
                result["acr_type"],
                result["structure_name"],
                result["outcome"],
            ]
            if result["go_terms"]:
                for term in result["go_terms"]:
                    writer.writerow(base_row + [
                        term["go_id"],
                        term["go_name"],
                        term["category"],
                        term["mapped_pdb"],
                        result["notes"]
                    ])
            else:
                writer.writerow(base_row + [
                    "N/A", "N/A", "N/A", "N/A",
                    result["notes"]
                ])

    print(f"\n  CSV report saved to: {csv_path}")


def write_summary_report(summary, total, output_dir):
    """Write a human-readable plain text summary report."""
    report_path = Path(output_dir) / "summary_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 55 + "\n")
        f.write("SIFTS ANNOTATION PIPELINE — SUMMARY REPORT\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Total proteins processed:  {total}\n\n")

        categories = {
            "confirmed":  "Confirmed (SIFTS GO terms found)",
            "novel":      "Novel (SIFTS record exists, no GO terms)",
            "no_sifts":   "No SIFTS record (UniProt mapped, no PDB)",
            "no_uniprot": "No UniProt mapping (NCBI accession unresolved)",
        }
        for key, label in categories.items():
            ids = summary[key]
            pct = (len(ids) / total * 100) if total > 0 else 0
            f.write(f"{label}:\n")
            f.write(f"  Count: {len(ids)} ({pct:.1f}%)\n")
            if ids:
                f.write(f"  IDs:   {', '.join(ids)}\n")
            f.write("\n")

    print(f"  Summary report saved to:  {report_path}")


def print_summary(summary, total):
    """Print a concise summary table to the console."""
    print("\n" + "=" * 55)
    print("PIPELINE COMPLETE — RESULTS SUMMARY")
    print("=" * 55)
    print(f"{'Total processed:':<40} {total}")
    print(f"{'Confirmed (GO terms found):':<40} {len(summary['confirmed'])}")
    print(f"{'Novel (SIFTS record, no GO terms):':<40} {len(summary['novel'])}")
    print(f"{'No SIFTS record:':<40} {len(summary['no_sifts'])}")
    print(f"{'No UniProt mapping:':<40} {len(summary['no_uniprot'])}")
    print("=" * 55)


def write_outputs(output_dir):
    """
    Regenerate CSV and summary report from the checkpoint file.
    Called on both clean finish and interrupted exit.
    """
    checkpoint_path = Path(output_dir) / "checkpoint.json"
    if not checkpoint_path.exists():
        print("  No checkpoint found — nothing to write.")
        return

    with open(checkpoint_path, "r") as f:
        data = json.load(f)

    all_results = list(data.values())
    summary     = build_summary_from_results(all_results)
    total       = len(all_results)

    write_csv_report(all_results, output_dir)
    write_summary_report(summary, total, output_dir)
    print_summary(summary, total)

def run_full_pipeline(base_dir, output_dir, max_workers=5):
    """
    Orchestrates the full parallel pipeline with checkpoint support:
        1. Load checkpoint — skip already-processed proteins
        2. Traverse PDB files
        3. Process remaining proteins in parallel
           (NCBI -> UniProt -> SIFTS -> GO terms)
        4. Save each result to checkpoint immediately on completion
        5. Write CSV and summary on finish or interrupt
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Register signal handler so Ctrl+C still writes outputs
    def handle_interrupt(sig, frame):
        print("\n\nInterrupted — writing outputs from checkpoint so far...")
        write_outputs(output_dir)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    # Load checkpoint — skip already-processed proteins
    checkpoint      = load_checkpoint(output_dir)
    all_pdb_files   = get_pdb_files_flat(base_dir) ### switch as needed
    remaining       = [e for e in all_pdb_files
                       if e["ncbi_id"] not in checkpoint]
    total_all       = len(all_pdb_files)
    total_remaining = len(remaining)

    print(f"\nFound {total_all} PDB files total.")
    print(f"Remaining to process: {total_remaining}\n")

    if not remaining:
        print("All proteins already processed — writing outputs from checkpoint.")
        write_outputs(output_dir)
        return

    # Run in parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_protein, entry, output_dir): entry
            for entry in remaining
        }
        for future in as_completed(futures):
            entry = futures[future]
            try:
                result = future.result()
                completed += 1
                print(f"  [{completed}/{total_remaining}] "
                      f"{result['ncbi_id']} -> {result['outcome']}")
            except Exception as e:
                print(f"  Unexpected error processing "
                      f"{entry['ncbi_id']}: {e}")

    print("\nAll proteins processed — writing final outputs...")
    write_outputs(output_dir)


if __name__ == "__main__":
    run_full_pipeline(
        base_dir    = "/home/epi2melabs/AcrNetworks-clean-v2/5_protein_analysis/AcrProts",
        output_dir  = OUTPUT_DIR,
        max_workers = 4
    )
