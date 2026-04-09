# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

# ---------------------------------------------------------------------------
# RCSB GraphQL endpoint
# ---------------------------------------------------------------------------
RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
 
 
# ---------------------------------------------------------------------------
# GraphQL query
# ---------------------------------------------------------------------------
ASSEMBLY_QUERY = """
query AssemblyInfo($pdb_id: String!) {
  entry(entry_id: $pdb_id) {
    rcsb_id
    assemblies {
      rcsb_id
      rcsb_assembly_info {
        assembly_id
        polymer_entity_instance_count
        selected_polymer_entity_instance_count
      }
      pdbx_struct_assembly {
        oligomeric_count
        oligomeric_details
      }
      rcsb_struct_symmetry {
        kind
        stoichiometry
        symbol
        type
      }
    }
  }
}
"""
 
 
# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
 
def load_checkpoint(checkpoint_path: str) -> dict:
    """
    Load previously saved results from the checkpoint file.
    Returns an empty dict if the file does not exist yet.
 
    The checkpoint file is a JSON dict mapping uppercase PDB ID -> result dict.
    Only successfully completed queries (no API failure) are saved, so any
    entry that errored due to a network issue will be retried on the next run.
    """
    path = Path(checkpoint_path)
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
        print(f"Checkpoint loaded: {len(data)} previously queried IDs found in '{checkpoint_path}'.")
        return data
    return {}
 
 
def save_checkpoint(checkpoint_path: str, classifications: dict) -> None:
    """
    Save the current classifications dict to the checkpoint file.
    Called after every successful query so progress is never lost.
    """
    with open(checkpoint_path, "w") as f:
        json.dump(classifications, f, indent=2)
 
 
# ---------------------------------------------------------------------------
# API + classification logic
# ---------------------------------------------------------------------------
 
def query_assembly(pdb_id: str, retries: int = 3, retry_delay: float = 1.0) -> dict | None:
    """
    Query the RCSB GraphQL API for assembly information.
 
    Parameters
    ----------
    pdb_id : str
        4-character PDB ID (case-insensitive).
    retries : int
        Number of retry attempts on failure.
    retry_delay : float
        Seconds to wait between retries.
 
    Returns
    -------
    dict | None
        Parsed JSON response, or None on repeated failure.
    """
    pdb_id = pdb_id.upper()
    payload = {"query": ASSEMBLY_QUERY, "variables": {"pdb_id": pdb_id}}
 
    for attempt in range(retries):
        try:
            resp = requests.post(
                RCSB_GRAPHQL_URL,
                json=payload,
                timeout=15,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            print(f"  [warn] {pdb_id} attempt {attempt + 1}/{retries} failed: {exc}")
            time.sleep(retry_delay)
 
    return None
 
 
def is_homomultimer(assembly: dict) -> bool:
    """
    Determine whether a single assembly record represents a
    homomultimeric assembly.
 
    Decision logic (applied in order — first match wins):
    1. If rcsb_struct_symmetry.kind == "Global Symmetry" AND
       stoichiometry contains only one unique entity letter (e.g. ["A","A"])
       AND oligomeric_count > 1  -> homomultimer.
    2. If oligomeric_details string contains "HOMO" -> homomultimer.
    3. If oligomeric_count > 1 AND all polymer instances are the same
       entity -> likely homomultimer.
    """
    asm_info = assembly.get("rcsb_assembly_info") or {}
    struct_asm = assembly.get("pdbx_struct_assembly") or {}
    symmetry_list = assembly.get("rcsb_struct_symmetry") or []
 
    oligomeric_count = struct_asm.get("oligomeric_count") or 0
    oligomeric_details = (struct_asm.get("oligomeric_details") or "").upper()
    n_instances = asm_info.get("polymer_entity_instance_count") or 0
    n_selected = asm_info.get("selected_polymer_entity_instance_count") or 0
 
    # Rule 1: symmetry-based check
    for sym in symmetry_list:
        if sym.get("kind") == "Global Symmetry":
            stoich = sym.get("stoichiometry") or []
            unique_entities = set(stoich)
            if len(unique_entities) == 1 and oligomeric_count > 1:
                return True
 
    # Rule 2: oligomeric_details string check
    if "HOMO" in oligomeric_details:
        return True
 
    # Rule 3: instance-count heuristic
    if oligomeric_count > 1 and n_instances > 1 and n_selected == n_instances:
        return True
 
    return False
 
 
def classify_entry(pdb_id: str) -> dict:
    """
    Query RCSB and classify all assemblies for a single PDB entry.
 
    Returns a dict with keys:
        pdb_id, has_homomultimer, homomeric_assembly_ids,
        oligomeric_details, max_oligomeric_count, error
    """
    result = {
        "pdb_id": pdb_id.upper(),
        "has_homomultimer": False,
        "homomeric_assembly_ids": "",
        "oligomeric_details": "",
        "max_oligomeric_count": 0,
        "error": None,
    }
 
    data = query_assembly(pdb_id)
    if data is None:
        result["error"] = "API request failed"
        return result
 
    entry = (data.get("data") or {}).get("entry")
    if entry is None:
        result["error"] = "Entry not found in PDB"
        return result
 
    assemblies = entry.get("assemblies") or []
    homo_ids = []
    max_count = 0
 
    for asm in assemblies:
        struct_asm = asm.get("pdbx_struct_assembly") or {}
        details = struct_asm.get("oligomeric_details") or ""
        count = struct_asm.get("oligomeric_count") or 0
 
        if count > max_count:
            max_count = count
            result["oligomeric_details"] = details
 
        if is_homomultimer(asm):
            asm_info = asm.get("rcsb_assembly_info") or {}
            homo_ids.append(str(asm_info.get("assembly_id", "?")))
 
    result["has_homomultimer"] = len(homo_ids) > 0
    result["homomeric_assembly_ids"] = ", ".join(homo_ids)
    result["max_oligomeric_count"] = max_count
    return result
 
 
# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
 
def run_pipeline(
    input_path: str,
    output_filtered: str,
    output_full: str,
    checkpoint_path: str,
    api_delay: float,
):
    """
    Full pipeline: load CSV -> query RCSB -> write filtered and full outputs.
    Resumes automatically from a checkpoint file if one exists.
 
    Parameters
    ----------
    input_path : str
        Path to your input CSV file.
    output_filtered : str
        Path for the CSV containing only homomultimeric entries.
    output_full : str
        Path for the CSV containing all entries with classification columns appended.
    checkpoint_path : str
        Path for the checkpoint JSON file used to track progress.
        Created automatically on first run; reused on subsequent runs.
    api_delay : float
        Seconds to pause between API calls. Use 0.1 for small sets,
        0.5-1.0 for larger sets to avoid rate limiting.
    """
    # Load input CSV
    df = pd.read_csv(input_path)
    if "pdb_id" not in df.columns:
        raise ValueError("Input CSV must contain a 'pdb_id' column.")
 
    unique_ids = df["pdb_id"].str.upper().unique().tolist()
    print(f"Loaded {len(df)} rows across {len(unique_ids)} unique PDB IDs.")
 
    # Load checkpoint — any IDs already queried successfully are skipped
    classifications = load_checkpoint(checkpoint_path)
 
    # Only retry IDs that either haven't been queried yet, or previously
    # failed due to an API error (not "Entry not found", which is permanent)
    remaining = [
        pdb_id for pdb_id in unique_ids
        if pdb_id not in classifications
        or classifications[pdb_id].get("error") == "API request failed"
    ]
 
    if not remaining:
        print("All IDs already queried — writing outputs from checkpoint.\n")
    else:
        print(f"{len(classifications)} already done, {len(remaining)} remaining.\n")
        print("Querying RCSB PDB API... (safe to interrupt with Ctrl+C)\n")
 
        try:
            for pdb_id in tqdm(remaining):
                result = classify_entry(pdb_id)
 
                # Save to checkpoint immediately.
                # API failures are NOT saved so they get retried next run.
                # "Entry not found" IS saved since retrying won't help.
                if result["error"] != "API request failed":
                    classifications[pdb_id] = result
                    save_checkpoint(checkpoint_path, classifications)
 
                time.sleep(api_delay)
 
        except KeyboardInterrupt:
            # Graceful exit: checkpoint is already up to date at this point
            print("\n\nInterrupted. Progress saved to checkpoint.")
            print(f"Re-run the script to continue from where you left off.")
            return
 
    # Build output dataframes from the completed classifications
    class_df = pd.DataFrame(list(classifications.values()))
    df["pdb_id_upper"] = df["pdb_id"].str.upper()
    merged = df.merge(
        class_df,
        left_on="pdb_id_upper",
        right_on="pdb_id",
        suffixes=("", "_api"),
    )
    merged = merged.drop(columns=["pdb_id_upper", "pdb_id_api"], errors="ignore")
 
    # Write outputs
    merged.to_csv(output_full, index=False)
    print(f"Full annotated table saved  -> {output_full}")
 
    homo_df = merged[merged["has_homomultimer"] == True].copy()
    homo_df.to_csv(output_filtered, index=False)
    print(f"Homomultimer entries saved  -> {output_filtered}")
 
    # Summary
    n_fail = sum(1 for v in classifications.values() if v["error"])
    print(f"\n--- Summary ---")
    print(f"  Total unique PDB IDs queried : {len(unique_ids)}")
    print(f"  With homomultimeric assembly : {homo_df['pdb_id'].nunique()}")
    print(f"  API errors / not found       : {n_fail}")
    if n_fail:
        failed = [v["pdb_id"] for v in classifications.values() if v["error"]]
        print(f"  Failed IDs: {failed}")
 
 
# ---------------------------------------------------------------------------
# Edit the variables below, then run:  python filter_homomultimers.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
 
    # Path to your input CSV file
    INPUT_CSV = DATA_DIR / "FASTA" / "biolip_functional_residues.csv"
 
    # Path for the filtered output (homomultimeric entries only)
    OUTPUT_FILTERED = DATA_DIR / "FASTA" / "homomers_filtered.csv"
 
    # Path for the full output (all entries + classification columns appended)
    OUTPUT_FULL = DATA_DIR / "FASTA" / "homomers_all.csv"
 
    # Checkpoint file — tracks which PDB IDs have already been queried.
    # Created automatically on first run. Delete it to start fresh.
    CHECKPOINT = DATA_DIR / "FASTA" / "homomers_checkpoint.json"
 
    # Seconds to pause between API calls.
    # 0.1 is fine for small datasets (<100 entries).
    # Increase to 0.5 or 1.0 for larger datasets to avoid rate limiting.
    API_DELAY = 1.0
 
    # --- Do not edit below this line ---
    run_pipeline(INPUT_CSV, OUTPUT_FILTERED, OUTPUT_FULL, CHECKPOINT, API_DELAY)
