# Parallel Arpeggio Pipeline
# Runs clean -> CIF -> Arpeggio in parallel across multiple PDB files.

import os
import subprocess
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "3_data"

CLEAN_PDB_SCRIPT = Path("/home/epi2melabs/projects/pdbtools/clean_pdb.py")
MAX_WORKERS = 8          # safe for 14 cores / 20 logical processors
MAX_SIZE_KB = 450         # skip CIF files larger than this for Arpeggio
ARPEGGIO_TIMEOUT = 1800   # 30 min (1800 seconds) timeout per structure


def get_all_pdb_files(base_dir):
    """
    Recursively find all PDB files in the directory structure.
    
    Returns list of tuples: (pdb_file_path, acr_type, structure_name)
    """
    pdb_files = []
    acr_pdb_dir = Path(base_dir) / "all_acrs" / "Acr_PDB"
    
    if not acr_pdb_dir.exists():
        print(f"Warning: {acr_pdb_dir} does not exist")
        return pdb_files
    
    for acr_type_dir in acr_pdb_dir.iterdir():
        if acr_type_dir.is_dir():
            acr_type = acr_type_dir.name
            for pdb_file in acr_type_dir.glob("*.pdb"):
                structure_name = pdb_file.stem
                pdb_files.append((pdb_file, acr_type, structure_name))
    
    return pdb_files


def process_structure(pdb_file, acr_type, structure_name, cleaned_dir, cifs_dir, jsons_dir):
    """
    Run the full pipeline for a single structure: clean -> CIF -> Arpeggio.
    
    Returns a dict with status info for logging/summary.
    """
    clean_out_dir = cleaned_dir / acr_type
    cif_out_dir   = cifs_dir / acr_type
    json_out_dir  = jsons_dir / acr_type

    for d in [clean_out_dir, cif_out_dir, json_out_dir]:
        d.mkdir(parents=True, exist_ok=True)

    result_info = {
        'acr_type': acr_type,
        'structure': structure_name,
        'status': 'success',
        'step_failed': None,
        'error': None,
        'elapsed': 0,
        'skipped_size': False
    }

    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. Clean PDB
    # ------------------------------------------------------------------
    clean_pdb_path = clean_out_dir / f"{structure_name}_clean.pdb"

    if not clean_pdb_path.exists():
        result = subprocess.run(
            f"python3 {CLEAN_PDB_SCRIPT} -kh '{str(pdb_file)}'",
            shell=True,
            capture_output=True,
            text=True
        )
        expected_output = pdb_file.parent / f"{structure_name}.clean.pdb"

        if result.returncode == 0 and expected_output.exists():
            os.rename(expected_output, clean_pdb_path)
        else:
            result_info['status'] = 'failed'
            result_info['step_failed'] = 'CLEAN'
            result_info['error'] = result.stderr
            result_info['elapsed'] = time.time() - start_time
            return result_info

    # ------------------------------------------------------------------
    # 2. Convert to CIF
    # ------------------------------------------------------------------
    cif_path = cif_out_dir / f"{structure_name}_clean.cif"

    if not cif_path.exists():
        result = subprocess.run(
            ["gemmi", "convert", str(clean_pdb_path), str(cif_path)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            result_info['status'] = 'failed'
            result_info['step_failed'] = 'CIF'
            result_info['error'] = result.stderr
            result_info['elapsed'] = time.time() - start_time
            return result_info

    # ------------------------------------------------------------------
    # 3. Run Arpeggio (with size filter)
    # ------------------------------------------------------------------
    arpeggio_stem = cif_path.name.split(".")[0]
    json_path = json_out_dir / f"{arpeggio_stem}.json"

    if not json_path.exists():
        # Check file size before running arpeggio
        file_size_kb = os.path.getsize(cif_path) / 1024
        if file_size_kb > MAX_SIZE_KB:
            result_info['status'] = 'skipped'
            result_info['skipped_size'] = True
            result_info['error'] = f"CIF size {file_size_kb:.1f} KB exceeds limit {MAX_SIZE_KB} KB"
            result_info['elapsed'] = time.time() - start_time
            return result_info

        try:
            subprocess.run(
                ["pdbe-arpeggio", str(cif_path), "-sa", "-co", "1.0", "-o", str(json_out_dir)],
                check=True,
                timeout=ARPEGGIO_TIMEOUT,
                capture_output=True,
                text=True
            )
        except subprocess.TimeoutExpired:
            result_info['status'] = 'failed'
            result_info['step_failed'] = 'ARPEGGIO'
            result_info['error'] = f"Timeout after {ARPEGGIO_TIMEOUT}s"
            result_info['elapsed'] = time.time() - start_time
            return result_info
        except Exception as e:
            result_info['status'] = 'failed'
            result_info['step_failed'] = 'ARPEGGIO'
            result_info['error'] = str(e)
            result_info['elapsed'] = time.time() - start_time
            return result_info

    result_info['elapsed'] = time.time() - start_time
    return result_info


if __name__ == "__main__":

    ## Output directories
    cleaned_dir = DATA_DIR / "arp_results" / "PDBs_cleaned"
    cifs_dir    = DATA_DIR / "arp_results" / "CIFs"
    jsons_dir   = DATA_DIR / "arp_results" / "JSONs"
    log_dir     = DATA_DIR / "arp_results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ## Gather PDB files and sort by size (smallest first for better load balancing)
    pdb_files = get_all_pdb_files(DATA_DIR)
    pdb_files = sorted(pdb_files, key=lambda x: os.path.getsize(x[0]))
    print(f"Found {len(pdb_files)} PDB files across all Acr types")

    ## mini test - uncomment to test with a small batch
    # pdb_files = pdb_files[:5]
    # print(f"test: run on {len(pdb_files)} files: {[s for _, _, s in pdb_files]}")

    ## Run in parallel
    results = []
    print(f"Starting parallel processing with {MAX_WORKERS} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(
                process_structure,
                pdb_file, acr_type, structure_name,
                cleaned_dir, cifs_dir, jsons_dir
            ): structure_name
            for pdb_file, acr_type, structure_name in pdb_files
        }

        for future in concurrent.futures.as_completed(future_to_file):
            name = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                status_char = "OK" if result['status'] == 'success' else "FAIL" if result['status'] == 'failed' else "SKIP"
                print(f"  [{status_char}] {result['acr_type']}/{result['structure']} "
                      f"({result['elapsed']:.1f}s) {result.get('step_failed') or ''}")
            except Exception as exc:
                print(f"  [ERR] {name} raised exception: {exc}")
                results.append({
                    'acr_type': '?', 'structure': name,
                    'status': 'failed', 'step_failed': 'EXECUTOR',
                    'error': str(exc), 'elapsed': 0
                })

    ## Summary
    successes = [r for r in results if r['status'] == 'success']
    failures  = [r for r in results if r['status'] == 'failed']
    skipped   = [r for r in results if r['status'] == 'skipped']

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"  Successful:     {len(successes)}")
    print(f"  Failed:         {len(failures)}")
    print(f"  Skipped (size): {len(skipped)}")
    if successes:
        avg = sum(r['elapsed'] for r in successes) / len(successes)
        print(f"  Avg time per structure: {avg:.1f}s")
    print(f"{'='*60}")

    ## Write log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"arpeggio_run_{timestamp}.log"
    with open(log_file, "w") as f:
        f.write(f"Arpeggio pipeline run: {datetime.now()}\n")
        f.write(f"Workers: {MAX_WORKERS}\n")
        f.write(f"Total files: {len(pdb_files)}\n")
        f.write(f"Successful: {len(successes)}\n")
        f.write(f"Failed: {len(failures)}\n")
        f.write(f"Skipped (size): {len(skipped)}\n\n")

        if failures:
            f.write("FAILURES:\n")
            for r in failures:
                f.write(f"  {r['acr_type']}/{r['structure']} "
                        f"step={r['step_failed']} error={r['error']}\n")

        if skipped:
            f.write("\nSKIPPED (size limit):\n")
            for r in skipped:
                f.write(f"  {r['acr_type']}/{r['structure']} - {r['error']}\n")

    print(f"Log written to: {log_file}")
    print("Run 2_json_to_rig.py next to convert JSONs to RIG GraphML files.")
