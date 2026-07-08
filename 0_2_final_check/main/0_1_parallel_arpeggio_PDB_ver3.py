# Parallel Arpeggio Pipeline
# Runs clean -> CIF -> Arpeggio in parallel across multiple PDB files.
### Note - also updated -co (compensation factor for VdW radii dependent interaction types) argument, back to default

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from imports import *

CLEAN_PDB_SCRIPT = Path("/home/epi2melabs/projects/pdbtools/clean_pdb.py")
MAX_WORKERS = 8          # safe for 14 cores / 20 logical processors
MAX_SIZE_KB = 450         # skip CIF files larger than this for Arpeggio
ARPEGGIO_TIMEOUT = 1800   # 30 min (1800 seconds) timeout per structure

# Resolve full paths to external binaries at import time so that worker
# processes spawned by ProcessPoolExecutor inherit the resolved path rather
# than relying on a PATH that may be stripped in the subprocess environment.
GEMMI_BIN    = shutil.which("gemmi")
ARPEGGIO_BIN = shutil.which("pdbe-arpeggio")

if GEMMI_BIN is None:
    raise EnvironmentError(
        "Could not find 'gemmi' on PATH. "
        "Activate the correct conda/venv environment before running."
    )
if ARPEGGIO_BIN is None:
    raise EnvironmentError(
        "Could not find 'pdbe-arpeggio' on PATH. "
        "Activate the correct conda/venv environment before running."
    )


def get_all_pdb_files(pdb_dir, group_label="structures"):
    """
    Find all PDB files directly within pdb_dir (flat layout).

    All files are assigned the same group_label, which is used as the
    output subdirectory name under cleaned_dir / cifs_dir / jsons_dir.

    Returns list of tuples: (pdb_file_path, group_label, structure_name)
    """
    pdb_files = []

    if not pdb_dir.exists():
        print(f"Warning: {pdb_dir} does not exist")
        return pdb_files

    for pdb_file in sorted(pdb_dir.glob("*.pdb")):
        structure_name = pdb_file.stem
        pdb_files.append((pdb_file, group_label, structure_name))

    return pdb_files


def process_structure(pdb_file, group, structure_name, cleaned_dir, cifs_dir, jsons_dir):
    """
    Run the full pipeline for a single structure: clean -> CIF -> Arpeggio.

    Returns a dict with status info for logging/summary.
    """
    clean_out_dir = cleaned_dir / group
    cif_out_dir   = cifs_dir / group
    json_out_dir  = jsons_dir / group

    for d in [clean_out_dir, cif_out_dir, json_out_dir]:
        d.mkdir(parents=True, exist_ok=True)

    result_info = {
        'group': group,
        'structure': structure_name,
        'status': 'success',
        'step_failed': None,
        'error': None,
        'elapsed': 0,
        'skipped_size': False,
        'skipped_existing': False
    }

    start_time = time.time()

    # 1. Clean PDB
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

    # 2. Convert to CIF
    cif_path = cif_out_dir / f"{structure_name}_clean.cif"

    if not cif_path.exists():
        result = subprocess.run(
            [GEMMI_BIN, "convert", str(clean_pdb_path), str(cif_path)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            result_info['status'] = 'failed'
            result_info['step_failed'] = 'CIF'
            result_info['error'] = result.stderr
            result_info['elapsed'] = time.time() - start_time
            return result_info

    # 3. Run Arpeggio (with size filter)
    arpeggio_stem = cif_path.name.split(".")[0]
    print(arpeggio_stem)
    json_path = json_out_dir / f"{arpeggio_stem}.json"

    if json_path.exists():
        result_info['skipped_existing'] = True
        result_info['elapsed'] = time.time() - start_time
        return result_info

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
            [ARPEGGIO_BIN, str(cif_path), "-sa", "-o", str(json_out_dir)],
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


def write_log_entry(log_file_handle, result):
    """Write a single result to the log file and flush immediately."""
    status = result['status'].upper()
    if result.get('skipped_existing'):
        status = "EXISTED"

    line = (
        f"[{status:<7}] {result['group']}/{result['structure']}"
        f"  ({result['elapsed']:.1f}s)"
    )
    if result.get('step_failed'):
        line += f"  step={result['step_failed']}"
    if result.get('error'):
        line += f"  error={result['error']}"

    log_file_handle.write(line + "\n")
    log_file_handle.flush()


if __name__ == "__main__":

    ## Configurable parameters
    group_label = "structures"   # name of the output subdirectory under cleaned/CIFs/JSONs

    ## Output directories
    source_dir  = DATA_DIR_3 / "ref" / "ref_set_raw"
    cleaned_dir = DATA_DIR_3 / "ref" / "PDBs_cleaned"
    cifs_dir    = DATA_DIR_3 / "ref" / "CIFs_cleaned"
    jsons_dir   = DATA_DIR_3 / "ref" / "JSONs"
    log_dir     = DATA_DIR_3 / "ref" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ## Diagnostic: confirm source_dir resolves and contains PDB files
    print(f"Source dir : {source_dir}")
    print(f"Exists     : {source_dir.exists()}")
    print(f"PDB count  : {len(list(source_dir.glob('*.pdb')))}")

    ## Gather PDB files and sort by size (smallest first for better load balancing)
    pdb_files = get_all_pdb_files(source_dir, group_label=group_label)


    pdb_files = sorted(pdb_files, key=lambda x: os.path.getsize(x[0]))
    print(f"Found {len(pdb_files)} PDB files in group '{group_label}'")

    ## mini test - uncomment to test with a small batch
    # pdb_files = pdb_files[:5]
    # print(f"test: run on {len(pdb_files)} files: {[s for _, _, s in pdb_files]}")

    ## Open log file BEFORE the executor so every result is recorded incrementally
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"arpeggio_run_{timestamp}.log"

    with open(log_file, "w") as log_fh:
        # Write header
        log_fh.write(f"Arpeggio pipeline run: {datetime.now()}\n")
        log_fh.write(f"Workers: {MAX_WORKERS}\n")
        log_fh.write(f"Total files: {len(pdb_files)}\n")
        log_fh.write(f"{'='*60}\n\n")
        log_fh.flush()

        ## Run in parallel
        results = []
        completed = 0
        print(f"Starting parallel processing with {MAX_WORKERS} workers...")
        print(f"Log file: {log_file}\n")

        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(
                    process_structure,
                    pdb_file, group, structure_name,
                    cleaned_dir, cifs_dir, jsons_dir
                ): structure_name
                for pdb_file, group, structure_name in pdb_files
            }

            for future in concurrent.futures.as_completed(future_to_file):
                name = future_to_file[future]
                completed += 1
                progress = f"[{completed}/{len(pdb_files)}]"

                try:
                    result = future.result()
                    results.append(result)

                    if result.get('skipped_existing'):
                        tag = "EXISTED"
                    elif result['status'] == 'success':
                        tag = "OK"
                    elif result['status'] == 'failed':
                        tag = "FAIL"
                    else:
                        tag = "SKIP"

                    print(f"  {progress} [{tag:<7}] {result['group']}/{result['structure']} "
                          f"({result['elapsed']:.1f}s) {result.get('step_failed') or ''}")

                    # Write to log immediately
                    write_log_entry(log_fh, result)

                except Exception as exc:
                    print(f"  {progress} [ERR    ] {name} raised exception: {exc}")
                    err_result = {
                        'group': '?', 'structure': name,
                        'status': 'failed', 'step_failed': 'EXECUTOR',
                        'error': str(exc), 'elapsed': 0
                    }
                    results.append(err_result)
                    write_log_entry(log_fh, err_result)

        ## Summary (written to both console and log)
        successes = [r for r in results if r['status'] == 'success']
        failures  = [r for r in results if r['status'] == 'failed']
        skipped   = [r for r in results if r['status'] == 'skipped']
        existed   = [r for r in results if r.get('skipped_existing')]

        summary_lines = [
            f"\n{'='*60}",
            "PIPELINE COMPLETE",
            f"  Successful:         {len(successes)}",
            f"  Already existed:    {len(existed)}",
            f"  Failed:             {len(failures)}",
            f"  Skipped (size):     {len(skipped)}",
        ]
        if successes:
            avg = sum(r['elapsed'] for r in successes) / len(successes)
            summary_lines.append(f"  Avg time (new):     {avg:.1f}s")
        summary_lines.append(f"{'='*60}")

        for line in summary_lines:
            print(line)
            log_fh.write(line + "\n")
        log_fh.flush()

    print(f"\nLog written to: {log_file}")