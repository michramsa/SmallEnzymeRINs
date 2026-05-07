# Parallel Arpeggio Pipeline (v5 – CIF-native input)
# Runs clean_cif -> Arpeggio in parallel across multiple CIF files.
#
# Changes from v4:
# - Removed all Path objects; uses os.path throughout
# - Removed Acr-type/subfolder references
# - All directory paths are editable string variables in __main__
# - No terminal flags for batch mode; edit variables directly
# - --single mode retained for validation/testing

import os
import subprocess
import time
import concurrent.futures
from datetime import datetime


# ======================================================================
# CONFIGURABLE CONSTANTS (edit these as needed)
# ======================================================================
MAX_WORKERS      = 8      # safe for 14 cores / 20 logical processors
MAX_SIZE_KB      = 450    # skip CIF files larger than this for Arpeggio
ARPEGGIO_TIMEOUT = 1800   # 30 min timeout per structure


def get_all_cif_files(input_dir):
    """
    Find all CIF files in input_dir.

    Expected structure:
        input_dir/
            1A4V.cif
            2LW5.cif
            ...

    Returns list of tuples: (cif_file_path, structure_name)
    """
    cif_files = []

    if not os.path.isdir(input_dir):
        print(f"Warning: {input_dir} does not exist")
        return cif_files

    for filename in os.listdir(input_dir):
        if filename.endswith(".cif"):
            cif_path = os.path.join(input_dir, filename)
            structure_name = os.path.splitext(filename)[0]
            cif_files.append((cif_path, structure_name))

    return cif_files


def process_structure(cif_file, structure_name, clean_cif_script,
                      clean_cif_dir, jsons_dir):
    """
    Run the full pipeline for a single structure: clean_cif -> Arpeggio.

    Returns a dict with status info for logging/summary.
    """
    os.makedirs(clean_cif_dir, exist_ok=True)
    os.makedirs(jsons_dir, exist_ok=True)

    result_info = {
        'structure':       structure_name,
        'status':          'success',
        'step_failed':     None,
        'error':           None,
        'elapsed':         0,
        'skipped_size':    False,
        'skipped_existing': False
    }

    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. Clean CIF
    # ------------------------------------------------------------------
    clean_cif_path = os.path.join(clean_cif_dir, f"{structure_name}_clean.cif")
    print(clean_cif_path)

    if not os.path.exists(clean_cif_path):
        result = subprocess.run(
            ["python3", clean_cif_script, cif_file, clean_cif_path],
            capture_output=True,
            text=True
        )
        if result.returncode != 0 or not os.path.exists(clean_cif_path):
            result_info['status']      = 'failed'
            result_info['step_failed'] = 'CLEAN_CIF'
            result_info['error']       = result.stderr
            result_info['elapsed']     = time.time() - start_time
            return result_info

    # ------------------------------------------------------------------
    # 2. Run Arpeggio (with size filter)
    # ------------------------------------------------------------------
    arpeggio_stem = os.path.splitext(os.path.basename(clean_cif_path))[0]
    json_path = os.path.join(jsons_dir, f"{arpeggio_stem}.json")

    if os.path.exists(json_path):
        result_info['skipped_existing'] = True
        result_info['elapsed'] = time.time() - start_time
        return result_info

    file_size_kb = os.path.getsize(clean_cif_path) / 1024
    if file_size_kb > MAX_SIZE_KB:
        result_info['status']       = 'skipped'
        result_info['skipped_size'] = True
        result_info['error']        = f"CIF size {file_size_kb:.1f} KB exceeds limit {MAX_SIZE_KB} KB"
        result_info['elapsed']      = time.time() - start_time
        return result_info

    try:
        subprocess.run(
            ["pdbe-arpeggio", clean_cif_path, "-sa", "-co", "1.0", "-o", jsons_dir],
            check=True,
            timeout=ARPEGGIO_TIMEOUT,
            capture_output=True,
            text=True
        )
    except subprocess.TimeoutExpired:
        result_info['status']      = 'failed'
        result_info['step_failed'] = 'ARPEGGIO'
        result_info['error']       = f"Timeout after {ARPEGGIO_TIMEOUT}s"
        result_info['elapsed']     = time.time() - start_time
        return result_info
    except Exception as e:
        result_info['status']      = 'failed'
        result_info['step_failed'] = 'ARPEGGIO'
        result_info['error']       = str(e)
        result_info['elapsed']     = time.time() - start_time
        return result_info

    result_info['elapsed'] = time.time() - start_time
    return result_info


def write_log_entry(log_file_handle, result):
    """Write a single result to the log file and flush immediately."""
    status = result['status'].upper()
    if result.get('skipped_existing'):
        status = "EXISTED"

    line = (
        f"[{status:<7}] {result['structure']}"
        f"  ({result['elapsed']:.1f}s)"
    )
    if result.get('step_failed'):
        line += f"  step={result['step_failed']}"
    if result.get('error'):
        line += f"  error={result['error']}"

    log_file_handle.write(line + "\n")
    log_file_handle.flush()


def run_single(cif_path, clean_cif_script, clean_cif_dir, jsons_dir):
    """
    Run the pipeline for a single CIF file.
    Used for validation and testing.
    """
    structure_name = os.path.splitext(os.path.basename(cif_path))[0]

    print(f"Single-file mode")
    print(f"  Input CIF:      {cif_path}")
    print(f"  Structure name: {structure_name}")
    print(f"  Clean CIF dir:  {clean_cif_dir}")
    print(f"  JSONs dir:      {jsons_dir}\n")

    result = process_structure(
        cif_path, structure_name,
        clean_cif_script, clean_cif_dir, jsons_dir
    )

    status = "EXISTED" if result.get('skipped_existing') else result['status'].upper()
    print(f"Result: [{status}]  elapsed={result['elapsed']:.1f}s")
    if result.get('step_failed'):
        print(f"  Step failed: {result['step_failed']}")
    if result.get('error'):
        print(f"  Error: {result['error']}")
    if result['status'] == 'success' and not result.get('skipped_existing'):
        print(f"  Cleaned CIF: {os.path.join(clean_cif_dir, structure_name + '_clean.cif')}")
        print(f"  JSON output: {os.path.join(jsons_dir, structure_name + '_clean.json')}")

    return result


if __name__ == "__main__":

    # ==================================================================
    # EDIT THESE VARIABLES TO CONFIGURE YOUR RUN
    # ==================================================================

    # Path to the clean_cif script
    clean_cif_script = "/home/epi2melabs/AcrNetworks-clean-v2/6_arpeggio_troubleshoot/clean_cif_ver2.py"

    # Input directory containing CIF files
    input_dir = "/home/epi2melabs/AcrNetworks-clean-v2/3_data/enzymes/sorted_enzymes_CIFs/enzymes_CIF_unflagged"

    # Output directories
    clean_cif_dir = "/home/epi2melabs/AcrNetworks-clean-v2/3_data/enzymes/sorted_enzymes_CIFs/enzymes_CIF_unflagged/CIFs_cleaned"
    jsons_dir     = "/home/epi2melabs/AcrNetworks-clean-v2/3_data/enzymes/sorted_enzymes_CIFs/enzymes_CIF_unflagged/JSONs"
    log_dir       = "/home/epi2melabs/AcrNetworks-clean-v2/3_data/enzymes/sorted_enzymes_CIFs/enzymes_CIF_unflagged/logs"

    # Single-file mode: set to a CIF file path to run one file, or None for batch
    single_file = None  # e.g. "/path/to/your/file.cif"

    # ==================================================================

    os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # SINGLE FILE MODE
    # ------------------------------------------------------------------
    if single_file:
        run_single(single_file, clean_cif_script, clean_cif_dir, jsons_dir)
        exit(0)

    # ------------------------------------------------------------------
    # BATCH MODE
    # ------------------------------------------------------------------
    cif_files = get_all_cif_files(input_dir)
    cif_files = sorted(cif_files, key=lambda x: os.path.getsize(x[0]))
    print(f"Found {len(cif_files)} CIF files")

    ## mini test - uncomment to test with a small batch
    # cif_files = cif_files[:5]
    # print(f"test: run on {len(cif_files)} files: {[s for _, s in cif_files]}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"arpeggio_run_{timestamp}.log")

    with open(log_file, "w") as log_fh:
        log_fh.write(f"Arpeggio pipeline run: {datetime.now()}\n")
        log_fh.write(f"Workers: {MAX_WORKERS}\n")
        log_fh.write(f"Total files: {len(cif_files)}\n")
        log_fh.write(f"{'='*60}\n\n")
        log_fh.flush()

        results   = []
        completed = 0
        print(f"Starting parallel processing with {MAX_WORKERS} workers...")
        print(f"Log file: {log_file}\n")

        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(
                    process_structure,
                    cif_file, structure_name,
                    clean_cif_script, clean_cif_dir, jsons_dir
                ): structure_name
                for cif_file, structure_name in cif_files
            }

            for future in concurrent.futures.as_completed(future_to_file):
                name      = future_to_file[future]
                completed += 1
                progress  = f"[{completed}/{len(cif_files)}]"

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

                    print(f"  {progress} [{tag:<7}] {result['structure']} "
                          f"({result['elapsed']:.1f}s) {result.get('step_failed') or ''}")

                    write_log_entry(log_fh, result)

                except Exception as exc:
                    print(f"  {progress} [ERR    ] {name} raised exception: {exc}")
                    err_result = {
                        'structure': name,
                        'status': 'failed', 'step_failed': 'EXECUTOR',
                        'error': str(exc), 'elapsed': 0
                    }
                    results.append(err_result)
                    write_log_entry(log_fh, err_result)

        ## Summary
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
    print("Run 2_json_to_rig.py next to convert JSONs to RIG GraphML files.")
