# Parallel Arpeggio Pipeline (v4 – CIF-native input)
# Runs clean_cif -> Arpeggio in parallel across multiple CIF files.
# Replaces the old clean_pdb -> gemmi convert -> Arpeggio chain.
#
# Changes from v3:
# - Input is mmCIF (not PDB); clean_cif.py replaces clean_pdb.py + gemmi convert
# - Removed: CLEAN_PDB_SCRIPT, cleaned_dir (PDBs_cleaned), cifs_dir (CIFs)
# - Added:   CLEAN_CIF_SCRIPT, clean_cif_dir (CIFs_cleaned)
# - Added:   --single flag for running on a single CIF file (validation/testing)
# - Directory structure: Acr_CIF/<acr_type>/*.cif  (mirrors old Acr_PDB layout)

import os
import argparse
import subprocess
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "3_data"

CLEAN_CIF_SCRIPT = Path(__file__).parent / "clean_cif_ver2.py"
MAX_WORKERS      = 8      # safe for 14 cores / 20 logical processors
MAX_SIZE_KB      = 450    # skip CIF files larger than this for Arpeggio
ARPEGGIO_TIMEOUT = 1800   # 30 min timeout per structure


def get_all_cif_files(base_dir):
    """
    Recursively find all CIF files in the Acr_CIF directory structure.

    Returns list of tuples: (cif_file_path, acr_type, structure_name)
    """
    cif_files = []
    acr_cif_dir = Path(base_dir) / "all_acrs" / "Acr_CIF"

    if not acr_cif_dir.exists():
        print(f"Warning: {acr_cif_dir} does not exist")
        return cif_files

    for acr_type_dir in acr_cif_dir.iterdir():
        if acr_type_dir.is_dir():
            acr_type = acr_type_dir.name
            for cif_file in acr_type_dir.glob("*.cif"):
                structure_name = cif_file.stem
                cif_files.append((cif_file, acr_type, structure_name))

    return cif_files


def process_structure(cif_file, acr_type, structure_name, clean_cif_dir, jsons_dir):
    """
    Run the full pipeline for a single structure: clean_cif -> Arpeggio.

    Returns a dict with status info for logging/summary.
    """
    clean_out_dir = clean_cif_dir
    json_out_dir  = jsons_dir

    for d in [clean_out_dir, json_out_dir]:
        d.mkdir(parents=True, exist_ok=True)

    result_info = {
        'acr_type':         acr_type,
        'structure':        structure_name,
        'status':           'success',
        'step_failed':      None,
        'error':            None,
        'elapsed':          0,
        'skipped_size':     False,
        'skipped_existing': False
    }

    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. Clean CIF
    # Replaces: clean_pdb.py + gemmi convert
    # ------------------------------------------------------------------
    clean_cif_path = clean_out_dir / f"{structure_name}_clean.cif"
    print(clean_cif_path)

    if not clean_cif_path.exists():
        result = subprocess.run(
            ["python3", str(CLEAN_CIF_SCRIPT), str(cif_file), str(clean_cif_path)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0 or not clean_cif_path.exists():
            result_info['status']      = 'failed'
            result_info['step_failed'] = 'CLEAN_CIF'
            result_info['error']       = result.stderr
            result_info['elapsed']     = time.time() - start_time
            return result_info

    # ------------------------------------------------------------------
    # 2. Run Arpeggio (with size filter)
    # ------------------------------------------------------------------
    arpeggio_stem = clean_cif_path.name.split(".")[0]
    json_path = json_out_dir / f"{arpeggio_stem}.json"

    if json_path.exists():
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
            ["pdbe-arpeggio", str(clean_cif_path), "-sa", "-co", "1.0", "-o", str(json_out_dir)],
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
        f"[{status:<7}] {result['acr_type']}/{result['structure']}"
        f"  ({result['elapsed']:.1f}s)"
    )
    if result.get('step_failed'):
        line += f"  step={result['step_failed']}"
    if result.get('error'):
        line += f"  error={result['error']}"

    log_file_handle.write(line + "\n")
    log_file_handle.flush()


def run_single(cif_path, clean_cif_dir, jsons_dir, acr_type='unknown'):
    """
    Run the pipeline for a single CIF file.
    Used for validation and testing (--single flag).
    acr_type is provided via --acr-type (defaults to 'unknown').
    """
    cif_file       = Path(cif_path)
    structure_name = cif_file.stem

    print(f"Single-file mode")
    print(f"  Input CIF:      {cif_file}")
    print(f"  ACR type:       {acr_type}")
    print(f"  Structure name: {structure_name}")
    print(f"  Clean CIF dir:  {clean_cif_dir}")
    print(f"  JSONs dir:      {jsons_dir}\n")

    result = process_structure(
        cif_file, acr_type, structure_name,
        clean_cif_dir, jsons_dir
    )

    status = "EXISTED" if result.get('skipped_existing') else result['status'].upper()
    print(f"Result: [{status}]  elapsed={result['elapsed']:.1f}s")
    if result.get('step_failed'):
        print(f"  Step failed: {result['step_failed']}")
    if result.get('error'):
        print(f"  Error: {result['error']}")
    if result['status'] == 'success' and not result.get('skipped_existing'):
        print(f"  Cleaned CIF: {clean_cif_dir / acr_type / f'{structure_name}_clean.cif'}")
        print(f"  JSON output: {jsons_dir / acr_type / f'{structure_name}_clean.json'}")

    return result


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # ARGUMENT PARSING
    # Supports both single-file (--single) and full batch mode
    # ------------------------------------------------------------------
    arg_parser = argparse.ArgumentParser(
        description="Arpeggio pipeline v4: CIF-native clean -> arpeggio"
    )
    arg_parser.add_argument(
        '--single', type=str, default=None, metavar='CIF_PATH',
        help='Run on a single CIF file (for testing/validation). '
             'Provide full path to the .cif file.'
    )
    arg_parser.add_argument(
        '--acr-type', type=str, default='unknown', metavar='ACR_TYPE',
        help='ACR type label for output subdirectory in single-file mode '
             '(e.g. AcrIF1). Defaults to "unknown".'
    )
    args = arg_parser.parse_args()

    ## Output directories (shared by both modes)
    clean_cif_dir = Path("/home/epi2melabs/AcrNetworks-clean-v2/6_arpeggio_troubleshoot/arp_results/CIFs_cleaned")
    jsons_dir     = Path("/home/epi2melabs/AcrNetworks-clean-v2/6_arpeggio_troubleshoot/arp_results/JSONs")
    log_dir       = Path("/home/epi2melabs/AcrNetworks-clean-v2/6_arpeggio_troubleshoot/arp_results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # SINGLE FILE MODE
    # ------------------------------------------------------------------
    if args.single:
        run_single(args.single, clean_cif_dir, jsons_dir, acr_type=args.acr_type)
        exit(0)

    # ------------------------------------------------------------------
    # BATCH MODE
    # ------------------------------------------------------------------
    cif_files = get_all_cif_files(DATA_DIR)
    cif_files = sorted(cif_files, key=lambda x: os.path.getsize(x[0]))
    print(f"Found {len(cif_files)} CIF files across all Acr types")

    ## mini test - uncomment to test with a small batch
    # cif_files = cif_files[:5]
    # print(f"test: run on {len(cif_files)} files: {[s for _, _, s in cif_files]}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"arpeggio_run_{timestamp}.log"

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
                    cif_file, acr_type, structure_name,
                    clean_cif_dir, jsons_dir
                ): structure_name
                for cif_file, acr_type, structure_name in cif_files
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

                    print(f"  {progress} [{tag:<7}] {result['acr_type']}/{result['structure']} "
                          f"({result['elapsed']:.1f}s) {result.get('step_failed') or ''}")

                    write_log_entry(log_fh, result)

                except Exception as exc:
                    print(f"  {progress} [ERR    ] {name} raised exception: {exc}")
                    err_result = {
                        'acr_type': '?', 'structure': name,
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
