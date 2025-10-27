# Arpeggio can take awhile to run, and could benefit from parallelization.
# Approximate time estimate - for smaller structures (less than 1k KB), approximate runtime could be 5 minutes a structure
# With about 4k structures -> 13 days of sequential run time
# Large structures can be above 10k KB in size

# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *
import numpy as np

# Function to process a single structure with file size filter
def process_structure(cif_file, out_dir, max_timeout=1800, max_size_kb=150):
    base_name = cif_file.stem
    file_size_kb = os.path.getsize(cif_file) / 1024  # Convert bytes to KB
    
    # Skip files larger than the size limit
    if file_size_kb > max_size_kb:
        print(f"[{datetime.now()}] Skipping {base_name} ({file_size_kb:.2f} KB) - exceeds size limit of {max_size_kb} KB")
        return (base_name, False, 0, "size_limit")
    
    print(f"[{datetime.now()}] Processing {base_name} ({file_size_kb:.2f} KB) with {max_timeout}s timeout")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ["pdbe-arpeggio", str(cif_file), "-sa", "-co", "1.0", "-o", out_dir],
            check=False,
            timeout=max_timeout,
            capture_output=True,
            text=True
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[{datetime.now()}] Success: {base_name} in {elapsed:.2f}s")
            print(f"FINISHED")
            return (base_name, True, elapsed, "success")
        else:
            print(f"[{datetime.now()}] Failed: {base_name} with code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return (base_name, False, elapsed, "error")
    
    except subprocess.TimeoutExpired:
        print(f"[{datetime.now()}] Timeout: {base_name} after {max_timeout}s")
        return (base_name, False, max_timeout, "timeout")
    except Exception as e:
        print(f"[{datetime.now()}] Error: {base_name} - {str(e)}")
        return (base_name, False, time.time() - start_time, "exception")

# Main processing function with parallel execution
def process_all_structures(cifs_dir, out_dir, max_workers=4, max_size_kb=150):
    os.makedirs(out_dir, exist_ok=True)
    
    # Get all CIF files and sort by size
    cif_files = list(Path(cifs_dir).glob("*.cif"))
    cif_files = sorted(cif_files, key=lambda p: os.path.getsize(p))
    
    print(f"Found {len(cif_files)} structures to process")
    
    # Check which files are already processed
    processed_files = set(f.stem for f in Path(out_dir).glob("*.json"))
    to_process = [f for f in cif_files if f.stem not in processed_files]

    # Filter by file size
    filtered_files = []
    oversized_files = []
    for f in to_process:
        if os.path.getsize(f) / 1024 <= max_size_kb:
            filtered_files.append(f)
        else:
            oversized_files.append(f)
    
    print(f"Already processed: {len(processed_files)}")
    print(f"Remaining to process: {len(to_process)}")
    print(f"Files within size limit ({max_size_kb} KB): {len(filtered_files)}")
    print(f"Files exceeding size limit: {len(oversized_files)}")
    # Save a list of skipped files for reference
    with open(os.path.join(out_dir, "oversized_files.txt"), "w") as f:
        for oversized in oversized_files:
            size_kb = os.path.getsize(oversized) / 1024
            f.write(f"{oversized.stem}: {size_kb:.2f} KB\n")
    
    # Process in parallel
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_structure, f, out_dir, 1800, max_size_kb): f for f in filtered_files
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                cif_file = future_to_file[future]
                print(f"Processing generated an exception: {cif_file.stem} - {exc}")
                results.append((cif_file.stem, False, 0, "executor_exception"))
    
    # Summarize results
    successes = [r for r in results if r[1]]
    failures = [r for r in results if not r[1] and r[3] != "size_limit"]
    skipped = [r for r in results if r[3] == "size_limit"]

    # Calculate statistics
    if successes:
        avg_time = np.mean([r[2] for r in successes])
        print(f"\nSuccessfully processed {len(successes)} structures (avg time: {avg_time:.2f}s)")
    
    if failures:
        print(f"\nFailed to process {len(failures)} structures:")
        for name, _, _, reason in failures:
            print(f"  - {name} (reason: {reason})")
    
    print(f"\nSkipped {len(skipped) + len(oversized_files)} structures due to size limit")
    
    # Write out results to a log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(out_dir, f"processing_log_{timestamp}.txt"), "w") as log:
        log.write(f"Processing run at {datetime.now()}\n")
        log.write(f"Total files found: {len(cif_files)}\n")
        log.write(f"Already processed: {len(processed_files)}\n")
        log.write(f"Files processed this run: {len(results)}\n")
        log.write(f"Successes: {len(successes)}\n")
        log.write(f"Failures: {len(failures)}\n")
        log.write(f"Skipped due to size: {len(skipped) + len(oversized_files)}\n\n")
        
        log.write("Failed files:\n")
        for name, _, _, reason in failures:
            log.write(f"  - {name} (reason: {reason})\n")
    
    return results

if __name__ == "__main__":
    cifs_dir = PISCES_DIR / "CIFs"  
    out_dir = PISCES_DIR / "JSONs" ## Deprecated - is Cleaned_JSONs
    
    results = process_all_structures(cifs_dir, out_dir, max_workers=4, max_size_kb=350)