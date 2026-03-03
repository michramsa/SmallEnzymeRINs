# need to see the sizes of the files - just to help better inform the MAX_SIZE_KB variable in 1_1_parallel_arpeggio_ver3.py

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "3_data"


def summarize_file_sizes(base_dir, extension=".pdb"):
    """
    Summarize file sizes for all files with the given extension 
    under the Acr_PDB directory.
    
    Parameters:
    - base_dir: Base directory containing all_acrs/Acr_PDB
    - extension: File extension to look for (".pdb" or ".cif")
    """
    search_dir = Path(base_dir) / "all_acrs" / "Acr_PDB"
    
    # If looking at CIFs, check the arp_results directory instead
    if extension == ".cif":
        search_dir = Path(base_dir) / "arp_results" / "CIFs"

    if not search_dir.exists():
        print(f"Directory not found: {search_dir}")
        return

    sizes_kb = []
    per_acr = {}

    for f in search_dir.rglob(f"*{extension}"):
        size_kb = os.path.getsize(f) / 1024
        sizes_kb.append((f.name, size_kb))

        # Track per Acr type (parent folder name)
        acr_type = f.parent.name
        if acr_type not in per_acr:
            per_acr[acr_type] = []
        per_acr[acr_type].append(size_kb)

    if not sizes_kb:
        print(f"No {extension} files found in {search_dir}")
        return

    # Sort by size
    sizes_kb.sort(key=lambda x: x[1])
    all_sizes = [s for _, s in sizes_kb]

    # Overall summary
    print(f"{'='*60}")
    print(f"FILE SIZE SUMMARY: {extension} files")
    print(f"Directory: {search_dir}")
    print(f"{'='*60}")
    print(f"  Total files:  {len(all_sizes)}")
    print(f"  Min:          {min(all_sizes):.1f} KB")
    print(f"  Max:          {max(all_sizes):.1f} KB")
    print(f"  Mean:         {sum(all_sizes)/len(all_sizes):.1f} KB")
    median_idx = len(all_sizes) // 2
    print(f"  Median:       {all_sizes[median_idx]:.1f} KB")

    # Distribution by size buckets
    buckets = [50, 100, 150, 200, 350, 500, 1000, 5000, 10000]
    print(f"\n  Size distribution:")
    prev = 0
    for threshold in buckets:
        count = sum(1 for s in all_sizes if prev < s <= threshold)
        cumulative = sum(1 for s in all_sizes if s <= threshold)
        pct = cumulative / len(all_sizes) * 100
        print(f"    {prev:>6} - {threshold:>6} KB:  {count:>5} files  "
              f"(cumulative: {cumulative:>5} = {pct:.1f}%)")
        prev = threshold
    remaining = sum(1 for s in all_sizes if s > buckets[-1])
    if remaining:
        print(f"    > {buckets[-1]:>5} KB:          {remaining:>5} files")

    # Per Acr type summary
    print(f"\n  Per Acr type:")
    for acr_type in sorted(per_acr.keys()):
        acr_sizes = per_acr[acr_type]
        print(f"    {acr_type:<15} {len(acr_sizes):>4} files  "
              f"min={min(acr_sizes):.0f}  max={max(acr_sizes):.0f}  "
              f"mean={sum(acr_sizes)/len(acr_sizes):.0f} KB")

    # Show the largest files
    print(f"\n  Top 10 largest files:")
    for name, size in sizes_kb[-10:]:
        print(f"    {name:<40} {size:.1f} KB")


if __name__ == "__main__":
    # print("Checking PDB files...\n")
    # summarize_file_sizes(DATA_DIR, extension=".pdb")

    # to check cif files too
    print("\n\nChecking CIF files...\n")
    summarize_file_sizes(DATA_DIR, extension=".cif")
