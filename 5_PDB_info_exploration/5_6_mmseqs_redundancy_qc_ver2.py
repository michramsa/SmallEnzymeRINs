"""
mmseqs_redundancy_qc.py

Parses mmseqs2 cluster TSV files and cross-references them against a RIN
protein set (defined by a CSV file) to produce a redundancy QC report.

Expected inputs:
    - A CSV file with columns: pdb_id, chain, original_header, ...
      (chain is uppercase, e.g. "A"; pdb_id is e.g. "207L")
    - A FASTA file whose headers match the mmseqs input
      (header format: pdb_id + lowercase chain, e.g. ">207lA")
    - Three mmseqs cluster TSV files (one per identity threshold)
      TSV format: two columns, representative <tab> member (no header)

Output:
    - Console summary table comparing all three thresholds
    - A per-protein CSV listing cluster assignment and status at each threshold
    - A per-threshold CSV of cluster size distributions
"""
# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

# =============================================================================
# CONFIGURATION — edit these paths and thresholds in main()
# =============================================================================

def build_rin_id_set(csv_path: str) -> set:
    """
    Reads the RIN protein CSV and returns a set of FASTA-style IDs.
    FASTA IDs are constructed as: pdb_id (lowercase) + chain (lowercase).
    e.g., pdb_id='207L', chain='A' -> '207la'
    """
    df = pd.read_csv(csv_path)
    rin_ids = set()
    for _, row in df.iterrows():
        fasta_id = str(row["pdb_id"]).lower() + str(row["chain"]).upper()
        rin_ids.add(fasta_id)
    return rin_ids


def parse_cluster_tsv(tsv_path: str) -> dict:
    """
    Parses an mmseqs cluster TSV file.
    Returns a dict mapping each member ID -> representative ID.
    The representative maps to itself.
    """
    member_to_rep = {}
    with open(tsv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            rep, member = parts[0].strip(), parts[1].strip()
            member_to_rep[member] = rep
    return member_to_rep


def compute_cluster_stats(member_to_rep: dict, rin_ids: set) -> dict:
    """
    Given a member->rep mapping and the set of RIN protein IDs,
    computes clustering statistics restricted to the RIN set.

    Returns a dict with:
        - cluster_sizes: dict of rep -> count (within RIN set only)
        - per_protein: dict of protein_id -> {rep, is_representative, cluster_size}
        - n_clusters: int
        - n_singletons: int
        - n_redundant: int (proteins that are NOT the representative)
        - largest_cluster: int
        - proteins_not_in_tsv: set of RIN IDs absent from the TSV
    """
    # Filter member_to_rep to only RIN proteins
    rin_member_to_rep = {m: r for m, r in member_to_rep.items() if m in rin_ids}

    # Track which RIN proteins appear in the TSV at all
    proteins_not_in_tsv = rin_ids - set(rin_member_to_rep.keys())

    # Build cluster size counts (among RIN proteins only)
    cluster_sizes = defaultdict(int)
    for member, rep in rin_member_to_rep.items():
        cluster_sizes[rep] += 1

    per_protein = {}
    for member, rep in rin_member_to_rep.items():
        per_protein[member] = {
            "representative": rep,
            "is_representative": (member == rep),
            "cluster_size": cluster_sizes[rep],
        }

    n_clusters = len(cluster_sizes)
    n_singletons = sum(1 for s in cluster_sizes.values() if s == 1)
    n_redundant = sum(
        1 for p in per_protein.values() if not p["is_representative"]
    )
    largest_cluster = max(cluster_sizes.values()) if cluster_sizes else 0

    return {
        "cluster_sizes": dict(cluster_sizes),
        "per_protein": per_protein,
        "n_clusters": n_clusters,
        "n_singletons": n_singletons,
        "n_redundant": n_redundant,
        "largest_cluster": largest_cluster,
        "proteins_not_in_tsv": proteins_not_in_tsv,
    }


def print_summary_table(results: dict, rin_count: int):
    """Prints a formatted summary table comparing all thresholds."""
    print("\n" + "=" * 65)
    print("  MMSEQS REDUNDANCY QC SUMMARY")
    print(f"  Total RIN proteins: {rin_count}")
    print("=" * 65)
    header = f"{'Metric':<35} " + "  ".join(
        f"{label:>8}" for label in results.keys()
    )
    print(header)
    print("-" * 65)

    metrics = [
        ("Clusters found", "n_clusters"),
        ("Singleton clusters", "n_singletons"),
        ("Redundant proteins (non-rep)", "n_redundant"),
        ("Largest cluster size", "largest_cluster"),
        ("Proteins absent from TSV", "n_missing"),
    ]

    for label, key in metrics:
        row = f"{label:<35} " + "  ".join(
            f"{stats[key]:>8}" for stats in results.values()
        )
        print(row)

    print("=" * 65)
    print("\nNote: 'Redundant' = protein is a cluster member but not the")
    print("representative. Representatives are the suggested proteins to keep.")
    print()


def save_per_protein_csv(results: dict, rin_ids: set, output_path: str):
    """
    Saves a CSV with one row per RIN protein, showing cluster assignment
    and representative status at each threshold.
    """
    rows = []
    all_proteins = rin_ids.copy()
    for stats in results.values():
        all_proteins |= set(stats["per_protein"].keys())

    for protein in sorted(all_proteins):
        row = {"protein_id": protein}
        for threshold, stats in results.items():
            if protein in stats["per_protein"]:
                info = stats["per_protein"][protein]
                row[f"rep_{threshold}"] = info["representative"]
                row[f"is_rep_{threshold}"] = info["is_representative"]
                row[f"cluster_size_{threshold}"] = info["cluster_size"]
            else:
                row[f"rep_{threshold}"] = "NOT_IN_TSV"
                row[f"is_rep_{threshold}"] = None
                row[f"cluster_size_{threshold}"] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Per-protein CSV saved to: {output_path}")


def save_cluster_size_distribution(results: dict, output_path: str):
    """
    Saves a CSV showing how many clusters exist of each size,
    for each threshold — useful for plotting.
    """
    dist_data = []
    for threshold, stats in results.items():
        size_counts = defaultdict(int)
        for size in stats["cluster_sizes"].values():
            size_counts[size] += 1
        for size, count in sorted(size_counts.items()):
            dist_data.append({
                "threshold": threshold,
                "cluster_size": size,
                "num_clusters": count,
                "proteins_in_clusters_of_this_size": size * count,
            })

    df = pd.DataFrame(dist_data)
    df.to_csv(output_path, index=False)
    print(f"Cluster size distribution CSV saved to: {output_path}")


def main():
    # =========================================================================
    # USER INPUTS — modify these paths and settings
    # =========================================================================

    # Path to your RIN protein CSV
    rin_csv_path = ENZYMES_DIR / "info_CSVs" / "filtered_consolidated.csv"

    # Dict of threshold label -> path to mmseqs cluster TSV
    tsv_files = {
        "30%": DATA_DIR / "enzymes" / "mmseq" / "runs" / "050626_enzymes_nr30_cluster.tsv",
        "35%": DATA_DIR / "enzymes" / "mmseq" / "runs" / "050626_enzymes_nr35_cluster.tsv",
        "40%": DATA_DIR / "enzymes" / "mmseq" / "runs" / "050626_enzymes_nr40_cluster.tsv"
    }

    # Output directory for result files
    output_dir = ENZYMES_DIR / "mmseq"

    # =========================================================================
    # END OF USER INPUTS
    # =========================================================================

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading RIN protein list...")
    rin_ids = build_rin_id_set(rin_csv_path)
    print(f"  Found {len(rin_ids)} unique RIN proteins in CSV.")

    results = {}
    for label, tsv_path in tsv_files.items():
        print(f"\nParsing mmseqs TSV for threshold {label}: {tsv_path}")
        member_to_rep = parse_cluster_tsv(tsv_path)
        print(f"  Total entries in TSV: {len(member_to_rep)}")
        stats = compute_cluster_stats(member_to_rep, rin_ids)
        stats["n_missing"] = len(stats["proteins_not_in_tsv"])

        if stats["proteins_not_in_tsv"]:
            print(
                f"  WARNING: {stats['n_missing']} RIN proteins not found in TSV "
                f"(they may be singletons omitted by mmseqs or ID mismatches)."
            )
        results[label] = stats

    print_summary_table(results, len(rin_ids))

    per_protein_out = output_dir / "rin_redundancy_per_protein.csv"
    save_per_protein_csv(results, rin_ids, str(per_protein_out))

    dist_out = output_dir / "rin_cluster_size_distribution.csv"
    save_cluster_size_distribution(results, str(dist_out))

    # Print proteins absent from TSV at any threshold (potential ID issues)
    for label, stats in results.items():
        if stats["proteins_not_in_tsv"]:
            print(f"\nProteins missing from {label} TSV (first 10 shown):")
            for pid in sorted(stats["proteins_not_in_tsv"])[:10]:
                print(f"    {pid}")


if __name__ == "__main__":
    main()
