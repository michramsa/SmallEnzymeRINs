# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

def parse_fasta_headers(fasta_path):
    keys = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                raw = line[1:].strip().split()[0]  # e.g. "207lA"
                pdb_id = raw[:4].lower()
                chain  = raw[4:]
                keys.append((pdb_id, chain, raw))
    return keys


def parse_residues(field):
    """
    Extract unique residue identifiers from a BioLiP residue field.
    Handles space-separated residues and semicolon-separated catalytic sites.
    e.g. "M236 E219;E219" -> {"M236", "E219"}
    Empty or missing fields ("-") return an empty set.
    """
    if not field or field.strip() in ("", "-"):
        return set()
    # Replace semicolons with spaces, then split on whitespace
    tokens = field.replace(";", " ").split()
    return set(tokens)


def parse_biolip_annotations(annotation_path, target_keys):
    """
    For each target (pdb_id, chain), aggregate all functional residues
    across every ligand interaction row — pooling binding and catalytic
    sites, deduplicating across rows and fields.

    Returns a dict: {(pdb_id, chain): set of residue identifiers}
    """
    target_set = {(pdb_id, chain) for pdb_id, chain, _ in target_keys}
    results    = defaultdict(set)

    with open(annotation_path) as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 11:
                continue

            key = (cols[0].lower(), cols[1])
            if key not in target_set:
                continue

            # col 08: binding site residues (start from 1)
            results[key].update(parse_residues(cols[8]))
            # col 10: catalytic site residues (start from 1)
            results[key].update(parse_residues(cols[10]))

    return results


def write_results(results, target_keys, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow([
            "pdb_id", "chain", "original_header",
            "n_functional_residues", "functional_residues"
        ])
        for pdb_id, chain, raw in target_keys:
            key = (pdb_id, chain)
            residues = results.get(key)
            if not residues:
                writer.writerow([pdb_id, chain, raw, 0, "NOT_FOUND"])
            else:
                # Sort for readability
                sorted_residues = " ".join(sorted(residues))
                writer.writerow([
                    pdb_id, chain, raw,
                    len(residues), sorted_residues
                ])

    print(f"Results written to {output_path}")


if __name__ == "__main__":
    fasta_path      = DATA_DIR / "FASTA" / "biolip_filtered.fasta"
    annotation_path = DATA_DIR / "FASTA" / "BioLiP.txt"
    output_path     = DATA_DIR / "FASTA" / "biolip_functional_residues.csv"

    target_keys = parse_fasta_headers(fasta_path)
    print(f"Loaded {len(target_keys)} sequences from FASTA")

    results = parse_biolip_annotations(annotation_path, target_keys)
    print(f"Found annotations for {len(results)} unique chains")
    print(f"No annotations found for {len(target_keys) - len(results)} chains")

    write_results(results, target_keys, output_path)