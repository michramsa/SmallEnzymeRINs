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

    Uses PDB author residue numbering (columns 08 and 10, 1-indexed),
    which corresponds to auth_seq_id in mmCIF files and matches the
    residue numbers stored in the RIN centrality CSVs.

    BioLiP column reference (1-indexed, tab-separated):
        col 08 (index 7):  binding site residues  — PDB author numbering  <-- used here
        col 09 (index 8):  binding site residues  — renumbered from 1
        col 10 (index 9):  catalytic site residues — PDB author numbering <-- used here
        col 11 (index 10): catalytic site residues — renumbered from 1

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

            # col 08 (index 7): binding site residues with PDB author numbering
            results[key].update(parse_residues(cols[7]))
            # col 10 (index 9): catalytic site residues with PDB author numbering
            results[key].update(parse_residues(cols[9]))

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

def verify_csv_matches_fasta(fasta_path, csv_path):
    """
    Verify that every row in the CSV corresponds to a (pdb_id, chain) pair
    derived from the FASTA headers, and that no extra or missing entries exist.

    Checks performed:
        1. Every CSV row has a (pdb_id, chain) present in the FASTA.
        2. Every FASTA entry has a corresponding row in the CSV.
        3. The original_header column in the CSV matches the raw FASTA header token.
        4. No duplicate (pdb_id, chain) pairs exist in the CSV.

    Parameters
    ----------
    fasta_path : str or Path
        Path to the FASTA file used to generate the CSV.
    csv_path   : str or Path
        Path to the CSV output from write_results().

    Returns
    -------
    bool
        True if all checks pass, False if any discrepancy is found.
        All discrepancies are printed to stdout.
    """
    # --- load FASTA keys ---
    fasta_keys = parse_fasta_headers(fasta_path)                        # [(pdb_id, chain, raw), ...]
    fasta_pairs  = {(pdb_id, chain) for pdb_id, chain, _   in fasta_keys}
    fasta_raws   = {raw              for _,      _,     raw in fasta_keys}

    # --- load CSV ---
    df = pd.read_csv(csv_path)
    csv_pairs = list(zip(df["pdb_id"].str.lower(), df["chain"].astype(str)))

    all_passed = True

    # Check 1: every CSV row maps back to a FASTA entry
    csv_not_in_fasta = [(p, c) for p, c in csv_pairs if (p, c) not in fasta_pairs]
    if csv_not_in_fasta:
        all_passed = False
        print(f"[FAIL] Check 1 — {len(csv_not_in_fasta)} CSV row(s) have no matching FASTA entry:")
        for p, c in csv_not_in_fasta:
            print(f"         pdb_id={p!r}  chain={c!r}")
    else:
        print("[PASS] Check 1 — all CSV rows have a matching FASTA entry.")

    # Check 2: every FASTA entry has a row in the CSV
    fasta_not_in_csv = [(p, c) for p, c, _ in fasta_keys if (p, c) not in set(csv_pairs)]
    if fasta_not_in_csv:
        all_passed = False
        print(f"[FAIL] Check 2 — {len(fasta_not_in_csv)} FASTA entry/entries missing from CSV:")
        for p, c in fasta_not_in_csv:
            print(f"         pdb_id={p!r}  chain={c!r}")
    else:
        print("[PASS] Check 2 — all FASTA entries have a corresponding CSV row.")

    # Check 3: original_header column matches the raw FASTA token
    header_mismatches = df[~df["original_header"].isin(fasta_raws)]
    if not header_mismatches.empty:
        all_passed = False
        print(f"[FAIL] Check 3 — {len(header_mismatches)} CSV row(s) have an original_header not found in FASTA:")
        print(header_mismatches[["pdb_id", "chain", "original_header"]].to_string(index=False))
    else:
        print("[PASS] Check 3 — all original_header values match FASTA header tokens.")

    # Check 4: no duplicate (pdb_id, chain) pairs in the CSV
    duplicates = df[df.duplicated(subset=["pdb_id", "chain"], keep=False)]
    if not duplicates.empty:
        all_passed = False
        print(f"[FAIL] Check 4 — {len(duplicates)} duplicate (pdb_id, chain) row(s) found in CSV:")
        print(duplicates[["pdb_id", "chain", "original_header"]].to_string(index=False))
    else:
        print("[PASS] Check 4 — no duplicate (pdb_id, chain) pairs in CSV.")

    print(f"\n{'All checks passed.' if all_passed else 'One or more checks failed — review output above.'}")
    return all_passed


if __name__ == "__main__":

    ## 5/20/26 confirmed this was the same result as the original run 
    fasta_path      = DATA_DIR_2 / "biolip_filtered.fasta"
    annotation_path = DATA_DIR_2 / "BioLiP.txt"  ## you get this directly from the Biolip website
    output_path     = DATA_DIR_2 / "biolip_functional_residues.csv"

    target_keys = parse_fasta_headers(fasta_path)
    print(f"Loaded {len(target_keys)} sequences from FASTA")

    results = parse_biolip_annotations(annotation_path, target_keys)
    print(f"Found annotations for {len(results)} unique chains")
    print(f"No annotations found for {len(target_keys) - len(results)} chains")

    write_results(results, target_keys, output_path)

    # 5/20/26 - check
    fasta_path      = DATA_DIR_2 / "biolip_filtered.fasta"
    annotation_path = DATA_DIR_2 / "BioLiP.txt"
    output_path     = DATA_DIR_2 / "biolip_functional_residues.csv"

    target_keys = parse_fasta_headers(fasta_path)
    print(f"Loaded {len(target_keys)} sequences from FASTA")

    results = parse_biolip_annotations(annotation_path, target_keys)
    print(f"Found annotations for {len(results)} unique chains")
    print(f"No annotations found for {len(target_keys) - len(results)} chains")

    write_results(results, target_keys, output_path)

    print("\n--- Verifying CSV against FASTA ---")
    verify_csv_matches_fasta(fasta_path, output_path)


