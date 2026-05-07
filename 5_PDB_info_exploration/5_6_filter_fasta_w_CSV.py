# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

def filter_fasta_by_csv(
    csv_path,
    fasta_path,
    output_fasta_path,
    pdb_col="pdb_id",
    chain_col="chain"
):
    """
    Filters a FASTA file to only include sequences matching entries in a CSV file.

    The CSV must contain columns for PDB ID and chain. The FASTA headers are
    expected to follow the format: <pdb_id_lowercase><chain_lowercase>
    (e.g., '207la' for PDB ID '207L', chain 'A').

    Parameters
    ----------
    csv_path       : str or Path - path to the CSV file with sequences of interest
    fasta_path     : str or Path - path to the full FASTA file
    output_fasta_path : str or Path - path where the filtered FASTA will be saved
    pdb_col        : str - column name for PDB IDs in the CSV (default: 'pdb_id')
    chain_col      : str - column name for chain IDs in the CSV (default: 'chain')
    """
    # Build a set of target identifiers: pdb_id + chain, all lowercase
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df[pdb_col] = df[pdb_col].str.strip()
    df[chain_col] = df[chain_col].str.strip()
    targets = set(
        (row[pdb_col] + row[chain_col]).lower()
        for _, row in df.iterrows()
    )

    # Parse the FASTA and collect matching records
    matched_records = []
    not_found = set(targets.copy())

    for record in SeqIO.parse(fasta_path, "fasta"):
        # The header ID is the first whitespace-delimited token (e.g., '207la')
        seq_id = record.id.lower()
        if seq_id in targets:
            matched_records.append(record)
            not_found.discard(seq_id)

    # Write filtered FASTA
    output_fasta_path = Path(output_fasta_path)
    output_fasta_path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(matched_records, output_fasta_path, "fasta")

    print(f"Matched {len(matched_records)} / {len(targets)} sequences.")
    if not_found:
        print(f"Warning: {len(not_found)} target(s) not found in FASTA:")
        for nf in sorted(not_found):
            print(f"  - {nf}")

    return matched_records


if __name__ == "__main__":
    filter_fasta_by_csv(
        csv_path=ENZYMES_DIR / "info_CSVs" / "filtered_consolidated.csv",
        fasta_path=DATA_DIR / "FASTA"/ "biolip_filtered.fasta",
        output_fasta_path=DATA_DIR / "enzymes" / "mmseq" / "enzymes_FASTA.fasta",
        pdb_col="pdb_id",
        chain_col="chain"
    )