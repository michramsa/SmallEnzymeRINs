# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

def parse_fasta(filepath):
    """Parse a FASTA file into a list of (header, sequence) tuples."""
    entries = []
    header = None
    seq_lines = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seq_lines)))
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)

        # Don't forget the last entry
        if header is not None:
            entries.append((header, "".join(seq_lines)))

    return entries


def filter_by_length(entries, max_length):
    """Return only entries whose sequence length is <= max_length."""
    return [(h, s) for h, s in entries if len(s) <= max_length]


def write_fasta(entries, filepath):
    """Write a list of (header, sequence) tuples to a FASTA file."""
    with open(filepath, "w") as f:
        for header, sequence in entries:
            f.write(f"{header}\n{sequence}\n")


def main(input_path, output_path, max_length):
    print(f"Reading sequences from: {input_path}")
    entries = parse_fasta(input_path)
    print(f"  Total sequences found: {len(entries)}")

    filtered = filter_by_length(entries, max_length)
    print(f"  Sequences <= {max_length} aa:  {len(filtered)}")
    print(f"  Sequences removed:      {len(entries) - len(filtered)}")

    write_fasta(filtered, output_path)
    print(f"Filtered FASTA written to: {output_path}")


if __name__ == "__main__":
    INPUT_PATH  = DATA_DIR / "FASTA"/ "protein_nr.fasta"   # Path to your input FASTA file
    OUTPUT_PATH = DATA_DIR / "FASTA"/ "biolip_filtered.fasta"    # Path for the filtered output FASTA file
    MAX_LENGTH  = 200                        

    main(INPUT_PATH, OUTPUT_PATH, MAX_LENGTH)

