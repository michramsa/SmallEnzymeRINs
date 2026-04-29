# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

def filter_fasta_by_csv(
    fasta_in: "Path",
    csv_in: "Path",
    fasta_out: "Path",
    header_col: str = "original_header",
    pdb_col: str = "pdb_id",
    chain_col: str = "chain",
) -> None:
    """
    Write a new FASTA containing only records whose header appears in the CSV.

    The set of allowed headers is built from `header_col` when that column
    exists; otherwise it falls back to concatenating `pdb_col` + `chain_col`
    (lowercased pdb, uppercased chain) to match the standard FASTA convention
    (e.g. "207l" + "A" -> "207lA").

    Parameters
    ----------
    fasta_in : Path
        Input FASTA file (the full, unfiltered set).
    csv_in : Path
        CSV whose entries define which FASTA records to keep.
    fasta_out : Path
        Where to write the filtered FASTA.
    header_col : str
        Preferred column holding the full FASTA header (e.g. "original_header").
    pdb_col, chain_col : str
        Fallback columns used if `header_col` is absent.
    """
    df = pd.read_csv(csv_in)

    if header_col in df.columns:
        headers = df[header_col].dropna().astype(str).str.strip()
    else:
        missing = {pdb_col, chain_col} - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. "
                f"Need either '{header_col}' or both '{pdb_col}' and '{chain_col}'."
            )
        pdb = df[pdb_col].astype(str).str.strip().str.lower()
        chain = df[chain_col].astype(str).str.strip().str.upper()
        headers = pdb + chain

    allowed = set(headers)
    print(f"Loaded {len(allowed):,} unique headers from: {csv_in.name}")

    fasta_out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    matched: set = set()

    with open(fasta_out, "w") as out_handle:
        for record in SeqIO.parse(fasta_in, "fasta"):
            total += 1
            if record.id in allowed:
                SeqIO.write(record, out_handle, "fasta")
                matched.add(record.id)
                kept += 1

    missing_in_fasta = allowed - matched
    pct = (100 * kept / total) if total else 0.0

    print(f"Input FASTA      : {fasta_in}")
    print(f"Records scanned  : {total:,}")
    print(f"Records kept     : {kept:,} ({pct:.1f}%)")
    print(f"Output FASTA     : {fasta_out}")

    if missing_in_fasta:
        preview = sorted(missing_in_fasta)[:5]
        print(
            f"\nWarning: {len(missing_in_fasta):,} CSV headers had no match in the FASTA."
            f"\n  First {len(preview)}: {preview}"
        )

if __name__ == "__main__":

    FASTA_DIR = DATA_DIR / "FASTA"

    # ------------------------------------------------------------------
    # Filter full FASTA down to entries present in the enzymes-no-mutants CSV
    # ------------------------------------------------------------------
    # 4/22/26
    ## Keeps only FASTA chains whose (pdb_id, chain) appears in the filtered
    ## CSV. Matches on the `original_header` column when present, otherwise
    ## reconstructs the header as pdb_id.lower() + chain.upper().

    INPUT_FASTA = DATA_DIR / "FASTA" / "biolip_filtered.fasta"
    INPUT_CSV   = DATA_DIR / "enzymes" / "enzymes_no_mutants.csv"
    OUTPUT_FASTA = DATA_DIR / "FASTA" / "enzymes_no_mutants.fasta"

    filter_fasta_by_csv(
        fasta_in=INPUT_FASTA,
        csv_in=INPUT_CSV,
        fasta_out=OUTPUT_FASTA,
    )