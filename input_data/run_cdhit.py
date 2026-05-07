# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

def run_mmseqs2(
    input_fasta: str,
    output_prefix: str,
    identity: float = 0.3,
    coverage: float = 0.8,
    cov_mode: int = 0,
    threads: int = 4,
    tmp_dir: str = "tmp_mmseqs",
) -> dict:
    """
    Run MMseqs2 easy-cluster to cluster sequences at low identity.

    MMseqs2 is preferred over CD-HIT for identity thresholds below ~0.4,
    where CD-HIT's k-mer heuristics become unreliable.

    Parameters
    ----------
    input_fasta : str
        Path to input FASTA file.
    output_prefix : str
        Prefix for output files. MMseqs2 writes several files with this
        prefix, including:
            <prefix>_rep_seq.fasta   - representative sequences
            <prefix>_all_seqs.fasta  - all sequences grouped by cluster
            <prefix>_cluster.tsv     - representative -> member mapping
    identity : float
        Minimum sequence identity threshold in [0, 1].
    coverage : float
        Minimum alignment coverage in [0, 1]. With cov_mode=0, this is
        coverage of the shorter sequence.
    cov_mode : int
        Coverage mode:
            0 = coverage of query AND target (bidirectional; strict)
            1 = coverage of target only
            2 = coverage of query only
        Mode 0 with coverage=0.8 is the field standard for redundancy
        reduction: alignment must cover 80% of the shorter sequence.
    threads : int
        Number of CPU threads.
    tmp_dir : str
        Temporary directory for MMseqs2 intermediate files.

    Returns
    -------
    dict
        Run metadata analogous to run_cdhit output.
    """
    rep_seq_path = f"{output_prefix}_rep_seq.fasta"

    cmd = [
        "mmseqs", "easy-cluster",
        str(input_fasta),
        str(output_prefix),
        str(tmp_dir),
        "--min-seq-id", str(identity),
        "-c", str(coverage),
        "--cov-mode", str(cov_mode),
        "--threads", str(threads),
    ]

    print(f"Running: {' '.join(cmd)}")
    start = datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    elapsed = (datetime.now() - start).total_seconds()

    input_count = count_sequences(input_fasta)
    output_count = count_sequences(rep_seq_path)

    return {
        "timestamp": start.isoformat(),
        "elapsed_seconds": elapsed,
        "tool": "mmseqs2",
        "command": " ".join(cmd),
        "parameters": {
            "identity": identity,
            "coverage": coverage,
            "cov_mode": cov_mode,
            "threads": threads,
        },
        "input_file": str(input_fasta),
        "output_file": rep_seq_path,
        "input_sequence_count": input_count,
        "output_sequence_count": output_count,
        "reduction_percent": 100 * (1 - output_count / input_count),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def run_cdhit(
    input_fasta: str,
    output_fasta: str,
    identity: float = 0.9,
    word_size: int = 5,
    memory_mb: int = 8000,
    threads: int = 4,
    keep_description: int = 0,
) -> dict:
    """
    Run CD-HIT to cluster protein sequences by identity.

    Parameters
    ----------
    input_fasta : str
        Path to input FASTA file.
    output_fasta : str
        Path for CD-HIT output (representatives). A .clstr file listing
        cluster members will be written to the same location automatically.
    identity : float
        Sequence identity threshold in [0, 1]. CD-HIT reliable above ~0.4;
        use PSI-CD-HIT or MMseqs2 below that.
    word_size : int
        k-mer word size for initial matching. MUST match identity:
            identity 0.7-1.0 -> word_size 5
            identity 0.6-0.7 -> word_size 4
            identity 0.5-0.6 -> word_size 3
            identity 0.4-0.5 -> word_size 2
    memory_mb : int
        Memory cap in megabytes.
    threads : int
        Number of CPU threads (-T 0 uses all available).
    keep_description : int
        -d flag; 0 = keep full FASTA headers in output (recommended).

    Returns
    -------
    dict
        Run metadata including parameters, timing, input/output counts,
        and full stdout/stderr from CD-HIT.
    """
    cmd = [
        "cd-hit",
        "-i", str(input_fasta),
        "-o", str(output_fasta),
        "-c", str(identity),
        "-n", str(word_size),
        "-M", str(memory_mb),
        "-T", str(threads),
        "-d", str(keep_description),
    ]

    print(f"Running: {' '.join(cmd)}")
    start = datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    elapsed = (datetime.now() - start).total_seconds()

    input_count = count_sequences(input_fasta)
    output_count = count_sequences(output_fasta)

    return {
        "timestamp": start.isoformat(),
        "elapsed_seconds": elapsed,
        "command": " ".join(cmd),
        "parameters": {
            "identity": identity,
            "word_size": word_size,
            "memory_mb": memory_mb,
            "threads": threads,
        },
        "input_file": input_fasta,
        "output_file": output_fasta,
        "input_sequence_count": input_count,
        "output_sequence_count": output_count,
        "reduction_percent": 100 * (1 - output_count / input_count),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def count_sequences(fasta_path: str) -> int:
    """Count sequences in a FASTA file by counting header lines."""
    with open(fasta_path) as f:
        return sum(1 for line in f if line.startswith(">"))


def write_log(metadata: dict, log_dir: str = "logs") -> Path:
    """Write run metadata to a timestamped log file."""
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = metadata["timestamp"].replace(":", "-").split(".")[0]
    log_path = Path(log_dir) / f"cdhit_run_{timestamp}.log"

    with open(log_path, "w") as f:
        tool = metadata.get("tool", "cd-hit")
        f.write(f"{tool.upper()} run log\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Timestamp:        {metadata['timestamp']}\n")
        f.write(f"Elapsed:          {metadata['elapsed_seconds']:.1f} seconds\n")
        f.write(f"Command:          {metadata['command']}\n")
        f.write(f"\nParameters:\n")
        for key, value in metadata["parameters"].items():
            f.write(f"  {key:<20} {value}\n")
        f.write(f"\nInput:            {metadata['input_file']}\n")
        f.write(f"Input count:      {metadata['input_sequence_count']}\n")
        f.write(f"Output:           {metadata['output_file']}\n")
        f.write(f"Output count:     {metadata['output_sequence_count']}\n")
        f.write(f"Reduction:        {metadata['reduction_percent']:.1f}%\n")
        f.write(f"\n--- CD-HIT stdout ---\n{metadata['stdout']}\n")
        f.write(f"\n--- CD-HIT stderr ---\n{metadata['stderr']}\n")

    return log_path


if __name__ == "__main__":
    ## Run 1: 90% identity sanity check (near-duplicate removal)
    ## 4/22/2026
    # metadata = run_cdhit(
    #     input_fasta= BASE_DIR / "input_data" / "enzymes_no_mutants.fasta",
    #     output_fasta= BASE_DIR / "input_data" / "enzymes_nr90.fasta",
    #     identity=0.9,
    #     word_size=5,
    # )
    # log_path = write_log(metadata)

    # print(f"\nDone.")
    # print(f"  Input:  {metadata['input_sequence_count']} sequences")
    # print(f"  Output: {metadata['output_sequence_count']} sequences")
    # print(f"  Reduction: {metadata['reduction_percent']:.1f}%")
    # print(f"  Log saved to: {log_path}")

    # Run 2: 30% identity clustering with MMseqs2 (strict)
    # metadata = run_mmseqs2(
    #     input_fasta= BASE_DIR / "input_data" / "enzymes_no_mutants.fasta",
    #     output_prefix= BASE_DIR / "input_data" / "enzymes_nr40",
    #     identity=0.40,
    #     coverage=0.8,
    #     cov_mode=0,
    # )
    # log_path = write_log(metadata)

    # print(f"\nDone.")
    # print(f"  Input:  {metadata['input_sequence_count']} sequences")
    # print(f"  Output: {metadata['output_sequence_count']} sequences")
    # print(f"  Reduction: {metadata['reduction_percent']:.1f}%")
    # print(f"  Log saved to: {log_path}")

    # Run 3 - 5/6/2026 filtered FASTAs only for the enzymes
    metadata = run_mmseqs2(
        input_fasta= DATA_DIR / "enzymes" / "mmseq"/ "enzymes_FASTA.fasta",
        output_prefix= DATA_DIR / "enzymes" / "mmseq"/ "runs" / "050626_enzymes_nr40",
        identity=0.40,
        coverage=0.8,
        cov_mode=0,
    )
    log_path = write_log(metadata)

    print(f"\nDone.")
    print(f"  Input:  {metadata['input_sequence_count']} sequences")
    print(f"  Output: {metadata['output_sequence_count']} sequences")
    print(f"  Reduction: {metadata['reduction_percent']:.1f}%")
    print(f"  Log saved to: {log_path}")