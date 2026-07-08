import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *


def print_chain_distribution(csv_path: Path, label: str = None) -> None:
    """
    Print a frequency table of n_chains values from a size distribution CSV
    produced by 0_4_enzyme_distribution_ver4.py.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.
    label : str, optional
        Dataset label for the header. Defaults to the filename stem.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"ERROR: file not found — {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if "n_chains" not in df.columns:
        print(f"ERROR: 'n_chains' column not found in {csv_path.name}")
        return

    label = label or csv_path.stem
    n_total  = len(df)
    n_failed = df["n_chains"].isna().sum()
    n_valid  = n_total - n_failed

    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  Total entries : {n_total:,}")
    print(f"  Parsed OK     : {n_valid:,}")
    print(f"  Failed/NaN    : {n_failed:,}")
    print(f"\n  Chain count distribution:")
    print(f"  {'n_chains':>10}  {'count':>8}  {'%':>7}")
    print(f"  {'-'*30}")

    counts = (
        df["n_chains"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )
    for n_chains, count in counts.items():
        pct = 100 * count / n_valid
        print(f"  {n_chains:>10}  {count:>8,}  {pct:>6.1f}%")

    print(f"  {'Total':>10}  {n_valid:>8,}  {'100.0%':>7}")


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Configurable paths
    # ------------------------------------------------------------------

    ENZYME_CSV = FINAL_DIR / "figures" / "enzyme_dist_no_mc" / "no_mc_size_dist_enzyme_test_set.csv"
    REF_CSV    = FINAL_DIR / "figures" / "enzyme_dist_no_mc" / "no_mc_size_dist_ref_set.csv"

    # ------------------------------------------------------------------

    print_chain_distribution(ENZYME_CSV, label="Enzyme test set")
    print_chain_distribution(REF_CSV,    label="Reference set")
