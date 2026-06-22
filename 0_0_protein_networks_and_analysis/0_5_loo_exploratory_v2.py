"""
loo_exploratory_v2.py
====================
Exploratory analysis of StdBetweenness, RCA_zscore, and RSA-empirical
distributions for functional vs. non-functional residues across the
BioLiP enzyme set.

Purpose:
    Before defining the LOO threshold search grid for StdBetweenness and
    RCA_zscore (residue centrality z-score), this script examines where
    functional residues fall in those feature spaces. Summary statistics
    and percentile tables are saved to CSV for grid bound selection.

Outputs (all written to OUT_DIR):
    - residue_labeled.csv         : full merged dataset with functional labels
    - summary_stats.csv           : per-group summary statistics
    - percentile_table.csv        : percentile breakdown for grid bound selection
    - flagged_missing_rsa.csv     : residues with missing RSA values
    - exploratory_skipped.log     : proteins skipped (no annotation / excluded)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

# ---------------------------------------------------------------------------
# Paths — edit here or in main()
# ---------------------------------------------------------------------------
MERGED_CSV    = DATA_DIR_2 / "figures" / "controlled_fig1_recreation_ver0.7" / "test_standardized.csv"
BIOLIP_CSV    = DATA_DIR_2 / "biolip_functional_residues.csv"
OUT_DIR       = DATA_DIR_2 / "loo_exploratory_v2_betweenness_rca"

# PDB IDs to exclude explicitly
EXCLUDE_IDS   = {"7ara", "4q95"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("loo_exploratory")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(out_dir / "exploratory_skipped.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def parse_functional_residues(biolip_csv: Path, exclude_ids: set, logger: logging.Logger) -> dict:
    """
    Parse the BioLiP functional residues CSV.

    Returns
    -------
    dict : {(pdb_id, chain): set of int residue numbers}
        Only proteins with valid annotations are included.
        Proteins with NOT_FOUND or excluded IDs are logged and omitted.
    """
    df = pd.read_csv(biolip_csv)
    functional_map = {}
    
    for _, row in df.iterrows():
        pdb_id = str(row["pdb_id"]).lower().strip()
        chain  = str(row["chain"]).strip()

        # explicit exclusions
        if pdb_id in exclude_ids:
            logger.info(f"Skipping explicitly excluded protein: {pdb_id}")
            continue

        # missing annotation
        raw = str(row["functional_residues"]).strip()
        if raw == "NOT_FOUND" or raw == "" or pd.isna(row["functional_residues"]):
            logger.info(f"Skipping {pdb_id} chain {chain}: no BioLiP annotation (NOT_FOUND)")
            continue

        # parse X### tokens -> residue numbers (int)
        residue_nums = set()
        for token in raw.split():
            # token format: one-letter residue type + residue number, e.g. "C95"
            # strip leading alphabetic characters to get the number
            num_str = token.lstrip("ACDEFGHIKLMNPQRSTVWY")
            if num_str.lstrip("-").isdigit():
                residue_nums.add(int(num_str))
            else:
                logger.warning(f"Could not parse residue token '{token}' for {pdb_id} chain {chain}")

        if residue_nums:
            key = (pdb_id, chain)
            if key not in functional_map:
                functional_map[key] = set()
            functional_map[key].update(residue_nums)

    return functional_map


def load_and_label(merged_csv: Path, functional_map: dict,
                   exclude_ids: set, logger: logging.Logger) -> tuple:
    """
    Load the merged residue CSV and assign binary functional labels.

    Returns
    -------
    df_labeled : pd.DataFrame
        Full dataset with 'is_functional' column (1 / 0).
    df_missing_rsa : pd.DataFrame
        Residues where RSA-empirical is NaN.
    """
    df = pd.read_csv(merged_csv)

    # normalise pdb_id to lowercase
    df["pdb_id"] = df["pdb_id"].str.lower().str.strip()

    # drop explicitly excluded proteins
    before = len(df)
    df = df[~df["pdb_id"].isin(exclude_ids)]
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} rows for explicitly excluded PDB IDs")

    # drop proteins with no BioLiP annotation
    annotated_ids = {pdb_id for (pdb_id, _) in functional_map}
    before = len(df)
    df = df[df["pdb_id"].isin(annotated_ids)]
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} rows for proteins with no BioLiP annotation")

    # flag missing RSA
    df_missing_rsa = df[df["RSA-empirical"].isna()].copy()
    if len(df_missing_rsa):
        logger.warning(f"{len(df_missing_rsa)} residues have missing RSA-empirical values — see flagged_missing_rsa.csv")

    # assign functional label
    def is_functional(row):
        key = (row["pdb_id"], str(row["Chain"]).strip())
        if key not in functional_map:
            return 0
        return 1 if int(row["Residue number"]) in functional_map[key] else 0

    df["is_functional"] = df.apply(is_functional, axis=1)

    n_func     = df["is_functional"].sum()
    n_total    = len(df)
    n_proteins = df["pdb_id"].nunique()
    logger.info(f"Dataset: {n_total} residues across {n_proteins} proteins")
    logger.info(f"Functional residues: {n_func} ({100*n_func/n_total:.2f}%)")
    logger.info(f"Non-functional residues: {n_total - n_func}")

    return df, df_missing_rsa


def compute_summary_stats(df: pd.DataFrame) -> tuple:
    """
    Compute summary statistics and percentile tables for StdBetweenness,
    RCA_zscore, and RSA-empirical, split by functional label.

    Returns
    -------
    summary_stats : pd.DataFrame
    percentile_table : pd.DataFrame
    """
    features = ["StdBetweenness", "RCA_zscore", "RSA-empirical"]
    percentiles = [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]

    groups = {
        "functional":     df[df["is_functional"] == 1],
        "non_functional": df[df["is_functional"] == 0],
        "all":            df,
    }

    # --- summary stats ---
    summary_rows = []
    for group_name, gdf in groups.items():
        for feat in features:
            col = gdf[feat].dropna()
            summary_rows.append({
                "group":   group_name,
                "feature": feat,
                "n":       len(col),
                "mean":    col.mean(),
                "median":  col.median(),
                "std":     col.std(),
                "min":     col.min(),
                "max":     col.max(),
                "q25":     col.quantile(0.25),
                "q75":     col.quantile(0.75),
            })
    summary_stats = pd.DataFrame(summary_rows)

    # --- percentile table ---
    pct_rows = []
    for group_name, gdf in groups.items():
        for feat in features:
            col = gdf[feat].dropna()
            row = {"group": group_name, "feature": feat}
            for p in percentiles:
                row[f"p{p}"] = col.quantile(p / 100)
            pct_rows.append(row)
    percentile_table = pd.DataFrame(pct_rows)

    return summary_stats, percentile_table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    merged_csv: Path  = MERGED_CSV,
    biolip_csv: Path  = BIOLIP_CSV,
    out_dir: Path     = OUT_DIR,
    exclude_ids: set  = EXCLUDE_IDS,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)

    logger.info("=" * 60)
    logger.info("LOO EXPLORATORY ANALYSIS — StdBetweenness & RCA_zscore")
    logger.info(f"  Merged CSV : {merged_csv}")
    logger.info(f"  BioLiP CSV : {biolip_csv}")
    logger.info(f"  Output dir : {out_dir}")
    logger.info("=" * 60)

    # 1. parse functional residues
    logger.info("Parsing BioLiP functional residues...")
    functional_map = parse_functional_residues(biolip_csv, exclude_ids, logger)
    logger.info(f"Loaded annotations for {len(functional_map)} (pdb_id, chain) pairs")

    # 2. load and label residues
    logger.info("Loading merged CSV and assigning functional labels...")
    df_labeled, df_missing_rsa = load_and_label(merged_csv, functional_map, exclude_ids, logger)

    # 3. compute summary statistics
    logger.info("Computing summary statistics...")
    summary_stats, percentile_table = compute_summary_stats(df_labeled)

    # 4. save outputs
    df_labeled.to_csv(out_dir / "residue_labeled.csv", index=False)
    logger.info(f"Saved residue_labeled.csv ({len(df_labeled)} rows)")

    summary_stats.to_csv(out_dir / "summary_stats.csv", index=False)
    logger.info("Saved summary_stats.csv")

    percentile_table.to_csv(out_dir / "percentile_table.csv", index=False)
    logger.info("Saved percentile_table.csv")

    if len(df_missing_rsa):
        df_missing_rsa.to_csv(out_dir / "flagged_missing_rsa.csv", index=False)
        logger.info(f"Saved flagged_missing_rsa.csv ({len(df_missing_rsa)} rows)")
    else:
        logger.info("No missing RSA values found.")

    # 5. print summary to console for quick inspection
    print("\n=== SUMMARY STATISTICS ===")
    print(summary_stats.to_string(index=False))
    print("\n=== PERCENTILE TABLE ===")
    print(percentile_table.to_string(index=False))

    logger.info("Exploratory analysis complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
