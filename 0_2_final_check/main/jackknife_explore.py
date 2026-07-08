"""
jackknife_explore.py
=====================
Exploratory analysis of StdCloseness, StdBetweenness, RCA_zscore,
StdPageRank, and RSA-empirical distributions across the BioLiP
enzyme set.

Purpose:
    Before defining the jackknife (leave-one-out) threshold search grid
    for StdCloseness, StdBetweenness, RCA_zscore, and StdPageRank, this
    script examines where functional residues fall in those feature
    spaces. Summary statistics and percentile tables are saved to CSV
    for grid bound selection.

    StdPageRank is the per-protein z-score of raw PageRank values
    computed by append_pagerank_to_centrality() in
    parallel_RIN_analysis.py (damping factor alpha=0.85,
    Brin & Page 1998). Z-scores use population std (ddof=0), consistent
    with StdCloseness and StdBetweenness.

Outputs (all written to out_dir):
    - residue_labeled.csv           : dataset with is_functional +
                                      functional_type columns (as read
                                      from test_standardized.csv)
    - summary_stats.csv             : per-group summary statistics
                                      (groups: all, functional, non_functional,
                                       catalytic, binding, dual)
    - percentile_table.csv          : percentile breakdown for all groups
    - rsa_type_comparison.csv       : focused RSA comparison across types
    - flagged_missing_rsa.csv       : residues with missing RSA-empirical
    - flagged_missing_pagerank.csv  : residues with missing StdPageRank
    - exploratory_skipped.log       : proteins skipped / warnings
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

# Paths — edit here or in main()

MERGED_CSV = FIGS_DIR / "test_standardized.csv"
OUT_DIR    = DATA_DIR_3 / "test" / "jackknife_explore_all_metrics"

# Functional types — priority order (dual > catalytic > binding)
FUNCTIONAL_TYPES = ["catalytic", "binding", "dual"]


# Logging


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("jackknife_explore")
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


# Load, exclude, and validate labels

def load_and_label(merged_csv: Path, exclude_ids: set,
                    logger: logging.Logger) -> tuple:
    """
    Load test_standardized.csv, exclude bad proteins, and validate that
    functional_type / is_functional are present.

    Both columns already live in test_standardized.csv, so no join
    against a separate BioLiP labels file is needed here.

    Returns
    -------
    df                   : pd.DataFrame  with 'functional_type', 'is_functional'
    df_missing_rsa       : pd.DataFrame  residues where RSA-empirical is NaN
    df_missing_pagerank  : pd.DataFrame  residues where StdPageRank is NaN
    """
    # 1. Load merged CSV
    df = pd.read_csv(merged_csv)
    df["pdb_id"] = df["pdb_id"].str.lower().str.strip()

    # 2. Drop excluded proteins
    before = len(df)
    df = df[~df["pdb_id"].isin(exclude_ids)]
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped:,} rows for explicitly excluded PDB IDs")

    # 3. Validate required label columns are present
    for col in ("functional_type", "is_functional"):
        if col not in df.columns:
            raise KeyError(
                f"'{col}' not found in {merged_csv}. This script expects "
                f"functional_type and is_functional to already be columns "
                f"in test_standardized.csv."
            )

    # Normalize is_functional to a clean 0/1 int regardless of whether it
    # was read in as bool, "True"/"False" strings, or 0/1.
    df["is_functional"] = df["is_functional"].astype(bool).astype(int)
    df["functional_type"] = df["functional_type"].fillna("non_functional")

    # 4. Flag missing RSA
    df_missing_rsa = df[df["RSA-empirical"].isna()].copy()
    if len(df_missing_rsa):
        logger.warning(
            f"{len(df_missing_rsa):,} residues have missing RSA-empirical — "
            f"see flagged_missing_rsa.csv"
        )

    # 5. Check for StdPageRank column
    if "StdPageRank" not in df.columns:
        logger.warning(
            "Column 'StdPageRank' not found in merged CSV. "
            "Run append_pagerank_to_centrality() in "
            "0_3_1_parallel_RIN_analysis_ver4.py and regenerate the merged CSV. "
            "StdPageRank will appear as NaN in all outputs."
        )
        df["StdPageRank"] = float("nan")

    df_missing_pagerank = df[df["StdPageRank"].isna()].copy()
    if len(df_missing_pagerank):
        logger.warning(
            f"{len(df_missing_pagerank):,} residues have missing StdPageRank — "
            f"see flagged_missing_pagerank.csv"
        )

    # 6. Log breakdown
    n_func     = int(df["is_functional"].sum())
    n_total    = len(df)
    n_proteins = df["pdb_id"].nunique()
    logger.info(f"Dataset: {n_total:,} residues across {n_proteins} proteins")
    logger.info(f"Functional residues : {n_func:,} ({100 * n_func / n_total:.2f}%)")
    logger.info(f"Non-functional      : {n_total - n_func:,}")

    counts = df["functional_type"].value_counts()
    logger.info("Functional type breakdown:")
    for ftype in FUNCTIONAL_TYPES + ["non_functional"]:
        logger.info(f"  {ftype:<16}: {counts.get(ftype, 0):,}")

    return df, df_missing_rsa, df_missing_pagerank

# Feature checks

def check_missing_features(df: pd.DataFrame, features: list,
                            logger: logging.Logger) -> None:
    """
    Warn if any feature column is absent or entirely NaN.
    """
    for feat in features:
        if feat not in df.columns:
            logger.warning(
                f"Feature column '{feat}' not found in merged CSV — "
                f"absent from all outputs."
            )
        else:
            n_nan = df[feat].isna().sum()
            if n_nan == len(df):
                logger.warning(
                    f"Feature column '{feat}' is entirely NaN — "
                    f"check upstream pipeline."
                )
            elif n_nan > 0:
                logger.info(
                    f"Feature '{feat}': {n_nan:,} NaN values "
                    f"({100 * n_nan / len(df):.1f}%) — excluded per residue."
                )


# Summary statistics (pooled + per-type)

def compute_summary_stats(df: pd.DataFrame, features: list) -> tuple:
    """
    Compute summary statistics and percentile tables for the given features.

    Groups:
        all, functional, non_functional, catalytic, binding, dual

    The per-type groups allow direct comparison of RSA (and centrality)
    distributions across functional subtypes before committing to a shared
    grid.

    Parameters
    ----------
    df       : pd.DataFrame
        Must contain 'is_functional', 'functional_type', and feature columns.
    features : list of str

    Returns
    -------
    summary_stats    : pd.DataFrame
    percentile_table : pd.DataFrame
    """
    percentiles = [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]

    groups = {
        "all":            df,
        "functional":     df[df["is_functional"] == 1],
        "non_functional": df[df["is_functional"] == 0],
    }
    for ftype in FUNCTIONAL_TYPES:
        groups[ftype] = df[df["functional_type"] == ftype]

    # --- summary stats ---
    summary_rows = []
    for group_name, gdf in groups.items():
        for feat in features:
            if feat not in gdf.columns:
                continue
            col = gdf[feat].dropna()
            if col.empty:
                continue
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
            if feat not in gdf.columns:
                continue
            col = gdf[feat].dropna()
            if col.empty:
                continue
            row = {"group": group_name, "feature": feat}
            for p in percentiles:
                row[f"p{p}"] = col.quantile(p / 100)
            pct_rows.append(row)
    percentile_table = pd.DataFrame(pct_rows)

    return summary_stats, percentile_table


# RSA type comparison table

def build_rsa_type_comparison(summary_stats: pd.DataFrame,
                               percentile_table: pd.DataFrame) -> pd.DataFrame:
    """
    Build a focused RSA comparison table across functional types.

    Rows:    non_functional, functional (pooled), catalytic, binding, dual
    Columns: group, n, mean, median, std, q25, q75, p10, p90

    Primary output for deciding whether a shared RSA band is appropriate
    or whether the types are too different to pool.
    """
    groups_order = ["non_functional", "functional", "catalytic", "binding", "dual"]
    feat = "RSA-empirical"
    rows = []

    for grp in groups_order:
        ss  = summary_stats[
            (summary_stats["group"] == grp) &
            (summary_stats["feature"] == feat)
        ]
        pct = percentile_table[
            (percentile_table["group"] == grp) &
            (percentile_table["feature"] == feat)
        ]
        if ss.empty:
            continue
        rows.append({
            "group":  grp,
            "n":      int(ss["n"].values[0]),
            "mean":   ss["mean"].values[0],
            "median": ss["median"].values[0],
            "std":    ss["std"].values[0],
            "q25":    ss["q25"].values[0],
            "q75":    ss["q75"].values[0],
            "p10":    pct["p10"].values[0] if not pct.empty else float("nan"),
            "p90":    pct["p90"].values[0] if not pct.empty else float("nan"),
        })

    return pd.DataFrame(rows)

# Main

def main(
    merged_csv  = MERGED_CSV,
    out_dir     = OUT_DIR,
    exclude_ids = {},
):
    merged_csv = Path(merged_csv)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir)

    logger.info("=" * 60)
    logger.info(
        "JACKKNIFE EXPLORATORY ANALYSIS — "
        "StdCloseness, StdBetweenness, RCA_zscore, StdPageRank + per-type RSA"
    )
    logger.info(f"  Merged CSV   : {merged_csv}")
    logger.info(f"  Output dir   : {out_dir}")
    logger.info(
        "  NOTE: is_functional and functional_type read directly from "
        "test_standardized.csv"
    )
    logger.info("=" * 60)

    # Feature lists
    features = [
        "StdCloseness", "StdBetweenness", "RCA_zscore",
        "StdPageRank", "RSA-empirical",
    ]

    # 1. Load, exclude, validate labels
    logger.info("Loading merged CSV and validating labels...")
    df_labeled, df_missing_rsa, df_missing_pagerank = load_and_label(
        merged_csv, exclude_ids, logger
    )

    # 2. Check feature availability
    logger.info("Checking feature columns...")
    check_missing_features(df_labeled, features, logger)

    # 3. Compute summary statistics (pooled + per-type)
    logger.info("Computing summary statistics...")
    summary_stats, percentile_table = compute_summary_stats(df_labeled, features)

    # 4. Build RSA type comparison
    rsa_comparison = build_rsa_type_comparison(summary_stats, percentile_table)

    # 5. Save outputs
    df_labeled.to_csv(out_dir / "residue_labeled.csv", index=False)
    logger.info(f"Saved residue_labeled.csv ({len(df_labeled):,} rows)")

    summary_stats.to_csv(out_dir / "summary_stats.csv", index=False)
    logger.info("Saved summary_stats.csv")

    percentile_table.to_csv(out_dir / "percentile_table.csv", index=False)
    logger.info("Saved percentile_table.csv")

    rsa_comparison.to_csv(out_dir / "rsa_type_comparison.csv", index=False)
    logger.info("Saved rsa_type_comparison.csv")

    if len(df_missing_rsa):
        df_missing_rsa.to_csv(out_dir / "flagged_missing_rsa.csv", index=False)
        logger.info(f"Saved flagged_missing_rsa.csv ({len(df_missing_rsa):,} rows)")
    else:
        logger.info("No missing RSA-empirical values found.")

    if len(df_missing_pagerank):
        df_missing_pagerank.to_csv(out_dir / "flagged_missing_pagerank.csv", index=False)
        logger.info(f"Saved flagged_missing_pagerank.csv ({len(df_missing_pagerank):,} rows)")
    else:
        logger.info("No missing StdPageRank values found.")

    # 6. Console output
    print("\n=== SUMMARY STATISTICS ===")
    print(summary_stats.to_string(index=False))
    print("\n=== PERCENTILE TABLE ===")
    print(percentile_table.to_string(index=False))

    logger.info("Jackknife exploratory analysis complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
