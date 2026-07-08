"""
jackknife_experiment.py
------------------------
Leave-one-out (jackknife) cross-validation for active site prediction.

Prediction rule (all metrics share this structure):
    A residue is predicted functional if ALL of the following hold:
        centrality_lower <= <centrality_metric> <= centrality_upper
        rsa_lower        <= RSA-empirical        <= rsa_upper

Grid:
    For each feature axis, candidate values are generated from
    [axis_min, axis_max] using np.linspace(axis_min, axis_max, n_axis),
    giving exactly n_axis evenly-spaced candidates regardless of the
    axis's range. The grid is the Cartesian product of all valid
    (lower, upper) pairs for each feature, where lower < upper. The
    jackknife searches all valid (lower, upper) pairs for each feature
    axis simultaneously:

        n_c_lower * n_c_upper * n_rsa_lower * n_rsa_upper combinations
        (minus invalid pairs where lower >= upper)

    which is equivalent to C(n_c, 2) * C(n_rsa, 2) when the lower
    and upper axes share the same candidate values.

Metrics:
    1. StdCloseness
    2. StdBetweenness
    3. RCA_zscore
    4. StdPageRank

Output (per metric, written to output_dir):
    jackknife_results_closeness.csv
    jackknife_results_betweenness.csv
    jackknife_results_rca.csv
    jackknife_results_pagerank.csv
    jackknife_summary_all_metrics.csv   -- one row per metric: median MCC,
                                            mean sensitivity/specificity,
                                            and per-functional-type
                                            nanmedian MCC
    jackknife_skipped.log

Result columns per protein:
    pdb_id, centrality_lower, centrality_upper, rsa_lower, rsa_upper,
    tp, tn, fp, fn, sensitivity, specificity, mcc,
    tp_{ftype}, tn_{ftype}, fp_{ftype}, fn_{ftype}, mcc_{ftype}
    for ftype in {catalytic, binding, dual}

Proteins with no functional residues are skipped and logged.
Undefined overall MCC is recorded as 0.0.
Undefined stratified MCC is recorded as NaN.
"""

import sys
import logging
import itertools
from math import comb
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from imports import FIGS_DIR, DATA_DIR_3

FUNCTIONAL_TYPES = ["catalytic", "binding", "dual"]

# Mapping from metric name to its column in the CSV
METRIC_COLUMN = {
    "closeness":   "StdCloseness",
    "betweenness": "StdBetweenness",
    "rca":         "RCA_zscore",
    "pagerank":    "StdPageRank",
}


# ---------------------------------------------------------------------------
# MCC helpers
# ---------------------------------------------------------------------------

def compute_mcc(y_true, y_pred):
    """
    Matthews Correlation Coefficient.
    Returns 0.0 for undefined cases (zero denominator).
    Used for overall MCC and the MCC matrix (training fold optimisation).
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def compute_mcc_nan(y_true, y_pred):
    """
    Matthews Correlation Coefficient.
    Returns np.nan for undefined cases (zero denominator).
    Used for stratified MCC so that unevaluable strata are excluded
    from nanmedian summaries rather than dragging the median toward zero.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return np.nan
    return (tp * tn - fp * fn) / denom


# ---------------------------------------------------------------------------
# Prediction rule (band-based)
# ---------------------------------------------------------------------------

def predict(centrality_arr, rsa_arr, c_lower, c_upper, rsa_lower, rsa_upper):
    """
    Apply the band-based threshold rule to per-residue feature arrays.

    A residue is predicted functional if it falls inside the window:
        c_lower    <= centrality <= c_upper
        rsa_lower  <= RSA        <= rsa_upper

    Returns a binary integer array (1 = predicted functional, 0 = not).
    """
    return (
        (centrality_arr >= c_lower)  &
        (centrality_arr <= c_upper)  &
        (rsa_arr        >= rsa_lower) &
        (rsa_arr        <= rsa_upper)
    ).astype(int)


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def build_grid(
    centrality_min, centrality_max, n_centrality,
    rsa_min,        rsa_max,        n_rsa,
):
    """
    Build the full band-based threshold grid.

    For each feature (centrality, RSA), a single set of candidate values
    is generated from [feature_min, feature_max] using
    np.linspace(feature_min, feature_max, n_feature) -- exactly n_feature
    evenly-spaced values including both endpoints. The grid is the
    Cartesian product of all valid (lower, upper) pairs for each
    feature, where lower < upper.

    Using n directly (rather than a step size) makes it easy to hold
    grid size fixed across metrics with different ranges: passing the
    same n_centrality for StdCloseness, StdBetweenness, RCA_zscore, and
    StdPageRank guarantees each metric searches the same number of
    candidate values -- C(n_centrality, 2) pairs -- even though their
    per-unit step sizes differ.

    This is equivalent to choosing all pairs from the candidate set:
        C(n_centrality, 2) * C(n_rsa, 2) combinations.

    For example, with n_centrality=10 and n_rsa=6:
        C(10,2) = 45 centrality pairs
        C(6,2)  = 15 RSA pairs
        Total   = 45 * 15 = 675 combinations

    Parameters
    ----------
    centrality_min/max : bounds for candidate centrality values
    n_centrality        : number of evenly-spaced centrality candidates
    rsa_min/max         : bounds for candidate RSA values
    n_rsa                : number of evenly-spaced RSA candidates

    Returns
    -------
    grid          : list of (c_lower, c_upper, rsa_lower, rsa_upper) tuples
    c_vals        : np.ndarray of candidate centrality values
    rsa_vals      : np.ndarray of candidate RSA values
    """
    c_vals = np.round(
        np.linspace(centrality_min, centrality_max, n_centrality), 6
    )
    rsa_vals = np.round(
        np.linspace(rsa_min, rsa_max, n_rsa), 6
    )

    grid = [
        (cl, cu, rl, ru)
        for (cl, cu), (rl, ru) in itertools.product(
            itertools.combinations(c_vals,   2),
            itertools.combinations(rsa_vals, 2),
        )
    ]
    # combinations() already guarantees cl < cu and rl < ru by construction
    return grid, c_vals, rsa_vals


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_validate(input_csv):
    """
    Load test_standardized.csv and validate required columns.

    Required columns:
        pdb_id, Chain, Residue number, RSA-empirical,
        StdCloseness, StdBetweenness, RCA_zscore, StdPageRank,
        functional_type, is_functional

    functional_type and is_functional are read directly from the CSV --
    there is no separate BioLiP labels file to join against anymore.

    Returns a cleaned DataFrame.
    """
    df = pd.read_csv(input_csv)

    required_cols = {
        "pdb_id", "Chain", "Residue number", "RSA-empirical",
        "StdCloseness", "StdBetweenness", "RCA_zscore", "StdPageRank",
        "functional_type", "is_functional",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df["StdCloseness"]    = pd.to_numeric(df["StdCloseness"],    errors="coerce")
    df["StdBetweenness"]  = pd.to_numeric(df["StdBetweenness"],  errors="coerce")
    df["RCA_zscore"]      = pd.to_numeric(df["RCA_zscore"],      errors="coerce")
    df["StdPageRank"]     = pd.to_numeric(df["StdPageRank"],     errors="coerce")
    df["RSA-empirical"]   = pd.to_numeric(df["RSA-empirical"],   errors="coerce")
    df["pdb_id"]          = df["pdb_id"].str.lower()

    # Normalize is_functional to a clean bool regardless of whether it was
    # read in as bool, "True"/"False" strings, or 0/1.
    df["is_functional"]   = df["is_functional"].astype(bool)
    df["functional_type"] = df["functional_type"].fillna("non_functional")

    return df


# ---------------------------------------------------------------------------
# Per-protein data builder
# ---------------------------------------------------------------------------

def build_protein_data(df, logger):
    """
    Group residue-level DataFrame by protein and extract feature arrays
    for all four centrality metrics and RSA.

    Proteins with no functional residues are skipped and logged.
    NaN values are logged; NaN comparisons return False so those
    residues are always predicted non-functional.

    Stores per-type boolean masks (functional_type is always present now).
    Returns protein_data dict and skipped list.
    """
    protein_data = {}
    skipped      = []
    has_types    = "functional_type" in df.columns

    for col in ["RSA-empirical", "StdCloseness", "StdBetweenness",
                "RCA_zscore", "StdPageRank"]:
        nan_rows = df[df[col].isna()]
        if not nan_rows.empty:
            per_prot = nan_rows.groupby("pdb_id").size()
            logger.warning(
                f"NaN '{col}' found in {len(per_prot)} protein(s), "
                f"{len(nan_rows)} residues total. These residues will always be "
                f"predicted non-functional."
            )
            for pdb_id, count in per_prot.items():
                func_nan = nan_rows[nan_rows["pdb_id"] == pdb_id]["is_functional"].sum()
                logger.warning(
                    f"  NaN {col}: {pdb_id} -- {count} residues "
                    f"({func_nan} functional, {count - func_nan} non-functional)"
                )

    for pdb_id, grp in df.groupby("pdb_id"):
        y_true = grp["is_functional"].values.astype(int)
        if y_true.sum() == 0:
            logger.info(f"SKIP (no functional residues): {pdb_id}")
            skipped.append((pdb_id, "no_functional_residues"))
            continue

        entry = {
            "closeness":   grp["StdCloseness"].values,
            "betweenness": grp["StdBetweenness"].values,
            "rca":         grp["RCA_zscore"].values,
            "pagerank":    grp["StdPageRank"].values,
            "rsa":         grp["RSA-empirical"].values,
            "y_true":      y_true,
        }

        if has_types:
            non_func_mask = grp["functional_type"].values == "non_functional"
            for ftype in FUNCTIONAL_TYPES:
                type_mask    = grp["functional_type"].values == ftype
                stratum_mask = type_mask | non_func_mask
                entry[f"y_true_{ftype}"]      = type_mask[stratum_mask].astype(int)
                entry[f"stratum_{ftype}"]      = stratum_mask
                entry[f"closeness_{ftype}"]   = grp["StdCloseness"].values[stratum_mask]
                entry[f"betweenness_{ftype}"] = grp["StdBetweenness"].values[stratum_mask]
                entry[f"rca_{ftype}"]         = grp["RCA_zscore"].values[stratum_mask]
                entry[f"pagerank_{ftype}"]    = grp["StdPageRank"].values[stratum_mask]
                entry[f"rsa_{ftype}"]         = grp["RSA-empirical"].values[stratum_mask]

        protein_data[pdb_id] = entry

    return protein_data, skipped


# ---------------------------------------------------------------------------
# MCC matrix and LOO
# ---------------------------------------------------------------------------

def compute_mcc_matrix(proteins, protein_data, grid, metric):
    """
    Pre-compute MCC for every (protein, threshold combo) pair.

    Parameters
    ----------
    proteins     : list of pdb_id strings
    protein_data : dict from build_protein_data()
    grid         : list of (c_lower, c_upper, rsa_lower, rsa_upper) tuples
    metric       : one of 'closeness', 'betweenness', 'rca', 'pagerank'

    Returns a float32 array of shape (n_proteins, n_combos).
    """
    n          = len(proteins)
    mcc_matrix = np.zeros((n, len(grid)), dtype=np.float32)

    for i, pdb_id in enumerate(proteins):
        d              = protein_data[pdb_id]
        centrality_arr = d[metric]
        for j, (cl, cu, rl, ru) in enumerate(grid):
            y_pred           = predict(centrality_arr, d["rsa"], cl, cu, rl, ru)
            mcc_matrix[i, j] = compute_mcc(d["y_true"], y_pred)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"    MCC matrix [{metric}]: {i + 1}/{n} proteins done")

    return mcc_matrix


def compute_stratified_confusion(d, y_pred, ftype):
    """
    Compute the confusion matrix for a single functional type stratum.

    y_pred is the full-protein prediction array; this function subsets
    it to the stratum (type residues + non-functional residues).

    Returns a dict with tp, tn, fp, fn, mcc — or None if no type data.
    mcc is np.nan if the denominator is zero (unevaluable stratum).
    """
    stratum_mask = d.get(f"stratum_{ftype}")
    y_true_strat = d.get(f"y_true_{ftype}")
    if stratum_mask is None or y_true_strat is None:
        return None

    y_pred_strat = y_pred[stratum_mask]

    tp  = int(np.sum((y_true_strat == 1) & (y_pred_strat == 1)))
    tn  = int(np.sum((y_true_strat == 0) & (y_pred_strat == 0)))
    fp  = int(np.sum((y_true_strat == 0) & (y_pred_strat == 1)))
    fn  = int(np.sum((y_true_strat == 1) & (y_pred_strat == 0)))
    mcc = float(compute_mcc_nan(y_true_strat, y_pred_strat))

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "mcc": mcc}


def run_loo(proteins, protein_data, grid, mcc_matrix, metric, logger):
    """
    For each held-out protein, select the best threshold combo from the
    remaining n-1 proteins (by median MCC) and score the held-out protein.

    Parameters
    ----------
    proteins     : list of pdb_id strings
    protein_data : dict from build_protein_data()
    grid         : list of (c_lower, c_upper, rsa_lower, rsa_upper) tuples
    mcc_matrix   : float32 array (n_proteins x n_combos)
    metric       : one of 'closeness', 'betweenness', 'rca', 'pagerank'
    logger       : logging.Logger

    Returns a list of result dicts.
    """
    results   = []
    n         = len(proteins)
    has_types = f"y_true_{FUNCTIONAL_TYPES[0]}" in protein_data[proteins[0]]

    for i, pdb_id in enumerate(proteins):
        training_rows = np.delete(mcc_matrix, i, axis=0)
        median_mccs   = np.median(training_rows, axis=0)
        best_j        = int(np.argmax(median_mccs))
        best_cl, best_cu, best_rl, best_ru = grid[best_j]

        d              = protein_data[pdb_id]
        centrality_arr = d[metric]
        y_pred         = predict(centrality_arr, d["rsa"],
                                 best_cl, best_cu, best_rl, best_ru)
        y_true         = d["y_true"]

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        mcc         = compute_mcc(y_true, y_pred)

        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denom == 0:
            logger.info(
                f"EDGE CASE [{metric}] (MCC undefined, recorded as 0): "
                f"{pdb_id} | tp={tp} tn={tn} fp={fp} fn={fn}"
            )

        result = {
            "pdb_id":             pdb_id,
            "centrality_lower":   best_cl,
            "centrality_upper":   best_cu,
            "rsa_lower":          best_rl,
            "rsa_upper":          best_ru,
            "tp":                 tp,
            "tn":                 tn,
            "fp":                 fp,
            "fn":                 fn,
            "sensitivity":        round(sensitivity, 6),
            "specificity":        round(specificity, 6),
            "mcc":                round(float(mcc), 6),
        }

        if has_types:
            for ftype in FUNCTIONAL_TYPES:
                confusion = compute_stratified_confusion(d, y_pred, ftype)
                if confusion is None:
                    result[f"tp_{ftype}"]  = None
                    result[f"tn_{ftype}"]  = None
                    result[f"fp_{ftype}"]  = None
                    result[f"fn_{ftype}"]  = None
                    result[f"mcc_{ftype}"] = None
                else:
                    strat_mcc = confusion["mcc"]
                    if np.isnan(strat_mcc):
                        logger.info(
                            f"EDGE CASE [{metric}] (stratified MCC undefined, "
                            f"recorded as NaN): {pdb_id} | stratum={ftype}"
                        )
                    result[f"tp_{ftype}"]  = confusion["tp"]
                    result[f"tn_{ftype}"]  = confusion["tn"]
                    result[f"fp_{ftype}"]  = confusion["fp"]
                    result[f"fn_{ftype}"]  = confusion["fn"]
                    result[f"mcc_{ftype}"] = (
                        round(strat_mcc, 6)
                        if not np.isnan(strat_mcc) else np.nan
                    )

        results.append(result)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"    LOO [{metric}]: {i + 1}/{n} proteins done")

    return results


# ---------------------------------------------------------------------------
# Shared printing / logging helpers
# ---------------------------------------------------------------------------

def _build_summary_row(results_df, metric_label):
    """
    Build a single summary dict for one metric's LOO results.

    Mirrors exactly what _print_summary() reports, so the CSV output
    and the console output never drift out of sync.
    """
    row = {
        "metric":          metric_label,
        "n_proteins":      len(results_df),
        "median_mcc":      round(float(results_df["mcc"].median()), 6),
        "mean_sensitivity": round(float(results_df["sensitivity"].mean()), 6),
        "mean_specificity": round(float(results_df["specificity"].mean()), 6),
    }

    if "mcc_catalytic" in results_df.columns:
        for ftype in FUNCTIONAL_TYPES:
            col    = f"mcc_{ftype}"
            n_eval = results_df[col].notna().sum()
            med_val = (
                np.nanmedian(results_df[col].dropna())
                if n_eval > 0 else float("nan")
            )
            row[f"nanmedian_mcc_{ftype}"] = (
                round(float(med_val), 6) if n_eval > 0 else np.nan
            )
            row[f"n_evaluable_{ftype}"] = int(n_eval)

    return row


def _print_summary(results_df, metric_label, results_path, log_path):
    print(f"\n-- LOO Summary [{metric_label}] " + "-" * 40)
    print(f"  Proteins evaluated          : {len(results_df)}")
    print(f"  Median MCC (overall)        : {results_df['mcc'].median():.4f}")
    print(f"  Mean sensitivity            : {results_df['sensitivity'].mean():.4f}")
    print(f"  Mean specificity            : {results_df['specificity'].mean():.4f}")
    if "mcc_catalytic" in results_df.columns:
        for ftype in FUNCTIONAL_TYPES:
            col     = f"mcc_{ftype}"
            n_eval  = results_df[col].notna().sum()
            med_val = (
                np.nanmedian(results_df[col].dropna())
                if n_eval > 0 else float("nan")
            )
            print(f"  Nanmedian MCC ({ftype:<12})  : "
                  f"{med_val:.4f}  (n evaluable = {n_eval}/{len(results_df)})")
    print(f"\n  Results saved to            : {results_path}")
    print(f"  Log saved to                : {log_path}")
    print("-" * 61)


def _log_summary(logger, results_df, metric_label):
    logger.info(
        f"[{metric_label}] LOO complete. "
        f"{len(results_df)} proteins evaluated."
    )
    logger.info(
        f"[{metric_label}] Median MCC (overall): "
        f"{results_df['mcc'].median():.4f}"
    )
    if "mcc_catalytic" in results_df.columns:
        for ftype in FUNCTIONAL_TYPES:
            col     = f"mcc_{ftype}"
            n_eval  = results_df[col].notna().sum()
            med_val = (
                np.nanmedian(results_df[col].dropna())
                if n_eval > 0 else float("nan")
            )
            logger.info(
                f"[{metric_label}] Nanmedian MCC ({ftype}): {med_val:.4f}  "
                f"(n evaluable = {n_eval}/{len(results_df)})"
            )


# ---------------------------------------------------------------------------
# Tester
# ---------------------------------------------------------------------------

def run_tester(
    input_csv    = FIGS_DIR / "test_standardized.csv",
    n_proteins   = 5,
    random_seed  = 42,

    # StdCloseness candidate values (shared lower/upper axis)
    closeness_min  = 0.4,
    closeness_max  = 1.5,
    closeness_n    = 10,

    # StdBetweenness candidate values
    betweenness_min = None,
    betweenness_max = None,
    betweenness_n   = None,

    # RCA_zscore candidate values
    rca_min = None,
    rca_max = None,
    rca_n   = None,

    # StdPageRank candidate values
    pagerank_min = None,
    pagerank_max = None,
    pagerank_n   = None,

    # RSA candidate values (shared lower/upper axis)
    rsa_min = 0.00,
    rsa_max = 0.84,
    rsa_n   = 10,
):
    """
    Two-stage tester for all four metrics.

    Stage 1 -- Data & Grid Check
        Loads the full CSV, reports shape, column dtypes, NaN counts,
        functional residue counts, functional type breakdown (read
        directly from the CSV -- no separate labels file needed), and
        confirms the grid size for each metric. Metrics whose bounds
        are None are reported as skipped.

    Stage 2 -- Small End-to-End LOO Run
        Randomly samples n_proteins proteins that have at least one
        functional residue and runs the full LOO procedure on those
        proteins. Prints per-protein results including stratified MCC.
    """
    input_csv = Path(input_csv)

    null_logger = logging.getLogger("jackknife_tester")
    null_logger.addHandler(logging.NullHandler())

    # -- Stage 1: Data & Grid Check ----------------------------------------
    print("=" * 60)
    print("STAGE 1: Data & Grid Check")
    print("=" * 60)

    df = load_and_validate(input_csv)

    print(f"\n  CSV path        : {input_csv}")
    print(f"  Total rows      : {len(df):,}")
    print(f"  Total proteins  : {df['pdb_id'].nunique()}")

    print(f"\n  Column dtypes:")
    for col in ["pdb_id", "StdCloseness", "StdBetweenness",
                "RCA_zscore", "StdPageRank", "RSA-empirical", "is_functional"]:
        print(f"    {col:<22} {df[col].dtype}")

    print(f"\n  NaN counts:")
    for col in ["StdCloseness", "StdBetweenness", "RCA_zscore",
                "StdPageRank", "RSA-empirical"]:
        print(f"    {col:<22} {df[col].isna().sum():,}")

    n_functional = df["is_functional"].sum()
    print(f"\n  Functional residues     : {n_functional:,} "
          f"({100 * n_functional / len(df):.1f}%)")
    print(f"  Non-functional residues : {(~df['is_functional']).sum():,}")

    proteins_with_func = df.groupby("pdb_id")["is_functional"].sum().gt(0).sum()
    print(f"\n  Proteins with >=1 functional residue : {proteins_with_func}")
    print(f"  Proteins with 0  functional residues : "
          f"{df['pdb_id'].nunique() - proteins_with_func}")

    ft_counts = df["functional_type"].value_counts()
    print(f"\n  Functional type breakdown (read directly from CSV):")
    for ftype in FUNCTIONAL_TYPES + ["non_functional"]:
        print(f"    {ftype:<16} : {ft_counts.get(ftype, 0):,}")

    # Build and report grids
    metric_grids = {}
    metric_configs = [
        ("closeness",   closeness_min,   closeness_max,   closeness_n),
        ("betweenness", betweenness_min, betweenness_max, betweenness_n),
        ("rca",         rca_min,         rca_max,         rca_n),
        ("pagerank",    pagerank_min,    pagerank_max,    pagerank_n),
    ]
    for metric_name, c_min, c_max, c_n in metric_configs:
        if None in (c_min, c_max, c_n):
            print(f"\n  [{metric_name}] Grid SKIPPED -- bounds not set.")
            metric_grids[metric_name] = None
            continue
        grid, c_vals, rsa_vals = build_grid(
            c_min, c_max, c_n,
            rsa_min, rsa_max, rsa_n,
        )
        n_c_pairs   = comb(len(c_vals),   2)
        n_rsa_pairs = comb(len(rsa_vals), 2)
        print(f"\n  [{metric_name}] Grid:")
        print(f"    Centrality candidates : {c_vals.tolist()}")
        print(f"    RSA candidates        : {rsa_vals.tolist()}")
        print(f"    Centrality pairs C({len(c_vals)},2)  : {n_c_pairs}")
        print(f"    RSA pairs      C({len(rsa_vals)},2)  : {n_rsa_pairs}")
        print(f"    Total combos          : {len(grid):,}  "
              f"({n_c_pairs} x {n_rsa_pairs})")
        metric_grids[metric_name] = (grid, c_vals, rsa_vals)

    # -- Stage 2: Small End-to-End LOO Run ---------------------------------
    active_metrics = [m for m, v in metric_grids.items() if v is not None]
    if not active_metrics:
        print("\n  No metrics have bounds set -- skipping Stage 2.")
        print("=" * 60)
        return

    print("\n" + "=" * 60)
    print(f"STAGE 2: End-to-End LOO on {n_proteins} Sampled Proteins")
    print(f"  Active metrics: {active_metrics}")
    print("=" * 60)

    eligible     = df.groupby("pdb_id")["is_functional"].sum().gt(0)
    eligible_ids = eligible[eligible].index.tolist()
    rng          = np.random.default_rng(random_seed)
    sampled      = sorted(rng.choice(
        eligible_ids,
        size    = min(n_proteins, len(eligible_ids)),
        replace = False,
    ).tolist())
    df_small = df[df["pdb_id"].isin(sampled)].copy()

    print(f"\n  Sampled proteins   : {sampled}")
    print(f"  Residues in sample : {len(df_small):,}")

    protein_data, _ = build_protein_data(df_small, null_logger)
    proteins_small  = sorted(protein_data.keys())

    for metric_name in active_metrics:
        grid = metric_grids[metric_name][0]
        print(f"\n  Building MCC matrix [{metric_name}] "
              f"({len(proteins_small)} x {len(grid)})...")
        mcc_matrix = compute_mcc_matrix(
            proteins_small, protein_data, grid, metric_name
        )
        print(f"  Running LOO [{metric_name}]...")
        results = run_loo(
            proteins_small, protein_data, grid,
            mcc_matrix, metric_name, null_logger
        )

        print(f"\n  Per-protein results [{metric_name}]:")
        print(f"  {'pdb_id':<10} {'c_low':>7} {'c_up':>7} "
              f"{'rsa_low':>8} {'rsa_up':>7} "
              f"{'tp':>4} {'fp':>4} {'fn':>4} "
              f"{'sens':>6} {'spec':>6} {'mcc':>6}")
        print("  " + "-" * 79)
        for r in results:
            print(f"  {r['pdb_id']:<10} "
                  f"{r['centrality_lower']:>7.3f} {r['centrality_upper']:>7.3f} "
                  f"{r['rsa_lower']:>8.3f} {r['rsa_upper']:>7.3f} "
                  f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
                  f"{r['sensitivity']:>6.3f} {r['specificity']:>6.3f} "
                  f"{r['mcc']:>6.3f}")

        results_df_small = pd.DataFrame(results)
        print(f"\n  Sample median MCC   : {results_df_small['mcc'].median():.4f}")
        print(f"  Sample mean sens    : {results_df_small['sensitivity'].mean():.4f}")
        print(f"  Sample mean spec    : {results_df_small['specificity'].mean():.4f}")
        if "mcc_catalytic" in results_df_small.columns:
            for ftype in FUNCTIONAL_TYPES:
                col     = f"mcc_{ftype}"
                n_eval  = results_df_small[col].notna().sum()
                med_val = (
                    np.nanmedian(results_df_small[col].dropna())
                    if n_eval > 0 else float("nan")
                )
                print(f"  Nanmedian MCC ({ftype:<10}): "
                      f"{med_val:.4f}  "
                      f"(n evaluable = {n_eval}/{len(results_df_small)})")

    print("\n  Tester complete. Run main() for the full dataset.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    input_csv   = FIGS_DIR / "test_standardized.csv",
    output_dir  = DATA_DIR_3 / "test" / "jackknife_experiment",
    exclude_ids = {},

    # StdCloseness candidate values (used for both lower and upper bound search)
    closeness_min = -0.281,
    closeness_max = 1.764,
    closeness_n   = 10,

    # StdBetweenness candidate values
    betweenness_min = -0.465,
    betweenness_max = 2.237,
    betweenness_n   = 10,

    # RCA_zscore candidate values
    rca_min = -1.015,
    rca_max = 2.123,
    rca_n   = 10,

    # StdPageRank candidate values
    pagerank_min = -0.663,
    pagerank_max = 1.447,
    pagerank_n   = 10,

    # RSA candidate values (used for both lower and upper bound search)
    rsa_min = 0.00,
    rsa_max = 0.90,
    rsa_n   = 10,
):
    input_csv  = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "jackknife_skipped.log"
    logging.basicConfig(
        filename = log_path,
        filemode = "w",
        level    = logging.INFO,
        format   = "%(asctime)s  %(levelname)s  %(message)s",
        datefmt  = "%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("jackknife_experiment")
    logger.info("Jackknife experiment started")
    logger.info(f"Input CSV  : {input_csv}")
    logger.info(f"Output dir : {output_dir}")
    logger.info(
        "NOTE: functional_type and is_functional read directly from "
        "test_standardized.csv -- no BioLiP labels file join."
    )

    print("Loading data...")
    df = load_and_validate(input_csv)
    before = len(df)
    df = df[~df["pdb_id"].isin({p.lower() for p in exclude_ids})]
    print(f"  {len(df):,} residues across {df['pdb_id'].nunique()} proteins "
          f"(excluded {before - len(df):,} rows for {len(exclude_ids)} excluded IDs).")

    ft_counts = df["functional_type"].value_counts()
    print(f"  Functional type breakdown:")
    for ftype in FUNCTIONAL_TYPES + ["non_functional"]:
        print(f"    {ftype:<16} : {ft_counts.get(ftype, 0):,}")
    n_func = df["is_functional"].sum()
    print(f"  Functional residues (total) : {n_func:,} ({100*n_func/len(df):.2f}%)")

    print("Pre-computing per-protein feature arrays...")
    protein_data, skipped = build_protein_data(df, logger)
    proteins = sorted(protein_data.keys())
    n        = len(proteins)
    print(f"  {n} proteins eligible for LOO (skipped {len(skipped)}).")

    metric_configs = [
        ("closeness",   closeness_min,   closeness_max,   closeness_n),
        ("betweenness", betweenness_min, betweenness_max, betweenness_n),
        ("rca",         rca_min,         rca_max,         rca_n),
        ("pagerank",    pagerank_min,    pagerank_max,    pagerank_n),
    ]

    all_results  = {}
    summary_rows = []

    for metric_name, c_min, c_max, c_n in metric_configs:

        if None in (c_min, c_max, c_n):
            print(f"\n[{metric_name}] Skipping -- bounds not set.")
            logger.warning(f"[{metric_name}] Skipped -- bounds not set in main().")
            continue

        print(f"\n{'=' * 60}")
        print(f"Metric: {metric_name.upper()}")
        print(f"{'=' * 60}")
        logger.info(f"Starting LOO for metric: {metric_name}")

        grid, c_vals, rsa_vals = build_grid(
            c_min, c_max, c_n,
            rsa_min, rsa_max, rsa_n,
        )
        print(f"  Centrality candidates : {len(c_vals)}  →  "
              f"C({len(c_vals)},2) = {comb(len(c_vals), 2)} pairs")
        print(f"  RSA candidates        : {len(rsa_vals)}  →  "
              f"C({len(rsa_vals)},2) = {comb(len(rsa_vals), 2)} pairs")
        print(f"  Total grid combos     : {len(grid):,}")
        logger.info(
            f"[{metric_name}] Grid: C({len(c_vals)},2) x C({len(rsa_vals)},2) "
            f"= {len(grid):,} combos"
        )

        print(f"Pre-computing MCC matrix ({n} proteins x {len(grid):,} combos)...")
        mcc_matrix = compute_mcc_matrix(proteins, protein_data, grid, metric_name)
        print("  MCC matrix complete.")

        print("Running LOO selection...")
        results = run_loo(
            proteins, protein_data, grid, mcc_matrix, metric_name, logger
        )

        results_df   = pd.DataFrame(results)
        results_path = output_dir / f"jackknife_results_{metric_name}.csv"
        results_df.to_csv(results_path, index=False)

        _print_summary(results_df, metric_name, results_path, log_path)
        _log_summary(logger, results_df, metric_name)

        summary_rows.append(_build_summary_row(results_df, metric_name))

        all_results[metric_name] = results_df

    if summary_rows:
        summary_df   = pd.DataFrame(summary_rows)
        summary_path = output_dir / "jackknife_summary_all_metrics.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary across all metrics saved to: {summary_path}")
        logger.info(f"Summary across all metrics saved to: {summary_path}")

    logger.info("Jackknife experiment complete.")
    return all_results


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Set centrality candidate ranges (min, max, n) in main() before running.
    # RSA range (rsa_min, rsa_max, rsa_n) has defaults set above.
    main()
