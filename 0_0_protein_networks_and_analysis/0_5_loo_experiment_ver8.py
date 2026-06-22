"""
0_5_loo_experiment_ver8.py
--------------------------
Leave-One-Out (LOO) cross-validation for active site prediction,
extending ver7 to record per-type confusion matrix values
(tp, tn, fp, fn) for each functional type (catalytic, binding, dual).

These per-type counts enable Aloy et al. Z-score calculation
stratified by functional residue type in 0_5_loo_figures.py.

    1. StdCloseness
    2. StdBetweenness
    3. RCA_zscore

Each metric is run as an independent LOO experiment sharing the
same RSA bounds but its own centrality threshold grid. The same
jackknife procedure from Amitai et al. (2004) is applied to all
three.

Prediction rule (all three metrics share this structure):
    A residue is predicted functional if ALL of the following hold:
        <centrality_metric>  >= centrality_lower_bound
        RSA-empirical        >= rsa_lower_bound
        RSA-empirical        <= rsa_upper_bound

Output (per metric, written to output_dir):
    loo_results_closeness_ver8.csv
    loo_results_betweenness_ver8.csv
    loo_results_rca_ver8.csv
    loo_skipped_ver8.log

New columns vs ver7 (per functional type):
    tp_{ftype}, tn_{ftype}, fp_{ftype}, fn_{ftype}

Proteins with no functional residues are skipped and logged.
Undefined overall MCC (edge cases) is recorded as 0.0 and logged.
Undefined stratified MCC (edge cases) is recorded as NaN -- these
proteins are excluded from the nanmedian summary rather than
treated as zero.
"""

import re
import sys
import logging
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from imports import DATA_DIR_2


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FUNCTIONAL_TYPES = ["catalytic", "binding", "dual"]

# Residue token validation patterns
EXPECTED_TOKEN  = re.compile(r'^[A-Z]\d+$')
INSERTION_TOKEN = re.compile(r'^[A-Z]\d+[A-Za-z]+$')

# Mapping from metric name to its column in the CSV
METRIC_COLUMN = {
    "closeness":   "StdCloseness",
    "betweenness": "StdBetweenness",
    "rca":         "RCA_zscore",
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
# Prediction
# ---------------------------------------------------------------------------

def predict(centrality_arr, rsa_arr, c_thresh, rl_thresh, ru_thresh):
    """
    Apply threshold rule to arrays of per-residue features.
    Returns a binary integer array (1 = predicted functional, 0 = not).

    Works for any centrality metric (closeness, betweenness, or RCA z-score)
    by accepting a generic centrality_arr.
    """
    return (
        (centrality_arr >= c_thresh) &
        (rsa_arr        >= rl_thresh) &
        (rsa_arr        <= ru_thresh)
    ).astype(int)


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def build_grid(
    centrality_min,  centrality_max,  centrality_step,
    rsa_lower_min,   rsa_lower_max,   rsa_lower_step,
    rsa_upper_min,   rsa_upper_max,   rsa_upper_step,
):
    """
    Build the full threshold grid and drop invalid combos where
    rsa_lower >= rsa_upper.

    Returns a list of (centrality_lower, rsa_lower, rsa_upper) tuples
    and the three value arrays.
    """
    centrality_vals = np.round(
        np.arange(centrality_min,
                  centrality_max + centrality_step / 2,
                  centrality_step), 4
    )
    rsa_lower_vals = np.round(
        np.arange(rsa_lower_min,
                  rsa_lower_max + rsa_lower_step / 2,
                  rsa_lower_step), 4
    )
    rsa_upper_vals = np.round(
        np.arange(rsa_upper_min,
                  rsa_upper_max + rsa_upper_step / 2,
                  rsa_upper_step), 4
    )
    grid = [
        (c, rl, ru)
        for c, rl, ru in itertools.product(
            centrality_vals, rsa_lower_vals, rsa_upper_vals
        )
        if rl < ru
    ]
    return grid, centrality_vals, rsa_lower_vals, rsa_upper_vals


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_validate(input_csv):
    """
    Load the residue-labeled CSV and validate required columns.

    Required columns:
        pdb_id, is_functional, RSA-empirical,
        StdCloseness, StdBetweenness, RCA_zscore

    Returns a cleaned DataFrame.
    """
    df = pd.read_csv(input_csv)

    required_cols = {
        "pdb_id", "is_functional", "RSA-empirical",
        "StdCloseness", "StdBetweenness", "RCA_zscore",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df["is_functional"]  = df["is_functional"].astype(bool)
    df["StdCloseness"]   = pd.to_numeric(df["StdCloseness"],   errors="coerce")
    df["StdBetweenness"] = pd.to_numeric(df["StdBetweenness"], errors="coerce")
    df["RCA_zscore"]     = pd.to_numeric(df["RCA_zscore"],     errors="coerce")
    df["RSA-empirical"]  = pd.to_numeric(df["RSA-empirical"],  errors="coerce")
    df["pdb_id"]         = df["pdb_id"].str.lower()

    return df


def load_functional_type_labels(detailed_csv, logger):
    """
    Parse biolip_functional_residues_detailed.csv and return a long-form
    DataFrame with one row per (pdb_id, chain, residue_num, functional_type).

    functional_type is one of: 'catalytic', 'binding', 'dual'.

    Residue tokens are validated at load time:
      - Expected format : [single uppercase letter][integer]  e.g. C95
      - Insertion codes : [letter][digits][trailing letter]   e.g. C95A  -> skipped
      - Other unexpected formats are also flagged
    """
    df = pd.read_csv(detailed_csv)
    df["pdb_id"] = df["pdb_id"].str.lower()
    df["chain"]  = df["chain"].str.strip()

    records             = []
    n_insertion         = 0
    n_unexpected        = 0
    insertion_examples  = []
    unexpected_examples = []

    type_cols = {
        "catalytic": "catalytic_residues",
        "binding":   "binding_residues",
        "dual":      "dual_residues",
    }

    for _, row in df.iterrows():
        pdb_id = row["pdb_id"]
        chain  = row["chain"]

        for ftype, col in type_cols.items():
            cell = row.get(col, None)
            if pd.isna(cell) or str(cell).strip() == "":
                continue
            tokens = str(cell).strip().split()
            for token in tokens:
                if EXPECTED_TOKEN.match(token):
                    residue_num = int(token[1:])
                    records.append({
                        "pdb_id":          pdb_id,
                        "chain":           chain,
                        "residue_num":     residue_num,
                        "functional_type": ftype,
                    })
                elif INSERTION_TOKEN.match(token):
                    n_insertion += 1
                    if len(insertion_examples) < 10:
                        insertion_examples.append(f"{pdb_id}:{chain}:{token}")
                else:
                    n_unexpected += 1
                    if len(unexpected_examples) < 10:
                        unexpected_examples.append(f"{pdb_id}:{chain}:{token}")

    if n_insertion > 0:
        logger.warning(
            f"Insertion code tokens found and skipped: {n_insertion}. "
            f"Examples: {insertion_examples}"
        )
    if n_unexpected > 0:
        logger.warning(
            f"Unexpected token formats found and skipped: {n_unexpected}. "
            f"Examples: {unexpected_examples}"
        )

    label_df = pd.DataFrame(records)
    label_df = label_df.drop_duplicates(
        subset=["pdb_id", "chain", "residue_num", "functional_type"]
    )
    logger.info(
        f"Functional type labels loaded: {len(label_df):,} tokens across "
        f"{label_df['pdb_id'].nunique()} proteins."
    )
    return label_df


def add_functional_type(df, label_df):
    """
    Join functional_type labels onto the residue-labeled DataFrame.

    Join key: pdb_id (lowercase) + Chain + Residue number.
    Residues not found in label_df receive functional_type = 'non_functional'.

    Priority when a residue appears under multiple types:
        dual > catalytic > binding

    Returns df with a new 'functional_type' column.
    """
    df = df.copy()
    df["pdb_id_lower"]   = df["pdb_id"].str.lower()
    df["Residue number"] = df["Residue number"].astype(int)

    label_df = label_df.copy()
    label_df["chain"] = label_df["chain"].str.strip()

    type_priority = {"dual": 0, "catalytic": 1, "binding": 2}
    label_df["_priority"] = label_df["functional_type"].map(type_priority)

    label_dedup = (
        label_df
        .sort_values("_priority")
        .drop_duplicates(subset=["pdb_id", "chain", "residue_num"], keep="first")
        .drop(columns=["_priority"])
    )

    merged = df.merge(
        label_dedup,
        left_on  = ["pdb_id_lower", "Chain", "Residue number"],
        right_on = ["pdb_id",       "chain", "residue_num"],
        how      = "left",
    )

    merged["functional_type"] = merged["functional_type"].fillna("non_functional")
    df["functional_type"] = merged["functional_type"].values
    df.drop(columns=["pdb_id_lower"], inplace=True)

    test_ids  = set(df["pdb_id"].str.lower().unique())
    label_ids = set(label_dedup["pdb_id"].str.lower().unique())
    missing_from_labels = test_ids - label_ids
    if missing_from_labels:
        import warnings
        warnings.warn(
            f"{len(missing_from_labels)} test-set protein(s) have no entry in "
            f"the detailed CSV -- all their residues will be labeled non_functional. "
            f"Missing: {sorted(missing_from_labels)}"
        )

    return df


# ---------------------------------------------------------------------------
# Per-protein data builder
# ---------------------------------------------------------------------------

def build_protein_data(df, logger):
    """
    Group residue-level DataFrame by protein and extract feature arrays
    for all three centrality metrics (closeness, betweenness, RCA z-score).

    Proteins with no functional residues are skipped and logged.
    Residues with NaN in any feature column are logged; NaN comparisons
    return False so those residues are always predicted non-functional.

    Stores per-type boolean masks if 'functional_type' column is present.
    Returns protein_data dict and skipped list.
    """
    protein_data = {}
    skipped      = []
    has_types    = "functional_type" in df.columns

    # Log NaN counts per feature column
    for col in ["RSA-empirical", "StdCloseness", "StdBetweenness", "RCA_zscore"]:
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
                entry[f"rsa_{ftype}"]         = grp["RSA-empirical"].values[stratum_mask]

        protein_data[pdb_id] = entry

    return protein_data, skipped


# ---------------------------------------------------------------------------
# MCC matrix and LOO (metric-aware)
# ---------------------------------------------------------------------------

def compute_mcc_matrix(proteins, protein_data, grid, metric):
    """
    Pre-compute MCC for every (protein, threshold combo) pair using the
    overall is_functional label for the given centrality metric.

    Parameters
    ----------
    proteins     : list of pdb_id strings
    protein_data : dict from build_protein_data()
    grid         : list of (centrality_lower, rsa_lower, rsa_upper) tuples
    metric       : one of 'closeness', 'betweenness', 'rca'

    Returns a float32 array of shape (n_proteins, n_combos).
    """
    n          = len(proteins)
    mcc_matrix = np.zeros((n, len(grid)), dtype=np.float32)

    for i, pdb_id in enumerate(proteins):
        d              = protein_data[pdb_id]
        centrality_arr = d[metric]
        for j, (c, rl, ru) in enumerate(grid):
            y_pred           = predict(centrality_arr, d["rsa"], c, rl, ru)
            mcc_matrix[i, j] = compute_mcc(d["y_true"], y_pred)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"    MCC matrix [{metric}]: {i + 1}/{n} proteins done")

    return mcc_matrix


def compute_stratified_confusion(d, y_pred, ftype):
    """
    Compute the confusion matrix for a single functional type stratum.
    y_pred is the full-protein prediction array; this function subsets
    it to the stratum (type residues + non-functional residues).

    Returns a dict with keys tp, tn, fp, fn and mcc, or None if the
    protein has no type-label data. mcc is np.nan if the denominator
    is zero (unevaluable stratum).
    """
    stratum_mask = d.get(f"stratum_{ftype}")
    y_true_strat = d.get(f"y_true_{ftype}")
    if stratum_mask is None or y_true_strat is None:
        return None

    y_pred_strat = y_pred[stratum_mask]

    tp = int(np.sum((y_true_strat == 1) & (y_pred_strat == 1)))
    tn = int(np.sum((y_true_strat == 0) & (y_pred_strat == 0)))
    fp = int(np.sum((y_true_strat == 0) & (y_pred_strat == 1)))
    fn = int(np.sum((y_true_strat == 1) & (y_pred_strat == 0)))
    mcc = float(compute_mcc_nan(y_true_strat, y_pred_strat))

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "mcc": mcc}


def run_loo(proteins, protein_data, grid, mcc_matrix, metric, logger):
    """
    For each held-out protein, select the best threshold combo from the
    remaining n-1 proteins (by median MCC) and score the held-out protein.

    Records overall sensitivity, specificity, MCC, stratified MCC, and
    per-type confusion matrix values (tp, tn, fp, fn) for each functional
    type (catalytic, binding, dual) if type data exists.

    Parameters
    ----------
    proteins     : list of pdb_id strings
    protein_data : dict from build_protein_data()
    grid         : list of (centrality_lower, rsa_lower, rsa_upper) tuples
    mcc_matrix   : float32 array (n_proteins x n_combos)
    metric       : one of 'closeness', 'betweenness', 'rca'
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
        best_c, best_rl, best_ru = grid[best_j]

        d              = protein_data[pdb_id]
        centrality_arr = d[metric]
        y_pred         = predict(centrality_arr, d["rsa"], best_c, best_rl, best_ru)
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
                f"EDGE CASE [{metric}] (MCC undefined, recorded as 0): {pdb_id} | "
                f"tp={tp} tn={tn} fp={fp} fn={fn}"
            )

        result = {
            "pdb_id":            pdb_id,
            "centrality_thresh": best_c,
            "rsa_lower":         best_rl,
            "rsa_upper":         best_ru,
            "tp":                tp,
            "tn":                tn,
            "fp":                fp,
            "fn":                fn,
            "sensitivity":       round(sensitivity, 6),
            "specificity":       round(specificity, 6),
            "mcc":               round(float(mcc), 6),
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
                            f"EDGE CASE [{metric}] (stratified MCC undefined, recorded as NaN): "
                            f"{pdb_id} | stratum={ftype}"
                        )
                    result[f"tp_{ftype}"]  = confusion["tp"]
                    result[f"tn_{ftype}"]  = confusion["tn"]
                    result[f"fp_{ftype}"]  = confusion["fp"]
                    result[f"fn_{ftype}"]  = confusion["fn"]
                    result[f"mcc_{ftype}"] = round(strat_mcc, 6) if not np.isnan(strat_mcc) else np.nan

        results.append(result)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"    LOO [{metric}]: {i + 1}/{n} proteins done")

    return results


# ---------------------------------------------------------------------------
# Shared printing / logging helpers
# ---------------------------------------------------------------------------

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
    input_csv    = DATA_DIR_2 / "loo_exploratory_v2_betweenness_rca" / "residue_labeled.csv",
    detailed_csv = DATA_DIR_2 / "biolip_functional_residues_detailed.csv",
    n_proteins   = 5,
    random_seed  = 42,

    # StdCloseness grid
    closeness_min  = 0.4,
    closeness_max  = 1.5,
    closeness_step = 0.1,

    # StdBetweenness grid  <-- set your bounds here
    betweenness_min  = None,
    betweenness_max  = None,
    betweenness_step = None,

    # RCA_zscore grid  <-- set your bounds here
    rca_min  = None,
    rca_max  = None,
    rca_step = None,

    # Shared RSA grid
    rsa_lower_min  = 0.00,
    rsa_lower_max  = 0.07,
    rsa_lower_step = 0.01,

    rsa_upper_min  = 0.20,
    rsa_upper_max  = 0.55,
    rsa_upper_step = 0.01,
):
    """
    Two-stage tester for all three metrics.

    Stage 1 -- Data & Grid Check
        Loads the full CSV, reports shape, column types, NaN counts,
        functional residue counts, functional type breakdown, and
        confirms the grid dimensions for each metric. Metrics whose
        bounds are None are reported as skipped.

    Stage 2 -- Small End-to-End LOO Run
        Randomly samples n_proteins proteins that have at least one
        functional residue and runs the full LOO procedure (for every
        metric with bounds set) on just those proteins. Prints
        per-protein results including stratified MCC.
    """
    input_csv    = Path(input_csv)
    detailed_csv = Path(detailed_csv)

    null_logger = logging.getLogger("loo_tester_v8")
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
                "RCA_zscore", "RSA-empirical", "is_functional"]:
        print(f"    {col:<22} {df[col].dtype}")

    print(f"\n  NaN counts:")
    for col in ["StdCloseness", "StdBetweenness", "RCA_zscore", "RSA-empirical"]:
        n_nan = df[col].isna().sum()
        print(f"    {col:<22} {n_nan:,}")

    n_functional     = df["is_functional"].sum()
    n_non_functional = (~df["is_functional"]).sum()
    print(f"\n  Functional residues     : {n_functional:,} "
          f"({100 * n_functional / len(df):.1f}%)")
    print(f"  Non-functional residues : {n_non_functional:,} "
          f"({100 * n_non_functional / len(df):.1f}%)")

    proteins_with_func = (
        df.groupby("pdb_id")["is_functional"].sum().gt(0).sum()
    )
    print(f"\n  Proteins with >=1 functional residue : {proteins_with_func}")
    print(f"  Proteins with 0  functional residues : "
          f"{df['pdb_id'].nunique() - proteins_with_func}")

    print(f"\n  Validating residue tokens in: {detailed_csv.name}")
    label_df = load_functional_type_labels(detailed_csv, null_logger)
    print(f"  Functional type label rows   : {len(label_df):,}")
    print(f"  Proteins in label file       : {label_df['pdb_id'].nunique()}")
    type_counts = label_df["functional_type"].value_counts()
    for ftype in FUNCTIONAL_TYPES:
        print(f"    {ftype:<12} : {type_counts.get(ftype, 0):,} residue tokens")

    df = add_functional_type(df, label_df)
    ft_counts = df["functional_type"].value_counts()
    print(f"\n  Functional type breakdown in residue_labeled.csv:")
    for ftype in FUNCTIONAL_TYPES + ["non_functional"]:
        print(f"    {ftype:<16} : {ft_counts.get(ftype, 0):,}")

    # Build and report grids for each metric
    metric_grids = {}
    metric_configs = [
        ("closeness",   closeness_min,   closeness_max,   closeness_step),
        ("betweenness", betweenness_min, betweenness_max, betweenness_step),
        ("rca",         rca_min,         rca_max,         rca_step),
    ]
    for metric_name, c_min, c_max, c_step in metric_configs:
        if None in (c_min, c_max, c_step):
            print(f"\n  [{metric_name}] Grid SKIPPED -- bounds not set.")
            metric_grids[metric_name] = None
            continue
        grid, c_vals, rl_vals, ru_vals = build_grid(
            c_min, c_max, c_step,
            rsa_lower_min, rsa_lower_max, rsa_lower_step,
            rsa_upper_min, rsa_upper_max, rsa_upper_step,
        )
        print(f"\n  [{metric_name}] Grid dimensions:")
        print(f"    Centrality values   : {c_vals.tolist()}")
        print(f"    RSA lower values    : {rl_vals.tolist()}")
        print(f"    RSA upper values    : {ru_vals.tolist()}")
        print(f"    Valid combos        : {len(grid):,}")
        metric_grids[metric_name] = (grid, c_vals, rl_vals, ru_vals)

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

    rng     = np.random.default_rng(random_seed)
    sampled = sorted(rng.choice(
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

        print(f"\n  Per-protein results [{metric_name}] (overall):")
        print(f"  {'pdb_id':<10} {'c_thresh':>9} {'rsa_low':>8} {'rsa_up':>8} "
              f"{'tp':>4} {'fp':>4} {'fn':>4} {'sens':>6} {'spec':>6} {'mcc':>6}")
        print("  " + "-" * 73)
        for r in results:
            print(f"  {r['pdb_id']:<10} {r['centrality_thresh']:>9.4f} "
                  f"{r['rsa_lower']:>8.2f} {r['rsa_upper']:>8.2f} "
                  f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
                  f"{r['sensitivity']:>6.3f} {r['specificity']:>6.3f} "
                  f"{r['mcc']:>6.3f}")

        results_df_small = pd.DataFrame(results)
        print(f"\n  Sample median MCC   : {results_df_small['mcc'].median():.4f}")
        print(f"  Sample mean sens    : {results_df_small['sensitivity'].mean():.4f}")
        print(f"  Sample mean spec    : {results_df_small['specificity'].mean():.4f}")
        if "mcc_catalytic" in results_df_small.columns:
            for ftype in FUNCTIONAL_TYPES:
                col    = f"mcc_{ftype}"
                n_eval = results_df_small[col].notna().sum()
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
    input_csv    = DATA_DIR_2 / "loo_exploratory_v2_betweenness_rca" / "residue_labeled.csv",
    detailed_csv = DATA_DIR_2 / "biolip_functional_residues_detailed.csv",
    output_dir   = DATA_DIR_2 / "loo_exploratory_ver8",

    # StdCloseness grid
    closeness_min  = 0.4,
    closeness_max  = 1.5,
    closeness_step = 0.1,

    # # StdBetweenness grid
    # betweenness_min  = 0.74,
    # betweenness_max  = 4.5,
    # betweenness_step = 0.1,

    # # RCA_zscore grid
    # rca_min  = 0.5,
    # rca_max  = 2.5,
    # rca_step = 0.1,

    # StdBetweenness grid
    betweenness_min  = None,
    betweenness_max  = None,
    betweenness_step = None,

    # RCA_zscore grid
    rca_min  = None,
    rca_max  = None,
    rca_step = None,

    # Shared RSA grid
    rsa_lower_min  = 0.00,
    rsa_lower_max  = 0.007,
    rsa_lower_step = 0.001,

    rsa_upper_min  = 0.50,
    rsa_upper_max  = 0.84,
    rsa_upper_step = 0.01,
):
    input_csv    = Path(input_csv)
    detailed_csv = Path(detailed_csv)
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- logging -----------------------------------------------------------
    log_path = output_dir / "loo_skipped_ver8.log"
    logging.basicConfig(
        filename = log_path,
        filemode = "w",
        level    = logging.INFO,
        format   = "%(asctime)s  %(levelname)s  %(message)s",
        datefmt  = "%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("loo_v8")
    logger.info("LOO experiment ver8 started")
    logger.info(f"Input CSV    : {input_csv}")
    logger.info(f"Detailed CSV : {detailed_csv}")
    logger.info(f"Output dir   : {output_dir}")

    # -- load and join functional type labels ------------------------------
    print("Loading data...")
    df = load_and_validate(input_csv)
    print(f"  {len(df):,} residues across {df['pdb_id'].nunique()} proteins.")

    print("Loading functional type labels...")
    label_df = load_functional_type_labels(detailed_csv, logger)
    df       = add_functional_type(df, label_df)
    ft_counts = df["functional_type"].value_counts()
    print(f"  Functional type breakdown:")
    for ftype in FUNCTIONAL_TYPES + ["non_functional"]:
        print(f"    {ftype:<16} : {ft_counts.get(ftype, 0):,}")

    # -- build protein data (shared across all metrics) --------------------
    print("Pre-computing per-protein feature arrays...")
    protein_data, skipped = build_protein_data(df, logger)
    proteins = sorted(protein_data.keys())
    n        = len(proteins)
    print(f"  {n} proteins eligible for LOO (skipped {len(skipped)}).")

    # -- define metric configs ---------------------------------------------
    metric_configs = [
        ("closeness",   closeness_min,   closeness_max,   closeness_step),
        ("betweenness", betweenness_min, betweenness_max, betweenness_step),
        ("rca",         rca_min,         rca_max,         rca_step),
    ]

    all_results = {}

    for metric_name, c_min, c_max, c_step in metric_configs:

        if None in (c_min, c_max, c_step):
            print(f"\n[{metric_name}] Skipping -- bounds not set.")
            logger.warning(f"[{metric_name}] Skipped -- bounds not set in main().")
            continue

        print(f"\n{'=' * 60}")
        print(f"Metric: {metric_name.upper()}")
        print(f"{'=' * 60}")
        logger.info(f"Starting LOO for metric: {metric_name}")

        # -- build grid ----------------------------------------------------
        grid, c_vals, rl_vals, ru_vals = build_grid(
            c_min, c_max, c_step,
            rsa_lower_min, rsa_lower_max, rsa_lower_step,
            rsa_upper_min, rsa_upper_max, rsa_upper_step,
        )
        print(f"  Threshold grid: {len(c_vals)} x {len(rl_vals)} x {len(ru_vals)} "
              f"= {len(grid)} valid combinations.")
        logger.info(
            f"[{metric_name}] Grid: {len(c_vals)} centrality vals x "
            f"{len(rl_vals)} RSA-lower vals x {len(ru_vals)} RSA-upper vals "
            f"= {len(grid)} combos"
        )

        # -- MCC matrix ----------------------------------------------------
        print(f"Pre-computing MCC matrix ({n} proteins x {len(grid)} combos)...")
        mcc_matrix = compute_mcc_matrix(proteins, protein_data, grid, metric_name)
        print("  MCC matrix complete.")

        # -- LOO -----------------------------------------------------------
        print("Running LOO selection...")
        results = run_loo(
            proteins, protein_data, grid, mcc_matrix, metric_name, logger
        )

        # -- save results --------------------------------------------------
        results_df   = pd.DataFrame(results)
        results_path = output_dir / f"loo_results_{metric_name}_ver8.csv"
        results_df.to_csv(results_path, index=False)

        _print_summary(results_df, metric_name, results_path, log_path)
        _log_summary(logger, results_df, metric_name)

        all_results[metric_name] = results_df

    logger.info("LOO experiment ver8 complete.")
    return all_results


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Fill in grid bounds in main() before running.
    main()
