"""
0_5_loo_experiment.py
---------------------
Leave-One-Out (LOO) cross-validation for active site prediction,
replicating the jackknife procedure from Amitai et al. (2004) with
modified threshold grids suited to this dataset.

Prediction rule:
    A residue is predicted functional if ALL of the following hold:
        StdCloseness   >= closeness_lower_bound
        RSA-empirical  >= rsa_lower_bound
        RSA-empirical  <= rsa_upper_bound

Threshold grid:
    StdCloseness lower : 0.4 -> 1.5,  step 0.1  (12 values)
    RSA lower          : 0.00 -> 0.07, step 0.01  (8 values)
    RSA upper          : 0.50 -> 0.84, step 0.01 (35 values)
    Total combinations : 12 x 8 x 35 = 3,360

For each held-out protein:
    1. Compute median MCC across all other proteins for every threshold combo
    2. Select the combo with the highest median MCC
    3. Apply that combo to the held-out protein
    4. Record overall sensitivity, specificity, MCC
    5. Record stratified MCC for each functional type:
       catalytic, binding, dual (each vs all non-functional residues)

Functional type labels are joined from biolip_functional_residues_detailed.csv.
Residue tokens are validated at load time; unexpected formats are logged.

Proteins with no functional residues are skipped and logged.
Undefined overall MCC (edge cases) is recorded as 0.0 and logged.
Undefined stratified MCC (edge cases) is recorded as NaN — these proteins
are excluded from the nanmedian summary rather than treated as zero.

Output:
    loo_results.csv   — per-protein results including stratified MCC
    loo_skipped.log   — skipped / edge-case proteins and token warnings
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
# Module-level helpers
# ---------------------------------------------------------------------------

FUNCTIONAL_TYPES = ["catalytic", "binding", "dual"]

# expected residue token: single uppercase letter followed by one or more digits
EXPECTED_TOKEN  = re.compile(r'^[A-Z]\d+$')
# insertion code pattern: single uppercase letter, digits, then a trailing letter
INSERTION_TOKEN = re.compile(r'^[A-Z]\d+[A-Za-z]+$')


def compute_mcc(y_true, y_pred):
    """
    Matthews Correlation Coefficient.
    Returns 0.0 for undefined cases (zero denominator).
    Used for overall MCC and for the MCC matrix (training fold optimisation).
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


def predict(closeness_arr, rsa_arr, c_thresh, rl_thresh, ru_thresh):
    """
    Apply threshold rule to arrays of per-residue features.
    Returns a binary integer array (1 = predicted functional, 0 = not).
    """
    return (
        (closeness_arr >= c_thresh) &
        (rsa_arr       >= rl_thresh) &
        (rsa_arr       <= ru_thresh)
    ).astype(int)


def build_grid(closeness_min, closeness_max, closeness_step,
               rsa_lower_min, rsa_lower_max, rsa_lower_step,
               rsa_upper_min, rsa_upper_max, rsa_upper_step):
    """
    Build the full threshold grid and drop invalid combos where
    rsa_lower >= rsa_upper.
    Returns a list of (closeness_lower, rsa_lower, rsa_upper) tuples.
    """
    closeness_vals = np.round(
        np.arange(closeness_min, closeness_max + closeness_step / 2, closeness_step), 4
    )
    rsa_lower_vals = np.round(
        np.arange(rsa_lower_min, rsa_lower_max + rsa_lower_step / 2, rsa_lower_step), 4
    )
    rsa_upper_vals = np.round(
        np.arange(rsa_upper_min, rsa_upper_max + rsa_upper_step / 2, rsa_upper_step), 4
    )
    grid = [
        (c, rl, ru)
        for c, rl, ru in itertools.product(closeness_vals, rsa_lower_vals, rsa_upper_vals)
        if rl < ru
    ]
    return grid, closeness_vals, rsa_lower_vals, rsa_upper_vals


def load_and_validate(input_csv):
    """
    Load the residue-labeled CSV and validate required columns.
    Returns a cleaned DataFrame.
    """
    df = pd.read_csv(input_csv)

    required_cols = {"pdb_id", "StdCloseness", "RSA-empirical", "is_functional"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df["is_functional"] = df["is_functional"].astype(bool)
    df["StdCloseness"]  = pd.to_numeric(df["StdCloseness"],  errors="coerce")
    df["RSA-empirical"] = pd.to_numeric(df["RSA-empirical"], errors="coerce")
    df["pdb_id"]        = df["pdb_id"].str.lower()

    return df


def load_functional_type_labels(detailed_csv, logger):
    """
    Parse biolip_functional_residues_detailed.csv and return a long-form
    DataFrame with one row per (pdb_id, chain, residue_num, functional_type).

    functional_type is one of: 'catalytic', 'binding', 'dual'.

    Residue tokens are validated at load time:
      - Expected format : [single uppercase letter][integer]  e.g. C95
      - Insertion codes : [letter][digits][trailing letter]   e.g. C95A
      - Other unexpected formats are also flagged

    Insertion code tokens are skipped and logged. The residue number is
    extracted by stripping the leading amino acid letter.
    """
    df = pd.read_csv(detailed_csv)
    df["pdb_id"] = df["pdb_id"].str.lower()
    df["chain"]  = df["chain"].str.strip()

    records            = []
    n_insertion        = 0
    n_unexpected       = 0
    insertion_examples = []
    unexpected_examples= []

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
                        "pdb_id":         pdb_id,
                        "chain":          chain,
                        "residue_num":    residue_num,
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
    # drop duplicates in case the same residue appears in multiple BioLiP rows
    label_df = label_df.drop_duplicates(subset=["pdb_id", "chain", "residue_num",
                                                  "functional_type"])
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

    A residue may appear under multiple functional types in label_df
    (e.g. once as 'catalytic' and once as 'binding' across different
    BioLiP entries). In that case, the highest-priority type is kept:
        dual > catalytic > binding

    Returns df with a new 'functional_type' column.
    """
    df = df.copy()
    df["pdb_id_lower"]   = df["pdb_id"].str.lower()
    df["Residue number"] = df["Residue number"].astype(int)

    label_df = label_df.copy()
    label_df["chain"] = label_df["chain"].str.strip()

    # assign priority so that when duplicates exist we keep the highest
    # dual=0 (highest), catalytic=1, binding=2
    type_priority = {"dual": 0, "catalytic": 1, "binding": 2}
    label_df["_priority"] = label_df["functional_type"].map(type_priority)

    # deduplicate: keep the highest-priority type per (pdb_id, chain, residue_num)
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

    # cross-check: warn if any test-set proteins are missing from label_df
    test_ids  = set(df["pdb_id"].str.lower().unique())
    label_ids = set(label_dedup["pdb_id"].str.lower().unique())
    missing_from_labels = test_ids - label_ids
    if missing_from_labels:
        import warnings
        warnings.warn(
            f"{len(missing_from_labels)} test-set protein(s) have no entry in "
            f"the detailed CSV — all their residues will be labeled non_functional. "
            f"Missing: {sorted(missing_from_labels)}"
        )

    return df


def build_protein_data(df, logger):
    """
    Group residue-level DataFrame by protein and extract feature arrays.
    Proteins with no functional residues are skipped and logged.
    Residues with NaN in RSA-empirical are logged as warnings — they will
    always be predicted non-functional (NaN comparisons return False).

    Stores per-type boolean masks if 'functional_type' column is present.
    Returns protein_data dict and skipped list.
    """
    protein_data = {}
    skipped      = []
    has_types    = "functional_type" in df.columns

    # log any NaN RSA residues
    nan_rsa = df[df["RSA-empirical"].isna()]
    if not nan_rsa.empty:
        per_protein_nan = nan_rsa.groupby("pdb_id").size()
        logger.warning(
            f"NaN RSA-empirical found in {len(per_protein_nan)} protein(s), "
            f"{len(nan_rsa)} residues total. These residues will always be "
            f"predicted non-functional."
        )
        for pdb_id, count in per_protein_nan.items():
            func_nan = nan_rsa[nan_rsa["pdb_id"] == pdb_id]["is_functional"].sum()
            logger.warning(
                f"  NaN RSA: {pdb_id} — {count} residues "
                f"({func_nan} functional, {count - func_nan} non-functional)"
            )

    for pdb_id, grp in df.groupby("pdb_id"):
        y_true = grp["is_functional"].values.astype(int)
        if y_true.sum() == 0:
            logger.info(f"SKIP (no functional residues): {pdb_id}")
            skipped.append((pdb_id, "no_functional_residues"))
            continue

        entry = {
            "closeness": grp["StdCloseness"].values,
            "rsa":       grp["RSA-empirical"].values,
            "y_true":    y_true,
        }

        # store per-type masks against non-functional residues
        if has_types:
            non_func_mask = grp["functional_type"].values == "non_functional"
            for ftype in FUNCTIONAL_TYPES:
                type_mask = grp["functional_type"].values == ftype
                # y_true for this stratum: type residues = 1, non-functional = 0
                # only include rows that are either this type or non-functional
                stratum_mask = type_mask | non_func_mask
                entry[f"y_true_{ftype}"]    = type_mask[stratum_mask].astype(int)
                entry[f"stratum_{ftype}"]   = stratum_mask
                entry[f"closeness_{ftype}"] = grp["StdCloseness"].values[stratum_mask]
                entry[f"rsa_{ftype}"]       = grp["RSA-empirical"].values[stratum_mask]

        protein_data[pdb_id] = entry

    return protein_data, skipped


def compute_mcc_matrix(proteins, protein_data, grid):
    """
    Pre-compute MCC for every (protein, threshold combo) pair
    using the overall is_functional label.
    Returns a float32 array of shape (n_proteins, n_combos).
    """
    n          = len(proteins)
    mcc_matrix = np.zeros((n, len(grid)), dtype=np.float32)

    for i, pdb_id in enumerate(proteins):
        d = protein_data[pdb_id]
        for j, (c, rl, ru) in enumerate(grid):
            y_pred           = predict(d["closeness"], d["rsa"], c, rl, ru)
            mcc_matrix[i, j] = compute_mcc(d["y_true"], y_pred)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"    MCC matrix: {i + 1}/{n} proteins done")

    return mcc_matrix


def compute_stratified_mcc(d, y_pred, ftype):
    """
    Compute MCC for a single functional type stratum.
    y_pred is the full-protein prediction array; this function
    subsets it to the stratum (type residues + non-functional residues).

    Returns MCC as float, np.nan if undefined (zero denominator), or
    None if the protein has no type-label data (functional_type column absent).
    Using np.nan for undefined strata allows nanmedian summaries to exclude
    unevaluable proteins rather than treating them as zero.
    """
    stratum_mask  = d.get(f"stratum_{ftype}")
    y_true_strat  = d.get(f"y_true_{ftype}")
    if stratum_mask is None or y_true_strat is None:
        return None
    y_pred_strat = y_pred[stratum_mask]
    return float(compute_mcc_nan(y_true_strat, y_pred_strat))


def run_loo(proteins, protein_data, grid, mcc_matrix, logger):
    """
    For each held-out protein, select the best threshold combo from the
    remaining n-1 proteins (by median MCC) and score the held-out protein.

    Records overall sensitivity, specificity, MCC, and stratified MCC
    for each functional type (catalytic, binding, dual) if type data exists.

    Returns a list of result dicts.
    """
    results  = []
    n        = len(proteins)
    has_types = f"y_true_{FUNCTIONAL_TYPES[0]}" in protein_data[proteins[0]]

    for i, pdb_id in enumerate(proteins):
        training_rows    = np.delete(mcc_matrix, i, axis=0)
        median_mccs      = np.median(training_rows, axis=0)
        best_j           = int(np.argmax(median_mccs))
        best_c, best_rl, best_ru = grid[best_j]

        d      = protein_data[pdb_id]
        y_pred = predict(d["closeness"], d["rsa"], best_c, best_rl, best_ru)
        y_true = d["y_true"]

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
                f"EDGE CASE (MCC undefined, recorded as 0): {pdb_id} | "
                f"tp={tp} tn={tn} fp={fp} fn={fn}"
            )

        result = {
            "pdb_id":           pdb_id,
            "closeness_thresh": best_c,
            "rsa_lower":        best_rl,
            "rsa_upper":        best_ru,
            "tp":               tp,
            "tn":               tn,
            "fp":               fp,
            "fn":               fn,
            "sensitivity":      round(sensitivity, 6),
            "specificity":      round(specificity, 6),
            "mcc":              round(float(mcc), 6),
        }

        # stratified MCC per functional type
        if has_types:
            for ftype in FUNCTIONAL_TYPES:
                strat_mcc = compute_stratified_mcc(d, y_pred, ftype)
                if strat_mcc is None:
                    result[f"mcc_{ftype}"] = None
                elif np.isnan(strat_mcc):
                    logger.info(
                        f"EDGE CASE (stratified MCC undefined, recorded as NaN): "
                        f"{pdb_id} | stratum={ftype}"
                    )
                    result[f"mcc_{ftype}"] = np.nan
                else:
                    result[f"mcc_{ftype}"] = round(strat_mcc, 6)

        results.append(result)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"    LOO: {i + 1}/{n} proteins done")

    return results


# ---------------------------------------------------------------------------
# Tester
# ---------------------------------------------------------------------------

def run_tester(
    input_csv        = DATA_DIR_2 / "loo_exploratory" / "residue_labeled.csv",
    detailed_csv     = DATA_DIR_2 / "biolip_functional_residues_detailed.csv",
    n_proteins       = 5,
    random_seed      = 42,

    closeness_min    = 0.4,
    closeness_max    = 1.5,
    closeness_step   = 0.1,

    rsa_lower_min    = 0.00,
    rsa_lower_max    = 0.07,
    rsa_lower_step   = 0.01,

    rsa_upper_min    = 0.50,
    rsa_upper_max    = 0.84,
    rsa_upper_step   = 0.01,
):
    """
    Two-stage tester:

    Stage 1 — Data & Grid Check
        Loads the full CSV, reports shape, column types, NaN counts,
        functional residue counts, functional type breakdown, and
        confirms the grid dimensions. Also validates residue tokens
        in the detailed CSV and reports any insertion codes or
        unexpected formats.

    Stage 2 — Small End-to-End LOO Run
        Randomly samples n_proteins proteins that have at least one
        functional residue and runs the full LOO procedure on just
        those proteins. Prints per-protein results including stratified
        MCC so you can see what the output looks like before running
        the full dataset.
    """
    input_csv    = Path(input_csv)
    detailed_csv = Path(detailed_csv)

    null_logger = logging.getLogger("loo_tester")
    null_logger.addHandler(logging.NullHandler())

    # ── Stage 1: Data & Grid Check ────────────────────────────────────────
    print("=" * 60)
    print("STAGE 1: Data & Grid Check")
    print("=" * 60)

    df = load_and_validate(input_csv)

    print(f"\n  CSV path        : {input_csv}")
    print(f"  Total rows      : {len(df):,}")
    print(f"  Total proteins  : {df['pdb_id'].nunique()}")
    print(f"\n  Column dtypes:")
    for col in ["pdb_id", "StdCloseness", "RSA-empirical", "is_functional"]:
        print(f"    {col:<20} {df[col].dtype}")

    print(f"\n  NaN counts:")
    for col in ["StdCloseness", "RSA-empirical"]:
        n_nan = df[col].isna().sum()
        print(f"    {col:<20} {n_nan:,}")

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

    # validate detailed CSV tokens
    print(f"\n  Validating residue tokens in: {detailed_csv.name}")
    label_df = load_functional_type_labels(detailed_csv, null_logger)
    print(f"  Functional type label rows   : {len(label_df):,}")
    print(f"  Proteins in label file       : {label_df['pdb_id'].nunique()}")
    type_counts = label_df["functional_type"].value_counts()
    for ftype in FUNCTIONAL_TYPES:
        print(f"    {ftype:<12} : {type_counts.get(ftype, 0):,} residue tokens")

    # join types onto main df
    df = add_functional_type(df, label_df)
    ft_counts = df["functional_type"].value_counts()
    print(f"\n  Functional type breakdown in residue_labeled.csv:")
    for ftype in FUNCTIONAL_TYPES + ["non_functional"]:
        print(f"    {ftype:<16} : {ft_counts.get(ftype, 0):,}")

    grid, c_vals, rl_vals, ru_vals = build_grid(
        closeness_min, closeness_max, closeness_step,
        rsa_lower_min, rsa_lower_max, rsa_lower_step,
        rsa_upper_min, rsa_upper_max, rsa_upper_step,
    )
    print(f"\n  Grid dimensions:")
    print(f"    StdCloseness values : {c_vals.tolist()}")
    print(f"    RSA lower values    : {rl_vals.tolist()}")
    print(f"    RSA upper values    : {ru_vals.tolist()}")
    print(f"    Valid combos        : {len(grid):,}")

    # ── Stage 2: Small End-to-End LOO Run ────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STAGE 2: End-to-End LOO on {n_proteins} Sampled Proteins")
    print("=" * 60)

    eligible     = df.groupby("pdb_id")["is_functional"].sum().gt(0)
    eligible_ids = eligible[eligible].index.tolist()

    rng      = np.random.default_rng(random_seed)
    sampled  = sorted(rng.choice(eligible_ids,
                                  size    = min(n_proteins, len(eligible_ids)),
                                  replace = False).tolist())
    df_small = df[df["pdb_id"].isin(sampled)].copy()

    print(f"\n  Sampled proteins   : {sampled}")
    print(f"  Residues in sample : {len(df_small):,}")

    protein_data, skipped = build_protein_data(df_small, null_logger)
    proteins_small        = sorted(protein_data.keys())

    print(f"\n  Building MCC matrix ({len(proteins_small)} x {len(grid)})...")
    mcc_matrix_small = compute_mcc_matrix(proteins_small, protein_data, grid)

    print("\n  Running LOO...")
    results_small = run_loo(proteins_small, protein_data, grid,
                            mcc_matrix_small, null_logger)

    print("\n  Per-protein results (overall):")
    print(f"  {'pdb_id':<10} {'c_thresh':>8} {'rsa_low':>8} {'rsa_up':>8} "
          f"{'tp':>4} {'fp':>4} {'fn':>4} {'sens':>6} {'spec':>6} {'mcc':>6}")
    print("  " + "-" * 72)
    for r in results_small:
        print(f"  {r['pdb_id']:<10} {r['closeness_thresh']:>8.2f} "
              f"{r['rsa_lower']:>8.2f} {r['rsa_upper']:>8.2f} "
              f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
              f"{r['sensitivity']:>6.3f} {r['specificity']:>6.3f} {r['mcc']:>6.3f}")

    print("\n  Per-protein results (stratified MCC):")
    print(f"  {'pdb_id':<10} {'mcc_catalytic':>14} {'mcc_binding':>12} {'mcc_dual':>10}")
    print("  " + "-" * 50)
    for r in results_small:
        cat  = f"{r.get('mcc_catalytic', 'N/A'):>14}" if r.get('mcc_catalytic') is not None else f"{'N/A':>14}"
        bind = f"{r.get('mcc_binding',   'N/A'):>12}" if r.get('mcc_binding')   is not None else f"{'N/A':>12}"
        dual = f"{r.get('mcc_dual',      'N/A'):>10}" if r.get('mcc_dual')      is not None else f"{'N/A':>10}"
        print(f"  {r['pdb_id']:<10} {cat} {bind} {dual}")

    results_df_small = pd.DataFrame(results_small)
    print(f"\n  Sample median MCC              : {results_df_small['mcc'].median():.4f}")
    print(f"  Sample mean sensitivity        : {results_df_small['sensitivity'].mean():.4f}")
    print(f"  Sample mean specificity        : {results_df_small['specificity'].mean():.4f}")
    if "mcc_catalytic" in results_df_small.columns:
        for ftype in FUNCTIONAL_TYPES:
            col       = f"mcc_{ftype}"
            n_eval    = results_df_small[col].notna().sum()
            med_val   = np.nanmedian(results_df_small[col].dropna()) if n_eval > 0 else float("nan")
            print(f"  Sample nanmedian MCC ({ftype:<10}): "
                  f"{med_val:.4f}  (n evaluable = {n_eval}/{len(results_df_small)})")
    print("\n  Tester complete. Run main() for the full dataset.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    input_csv      = DATA_DIR_2 / "loo_exploratory" / "residue_labeled.csv",
    detailed_csv   = DATA_DIR_2 / "biolip_functional_residues_detailed.csv",
    output_dir     = DATA_DIR_2 / "loo_exploratory_ver6" / "loo_results_ver6",

    closeness_min  = 0.4,
    closeness_max  = 1.5,
    closeness_step = 0.1,

    rsa_lower_min  = 0.00,
    rsa_lower_max  = 0.07,
    rsa_lower_step = 0.01,

    rsa_upper_min  = 0.50,
    rsa_upper_max  = 0.84,
    rsa_upper_step = 0.01,
):
    input_csv    = Path(input_csv)
    detailed_csv = Path(detailed_csv)
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── logging ───────────────────────────────────────────────────────────
    log_path = output_dir / "loo_skipped_ver6.log"
    logging.basicConfig(
        filename = log_path,
        filemode = "w",
        level    = logging.INFO,
        format   = "%(asctime)s  %(levelname)s  %(message)s",
        datefmt  = "%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("loo")
    logger.info("LOO experiment started")
    logger.info(f"Input CSV    : {input_csv}")
    logger.info(f"Detailed CSV : {detailed_csv}")
    logger.info(f"Output dir   : {output_dir}")

    # ── load and join functional type labels ──────────────────────────────
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

    # ── build grid ────────────────────────────────────────────────────────
    grid, c_vals, rl_vals, ru_vals = build_grid(
        closeness_min, closeness_max, closeness_step,
        rsa_lower_min, rsa_lower_max, rsa_lower_step,
        rsa_upper_min, rsa_upper_max, rsa_upper_step,
    )
    print(f"  Threshold grid: {len(c_vals)} x {len(rl_vals)} x {len(ru_vals)} "
          f"= {len(grid)} valid combinations.")

    # ── build protein data ────────────────────────────────────────────────
    print("Pre-computing per-protein feature arrays...")
    protein_data, skipped = build_protein_data(df, logger)
    proteins = sorted(protein_data.keys())
    n        = len(proteins)
    print(f"  {n} proteins eligible for LOO (skipped {len(skipped)}).")

    # ── MCC matrix ────────────────────────────────────────────────────────
    print(f"\nPre-computing MCC matrix ({n} proteins x {len(grid)} combos)...")
    mcc_matrix = compute_mcc_matrix(proteins, protein_data, grid)
    print("  MCC matrix complete.")

    # ── LOO ───────────────────────────────────────────────────────────────
    print("Running LOO selection...")
    results = run_loo(proteins, protein_data, grid, mcc_matrix, logger)

    # ── save results ──────────────────────────────────────────────────────
    results_df   = pd.DataFrame(results)
    results_path = output_dir / "loo_results_ver6.csv"
    results_df.to_csv(results_path, index=False)

    print("\n── LOO Summary ──────────────────────────────────────────────")
    print(f"  Proteins evaluated          : {len(results_df)}")
    print(f"  Proteins skipped            : {len(skipped)}")
    print(f"  Median MCC (overall)        : {results_df['mcc'].median():.4f}")
    print(f"  Mean sensitivity            : {results_df['sensitivity'].mean():.4f}")
    print(f"  Mean specificity            : {results_df['specificity'].mean():.4f}")
    if "mcc_catalytic" in results_df.columns:
        for ftype in FUNCTIONAL_TYPES:
            col     = f"mcc_{ftype}"
            n_eval  = results_df[col].notna().sum()
            med_val = np.nanmedian(results_df[col].dropna()) if n_eval > 0 else float("nan")
            print(f"  Nanmedian MCC ({ftype:<12})  : "
                  f"{med_val:.4f}  (n evaluable = {n_eval}/{len(results_df)})")
    print(f"\n  Results saved to            : {results_path}")
    print(f"  Log saved to                : {log_path}")
    print("─────────────────────────────────────────────────────────────")

    logger.info(f"LOO complete. {len(results_df)} proteins evaluated, "
                f"{len(skipped)} skipped.")
    logger.info(f"Median MCC (overall): {results_df['mcc'].median():.4f}")
    if "mcc_catalytic" in results_df.columns:
        for ftype in FUNCTIONAL_TYPES:
            col     = f"mcc_{ftype}"
            n_eval  = results_df[col].notna().sum()
            med_val = np.nanmedian(results_df[col].dropna()) if n_eval > 0 else float("nan")
            logger.info(f"Nanmedian MCC ({ftype}): {med_val:.4f}  "
                        f"(n evaluable = {n_eval}/{len(results_df)})")

    return results_df


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_tester()
    #5/26/26
    main()
