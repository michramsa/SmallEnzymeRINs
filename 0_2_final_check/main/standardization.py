# standardization.py
#
# Purpose: Centrality standardization functions for use in the fig1 recreation
# pipeline and any downstream analyses that require standardized centrality values.
#
# All functions follow the Amitai et al. (2004) Methods conventions unless noted.
#
# Standardization methods:
#
#   CLOSENESS (StdCloseness):
#       z-scored per protein.
#       std_closeness(r) = (closeness_r - protein_mean) / protein_std
#
#   DEGREE (StdDegree):
#       z-scored per amino acid type, using reference population statistics.
#       std_degree(r) = (raw_degree_r - aa_type_mean) / aa_type_std
#       where mean/std are computed across ALL residues of that amino acid type
#       in the reference (PISCES) set.
#       NOTE: uses raw integer Degree counts (not normalized DegreeCentrality).
#
#   BETWEENNESS (StdBetweenness):
#       z-scored per protein (same approach as closeness).
#       std_betweenness(r) = (betweenness_r - protein_mean) / protein_std
#
#   RCA (RCA_zscore):
#       NOT re-standardized here. RCA_zscore is computed upstream by
#       append_rca_to_centrality() in parallel_RIN_analysis.py and written
#       directly into each _centrality.csv. It is loaded and used as-is.
#
#   PAGERANK (StdPageRank):
#       append_pagerank_to_centrality() in parallel_RIN_analysis.py already
#       computes a per-protein z-score (ddof=0) and writes it to each
#       _centrality.csv as PageRank_zscore. That column is simply renamed to
#       StdPageRank here for consistency with the other Std* columns.
#       No recomputation is performed.
#
# Usage example (mirrors 0_4_fig1_recreation_ver8.py / main_size_controlled):
#
#   from standardization import (
#       standardize_closeness_per_protein,
#       compute_degree_stats_by_aa,
#       compute_degree_stats_from_ref_dir,
#       standardize_degree_by_aa,
#       standardize_betweenness_per_protein,
#       rename_pagerank_zscore,
#       assign_rsa_category,
#       standardize_all,
#   )
#
#   # Option A — from an already-loaded reference DataFrame:
#   aa_stats = compute_degree_stats_by_aa(ref_df)
#   ref_df   = standardize_all(ref_df, aa_stats)
#
#   # Option B — directly from the reference centrality directory (no pre-load needed):
#   aa_stats = compute_degree_stats_from_ref_dir(ref_centrality_dir)
#   test_df  = standardize_all(test_df, aa_stats)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *
import numpy as np


# ── closeness ─────────────────────────────────────────────────────────────────

def standardize_closeness_per_protein(df: pd.DataFrame) -> pd.DataFrame:
    """
    Amitai method for closeness: z-score within each protein individually.
    Adds StdCloseness column.
    Proteins with std == 0 (degenerate case) get StdCloseness = 0.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'ClosenessCentrality' and 'pdb_id' columns.

    Returns
    -------
    pd.DataFrame with 'StdCloseness' column added.
    """
    df = df.copy()

    def _zscore(grp):
        mu  = grp["ClosenessCentrality"].mean()
        sig = grp["ClosenessCentrality"].std(ddof=0)
        grp = grp.copy()
        grp["StdCloseness"] = 0.0 if sig == 0 else (grp["ClosenessCentrality"] - mu) / sig
        return grp

    return df.groupby("pdb_id", group_keys=False).apply(_zscore)


# ── degree ────────────────────────────────────────────────────────────────────

def compute_degree_stats_by_aa(ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    Amitai method for degree: compute mean and std of raw Degree
    grouped by amino acid type across the entire reference population.

    This must be called on the reference set before standardizing either the
    reference set or the test set by amino acid type.

    Parameters
    ----------
    ref_df : pd.DataFrame
        Reference population. Must contain 'Residue' and 'Degree' columns.
        Should already be filtered to canonical amino acids only.

    Returns
    -------
    pd.DataFrame with columns: Residue, deg_mean, deg_std.
    """
    aa_stats = (
        ref_df.groupby("Residue")["Degree"]
        .agg(deg_mean="mean", deg_std=lambda x: x.std(ddof=0))
        .reset_index()
    )

    print("\n── Per-Amino-Acid-Type Degree Statistics (reference set) ────")
    print(aa_stats.to_string(index=False))
    print(f"\nTotal residues in reference set: {len(ref_df):,}")
    print("─────────────────────────────────────────────────────────────\n")

    return aa_stats


def compute_degree_stats_from_ref_dir(
    ref_centrality_dir,
    canonical_aa: set = None,
    exclude_pdb_ids: set = None,
) -> pd.DataFrame:
    """
    Build per-amino-acid-type degree statistics directly from a reference
    centrality directory, without loading the full DataFrames into memory.

    Only the 'Residue' and 'Degree' columns are read from each CSV, so this
    is fast even for large reference sets.  Non-canonical amino acids are
    excluded before computing statistics so the reference population exactly
    matches what load_centrality_dir() would produce.

    Use this function when you want to obtain aa_stats without first calling
    load_centrality_dir() on the entire reference set — for example, inside
    test_standardization() or any one-off analysis script.

    Parameters
    ----------
    ref_centrality_dir : str or Path
        Directory containing *_centrality.csv files for the reference
        (PISCES) population.
    canonical_aa : set of str, optional
        Set of three-letter residue names to retain.  Defaults to the 20
        standard amino acids.  Passed as uppercase strings (e.g. "ALA").
    exclude_pdb_ids : set of str, optional
        PDB IDs to skip (case-insensitive).  Useful when the reference
        directory contains proteins that should be excluded for quality
        or overlap reasons.

    Returns
    -------
    pd.DataFrame with columns: Residue, deg_mean, deg_std.
        Suitable for passing directly to standardize_degree_by_aa() or
        standardize_all().
    """
    if canonical_aa is None:
        canonical_aa = {
            "ALA", "ARG", "ASN", "ASP", "CYS",
            "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO",
            "SER", "THR", "TRP", "TYR", "VAL",
        }

    ref_centrality_dir = Path(ref_centrality_dir)
    exclude_pdb_ids    = {p.lower() for p in exclude_pdb_ids} if exclude_pdb_ids else set()

    csv_paths = sorted(ref_centrality_dir.glob("*_centrality.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No *_centrality.csv files found in: {ref_centrality_dir}"
        )

    frames = []
    for csv_path in csv_paths:
        pdb_id = csv_path.stem.replace("_centrality", "")
        if pdb_id.lower() in exclude_pdb_ids:
            continue
        df = pd.read_csv(csv_path, usecols=["Residue", "Degree"])
        df = df[df["Residue"].isin(canonical_aa)]
        frames.append(df)

    if not frames:
        raise ValueError(
            "No residues remained after filtering to canonical amino acids. "
            "Check that the CSVs contain a 'Residue' column with three-letter AA codes."
        )

    ref_df = pd.concat(frames, ignore_index=True)

    print(f"\n── compute_degree_stats_from_ref_dir ────────────────────────")
    print(f"  Reference directory : {ref_centrality_dir}")
    print(f"  CSV files read      : {len(frames):,}")
    print(f"  Total residues      : {len(ref_df):,}")

    aa_stats = compute_degree_stats_by_aa(ref_df)
    return aa_stats


def standardize_degree_by_aa(df: pd.DataFrame, aa_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Amitai method for degree: z-score each residue against its amino acid
    type's mean and std from the reference population.
    Adds StdDegree column.

    Parameters
    ----------
    df       : pd.DataFrame
        Must contain 'Degree' and 'Residue' columns.
    aa_stats : pd.DataFrame
        Output of compute_degree_stats_by_aa(). Contains Residue, deg_mean, deg_std.

    Returns
    -------
    pd.DataFrame with 'StdDegree' column added.
    """
    df = df.copy()
    df = df.merge(aa_stats, on="Residue", how="left")

    missing = df["deg_mean"].isna().sum()
    if missing > 0:
        unknown = df.loc[df["deg_mean"].isna(), "Residue"].unique()
        print(f"Warning: {missing} residues have no degree stats "
              f"(unexpected non-canonical AAs): {unknown}.")

    df["StdDegree"] = (df["Degree"] - df["deg_mean"]) / df["deg_std"]
    df.drop(columns=["deg_mean", "deg_std"], inplace=True)
    return df


# ── betweenness ───────────────────────────────────────────────────────────────

def standardize_betweenness_per_protein(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score BetweennessCentrality within each protein individually,
    following the same approach used for closeness (Amitai et al. Methods).

    Adds a StdBetweenness column.
    Proteins with std == 0 (degenerate case) receive StdBetweenness = 0.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'BetweennessCentrality' and 'pdb_id' columns.
        Run append_betweenness_to_centrality() in parallel_RIN_analysis.py
        first to populate the CSVs.

    Returns
    -------
    pd.DataFrame with 'StdBetweenness' column added.
    """
    if "BetweennessCentrality" not in df.columns:
        raise ValueError(
            "'BetweennessCentrality' column not found. "
            "Run append_betweenness_to_centrality() first to populate the CSVs."
        )

    df = df.copy()

    def _zscore(grp):
        mu  = grp["BetweennessCentrality"].mean()
        sig = grp["BetweennessCentrality"].std(ddof=0)
        grp = grp.copy()
        grp["StdBetweenness"] = (
            0.0 if sig == 0
            else (grp["BetweennessCentrality"] - mu) / sig
        )
        return grp

    return df.groupby("pdb_id", group_keys=False).apply(_zscore)


# ── PageRank ──────────────────────────────────────────────────────────────────

def rename_pagerank_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename the 'PageRank_zscore' column to 'StdPageRank' for consistency
    with the other Std* columns.

    PageRank_zscore is computed upstream by append_pagerank_to_centrality()
    in parallel_RIN_analysis.py as a per-protein z-score (ddof=0) of raw
    PageRank scores. No recomputation is performed here.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'PageRank_zscore' column.
        Run append_pagerank_to_centrality() in parallel_RIN_analysis.py
        first to populate the CSVs.

    Returns
    -------
    pd.DataFrame with 'PageRank_zscore' renamed to 'StdPageRank'.
    """
    if "PageRank_zscore" not in df.columns:
        raise ValueError(
            "'PageRank_zscore' column not found. "
            "Run append_pagerank_to_centrality() first to populate the CSVs."
        )

    return df.rename(columns={"PageRank_zscore": "StdPageRank"})


# ── RSA category ──────────────────────────────────────────────────────────────

def assign_rsa_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSA_category column using RSA-empirical values.

    Categories:
        'exposed'  — RSA > 0.5
        'core'     — RSA == 0.0
        'other'    — 0.0 < RSA <= 0.5
        'unknown'  — RSA-empirical column absent or NaN

    Parameters
    ----------
    df : pd.DataFrame
        Optionally contains 'RSA-empirical' column (from load_centrality_dir()
        with sasa_dir provided). If the column is absent, all rows are labeled
        'unknown'.

    Returns
    -------
    pd.DataFrame with 'RSA_category' column added.
    """
    df = df.copy()
    if "RSA-empirical" not in df.columns:
        df["RSA_category"] = "unknown"
        return df

    conditions = [df["RSA-empirical"] > 0.5, df["RSA-empirical"] == 0.0]
    choices    = ["exposed", "core"]
    df["RSA_category"] = np.select(conditions, choices, default="other")
    return df


# ── convenience wrapper ───────────────────────────────────────────────────────

def standardize_all(
    df: pd.DataFrame,
    aa_stats: pd.DataFrame,
    include_pagerank: bool = True,
) -> pd.DataFrame:
    """
    Apply all standardization steps in sequence to a single DataFrame.

    Intended for callers that want a one-liner after compute_degree_stats_by_aa()
    has been called on the reference set.

    Steps applied:
        1. standardize_closeness_per_protein()   → StdCloseness
        2. standardize_degree_by_aa()            → StdDegree
        3. standardize_betweenness_per_protein() → StdBetweenness
        4. rename_pagerank_zscore()              → StdPageRank  (if 'PageRank_zscore'
                                                   present and include_pagerank=True)
        5. assign_rsa_category()                 → RSA_category

    RCA (RCA_zscore) is not touched here — it is loaded directly from the
    centrality CSVs where it was written by append_rca_to_centrality().

    Parameters
    ----------
    df              : pd.DataFrame
        Loaded centrality data (output of load_centrality_dir() in
        0_4_fig1_recreation_ver8.py). Must contain at minimum:
        'pdb_id', 'ClosenessCentrality', 'Degree', 'Residue',
        'BetweennessCentrality'.
    aa_stats        : pd.DataFrame
        Output of compute_degree_stats_by_aa() called on the reference set.
    include_pagerank: bool
        If True (default), calls rename_pagerank_zscore() when
        'PageRank_zscore' is present in df. Set to False to skip.

    Returns
    -------
    pd.DataFrame with StdCloseness, StdDegree, StdBetweenness, RSA_category,
    and (conditionally) StdPageRank columns added.
    """
    df = standardize_closeness_per_protein(df)
    df = standardize_degree_by_aa(df, aa_stats)
    df = standardize_betweenness_per_protein(df)

    if include_pagerank and "PageRank_zscore" in df.columns:
        df = rename_pagerank_zscore(df)
    elif include_pagerank:
        print("  standardize_all: 'PageRank_zscore' column not found — skipping StdPageRank. "
              "Run append_pagerank_to_centrality() to populate the CSVs.")

    df = assign_rsa_category(df)
    return df


# ── append standardized columns to centrality CSVs ───────────────────────────

def append_standardized_to_centrality_dir(
    centrality_dir,
    aa_stats: pd.DataFrame,
) -> None:
    """
    Compute standardized centrality columns for every *_centrality.csv in
    centrality_dir and write them back into each file in-place (overwrite).

    Columns written (where source data is present):
        StdCloseness    — per-protein z-score of ClosenessCentrality
        StdDegree       — per-AA-type z-score of Degree (using aa_stats)
        StdBetweenness  — per-protein z-score of BetweennessCentrality
        StdPageRank     — rename of PageRank_zscore (if present)
        RSA_category    — 'exposed' / 'core' / 'other' / 'unknown'

    Each CSV is processed independently: the file is read, standardized
    columns are computed, and the full DataFrame (original + new columns)
    is written back to the same path.  Any pre-existing Std* or RSA_category
    columns are overwritten.

    Parameters
    ----------
    centrality_dir : str or Path
        Directory containing *_centrality.csv files to update in-place.
    aa_stats : pd.DataFrame
        Output of compute_degree_stats_by_aa() or
        compute_degree_stats_from_ref_dir(), computed on the reference
        (PISCES) population.  Must contain Residue, deg_mean, deg_std.
    """
    centrality_dir = Path(centrality_dir)
    csv_paths      = sorted(centrality_dir.glob("*_centrality.csv"))

    if not csv_paths:
        raise FileNotFoundError(
            f"No *_centrality.csv files found in: {centrality_dir}"
        )

    STD_COLS = ["StdCloseness", "StdDegree", "StdBetweenness", "StdPageRank", "RSA_category"]

    n_ok = n_errors = 0

    for csv_path in csv_paths:
        pdb_id = csv_path.stem.replace("_centrality", "")
        try:
            df = pd.read_csv(csv_path)

            # drop any pre-existing Std* / RSA_category columns so we get a clean write
            df.drop(columns=[c for c in STD_COLS if c in df.columns], inplace=True)

            df["pdb_id"] = pdb_id
            df = standardize_all(df, aa_stats)
            df.drop(columns=["pdb_id"], inplace=True)

            df.to_csv(csv_path, index=False)
            print(f"  Updated: {csv_path.name}")
            n_ok += 1

        except Exception as exc:
            print(f"  ERROR: {pdb_id}: {exc}")
            n_errors += 1

    print(
        f"\nappend_standardized_to_centrality_dir complete — "
        f"updated: {n_ok}, errors: {n_errors}"
    )


def test_standardization(centrality_csv, ref_centrality_dir) -> pd.DataFrame:
    """
    Load a single *_centrality.csv file, run all standardization steps,
    and print a summary of the results.

    Degree is standardized relative to the reference population (Amitai method),
    not relative to the single protein being tested.  All other centralities
    are per-protein z-scores as usual.

    Parameters
    ----------
    centrality_csv : str or Path
        Path to any existing *_centrality.csv file (typically from the test set).
    ref_centrality_dir : str or Path
        Directory containing the reference (PISCES) *_centrality.csv files.
        Used to compute per-amino-acid-type degree statistics via
        compute_degree_stats_from_ref_dir().

    Returns
    -------
    pd.DataFrame — the standardized DataFrame for further inspection.
    """
    centrality_csv = Path(centrality_csv)
    pdb_id = centrality_csv.stem.replace("_centrality", "")

    print(f"\nLoading: {centrality_csv.name}")
    df = pd.read_csv(centrality_csv)
    df["pdb_id"] = pdb_id

    print(f"  Rows loaded : {len(df)}")
    print(f"  Columns     : {df.columns.tolist()}")

    # Degree stats must come from the reference population, not from this single protein
    aa_stats = compute_degree_stats_from_ref_dir(ref_centrality_dir)
    result   = standardize_all(df, aa_stats)

    std_cols = ["StdCloseness", "StdDegree", "StdBetweenness", "StdPageRank", "RSA_category"]
    present  = [c for c in std_cols if c in result.columns]
    absent   = [c for c in std_cols if c not in result.columns]

    print("\n── Standardization results ──────────────────────────────────")
    for col in present:
        if col == "RSA_category":
            print(f"  {col:20s}: {result[col].value_counts().to_dict()}")
        else:
            print(f"  {col:20s}: mean={result[col].mean():.4f}  "
                  f"std={result[col].std():.4f}  "
                  f"min={result[col].min():.4f}  "
                  f"max={result[col].max():.4f}")
    if absent:
        print(f"\n  Skipped (columns not in CSV): {absent}")
    print("─────────────────────────────────────────────────────────────\n")

    print(result[["Node", "Residue"] + present].head(10).to_string(index=False))
    return result


if __name__ == "__main__":

    # ── Directories ───────────────────────────────────────────────────────────
    ref_centrality_dir  = DATA_DIR_3 / "ref"  / "Centralities"
    test_centrality_dir = DATA_DIR_3 / "test" / "Centralities"

    aa_stats = compute_degree_stats_from_ref_dir(ref_centrality_dir)

    print("\nStandardizing test centrality CSVs...")
    append_standardized_to_centrality_dir(test_centrality_dir, aa_stats)
