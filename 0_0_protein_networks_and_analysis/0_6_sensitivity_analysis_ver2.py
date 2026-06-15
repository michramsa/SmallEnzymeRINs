# sensitivity_analysis.py
# -----------------------
# Reference-set subsampling sensitivity analysis.
#
# Purpose:
#   Demonstrate that the per-amino-acid-type degree statistics used to
#   standardize StdDegree are stable with respect to the size of the PISCES
#   reference set. For each of N_REPS random subsamplings at each fraction
#   level, the full reference set is loaded, a random subset of proteins is
#   drawn, per-AA degree mean and std are recomputed, and the results are
#   compared against the full-set baseline.
#
# Output files (each figure is its own file per project conventions):
#   sensitivity_mean_deviation.tiff  — mean absolute deviation of per-AA means
#   sensitivity_std_deviation.tiff   — mean absolute deviation of per-AA stds
#   sensitivity_raw_curves.tiff      — per-AA mean ± std vs. fraction (all AAs)
#   sensitivity_summary.csv          — numeric results for all fractions and reps
#
# Standardization follows 0_4_fig1_recreation_ver2.py exactly:
#   DEGREE: z-scored per amino acid type using reference population statistics.
#           Uses raw integer Degree counts (not normalized DegreeCentrality).
#   Only the 20 canonical amino acids are included.

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from figure_style_ver2 import (
    apply_style, save_fig, FIG_SIZE, FONT, LAYOUT, DPI, COLORS, COLOR_CYCLE,
)

# ── constants ─────────────────────────────────────────────────────────────────

CANONICAL_AA = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]
CANONICAL_AA_SET = set(CANONICAL_AA)


# ── data loading ──────────────────────────────────────────────────────────────

def load_pisces_centrality(centrality_dir: Path) -> pd.DataFrame:
    """
    Load all *_centrality.csv files from the PISCES centrality directory into
    one concatenated DataFrame. Filters to canonical amino acids only.

    Requires columns: Residue, Degree (raw integer count).
    Adds a pdb_id column derived from the filename stem.

    Parameters
    ----------
    centrality_dir : Path
        Directory containing *_centrality.csv files.

    Returns
    -------
    pd.DataFrame with columns [pdb_id, Residue, Degree].
    """
    centrality_dir = Path(centrality_dir)
    frames = []

    csv_files = sorted(centrality_dir.glob("*_centrality.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No *_centrality.csv files found in {centrality_dir}"
        )

    for csv_path in csv_files:
        pdb_id = csv_path.stem.replace("_centrality", "")
        df     = pd.read_csv(csv_path)

        if "Degree" not in df.columns:
            print(f"  WARNING: {csv_path.name} has no Degree column — skipping.")
            continue

        df["pdb_id"] = pdb_id
        df = df[df["Residue"].isin(CANONICAL_AA_SET)][["pdb_id", "Residue", "Degree"]].copy()
        df["Degree"] = pd.to_numeric(df["Degree"], errors="coerce")
        df.dropna(subset=["Degree"], inplace=True)
        df["Degree"] = df["Degree"].astype(int)
        frames.append(df)

    if not frames:
        raise ValueError("No valid centrality CSVs could be loaded.")

    combined = pd.concat(frames, ignore_index=True)
    print(
        f"  Loaded {len(combined):,} residues from "
        f"{combined['pdb_id'].nunique()} proteins."
    )
    return combined


# ── per-AA statistics ─────────────────────────────────────────────────────────

def compute_degree_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-amino-acid-type mean and std (ddof=0) of raw Degree across
    all residues in df. Follows the Amitai et al. standardization exactly.

    Parameters
    ----------
    df : pd.DataFrame with columns [Residue, Degree].

    Returns
    -------
    pd.DataFrame with columns [Residue, deg_mean, deg_std], indexed by AA.
    """
    stats = (
        df.groupby("Residue")["Degree"]
        .agg(
            deg_mean="mean",
            deg_std=lambda x: x.std(ddof=0),
        )
        .reindex(CANONICAL_AA)   # consistent ordering across subsamples
        .reset_index()
    )
    return stats


# ── subsampling experiment ────────────────────────────────────────────────────

def run_subsampling(
    full_df:     pd.DataFrame,
    fractions:   list,
    n_reps:      int,
    random_seed: int,
) -> pd.DataFrame:
    """
    For each fraction in `fractions`, draw `n_reps` random subsets of proteins
    from `full_df`, compute per-AA degree statistics, and record deviation
    from the full-set baseline.

    Parameters
    ----------
    full_df     : Full PISCES DataFrame [pdb_id, Residue, Degree].
    fractions   : List of floats in (0, 1] — fraction of proteins to include.
    n_reps      : Number of independent random draws per fraction.
    random_seed : Base seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns:
        fraction, rep, aa, deg_mean, deg_std,
        mean_dev (|subsample_mean - baseline_mean|),
        std_dev  (|subsample_std  - baseline_std|).
    """
    rng          = np.random.default_rng(random_seed)
    all_proteins = full_df["pdb_id"].unique()
    n_total      = len(all_proteins)

    print(f"  Computing baseline statistics on {n_total} proteins...")
    baseline = compute_degree_stats(full_df).set_index("Residue")

    records = []

    for frac in fractions:
        n_sample = max(1, int(round(frac * n_total)))
        print(f"  Fraction {frac:.2f} ({n_sample}/{n_total} proteins) — {n_reps} reps")

        for rep in range(n_reps):
            seed = int(rng.integers(0, 2**31))
            rng_rep  = np.random.default_rng(seed)
            sampled  = rng_rep.choice(all_proteins, size=n_sample, replace=False)
            sub_df   = full_df[full_df["pdb_id"].isin(sampled)]
            sub_stats = compute_degree_stats(sub_df).set_index("Residue")

            for aa in CANONICAL_AA:
                s_mean = sub_stats.loc[aa, "deg_mean"] if aa in sub_stats.index else np.nan
                s_std  = sub_stats.loc[aa, "deg_std"]  if aa in sub_stats.index else np.nan
                b_mean = baseline.loc[aa, "deg_mean"]
                b_std  = baseline.loc[aa, "deg_std"]

                records.append({
                    "fraction":  frac,
                    "n_proteins": n_sample,
                    "rep":       rep,
                    "aa":        aa,
                    "deg_mean":  s_mean,
                    "deg_std":   s_std,
                    "mean_dev":  abs(s_mean - b_mean) if not np.isnan(s_mean) else np.nan,
                    "std_dev":   abs(s_std  - b_std)  if not np.isnan(s_std)  else np.nan,
                })

    return pd.DataFrame(records)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_aggregate_deviation(
    results_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    output_path_mean: Path,
    output_path_std:  Path,
) -> None:
    """
    Two separate figures:
      1. Mean absolute deviation of per-AA degree MEANS vs. subsample fraction.
      2. Mean absolute deviation of per-AA degree STDs  vs. subsample fraction.

    Each figure shows the median across AAs and reps as a line with a shaded
    band spanning the 25th–75th percentile across AAs, giving a sense of
    per-AA variability.

    Parameters
    ----------
    results_df       : Output of run_subsampling().
    baseline_df      : Full-set per-AA stats from compute_degree_stats().
    output_path_mean : Save path for the means deviation figure.
    output_path_std  : Save path for the stds deviation figure.
    """
    apply_style()

    # Aggregate: for each (fraction, rep) compute the mean absolute deviation
    # averaged across all 20 AAs — then take median and IQR across reps.
    agg = (
        results_df
        .groupby(["fraction", "n_proteins", "rep"])[["mean_dev", "std_dev"]]
        .mean()                          # mean over 20 AAs within each rep
        .reset_index()
        .groupby(["fraction", "n_proteins"])
        .agg(
            mean_dev_med=("mean_dev", "median"),
            mean_dev_q25=("mean_dev", lambda x: np.nanpercentile(x, 25)),
            mean_dev_q75=("mean_dev", lambda x: np.nanpercentile(x, 75)),
            std_dev_med =("std_dev",  "median"),
            std_dev_q25 =("std_dev",  lambda x: np.nanpercentile(x, 25)),
            std_dev_q75 =("std_dev",  lambda x: np.nanpercentile(x, 75)),
        )
        .reset_index()
        .sort_values("fraction")
    )

    fracs      = agg["fraction"].values
    n_proteins = agg["n_proteins"].values

    # ── Figure 1: mean deviation ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_SIZE["half_page"])

    ax.fill_between(
        fracs,
        agg["mean_dev_q25"].values,
        agg["mean_dev_q75"].values,
        color=COLORS["blue"],
        alpha=0.20,
        linewidth=0,
        label="IQR across reps",
    )
    ax.plot(
        fracs,
        agg["mean_dev_med"].values,
        color=COLORS["blue"],
        linewidth=LAYOUT["line_width"],
        marker="o",
        markersize=LAYOUT["marker_size"],
        label="Median |deviation|",
    )

    # secondary x-axis showing absolute protein count
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_fracs = fracs
    ax2.set_xticks(tick_fracs)
    ax2.set_xticklabels(
        [str(n) for n in n_proteins],
        fontsize=FONT["size_tick"],
        rotation=45,
    )
    ax2.set_xlabel("Number of proteins", fontsize=FONT["size_label"])
    ax2.spines["top"].set_visible(True)
    ax2.spines["top"].set_linewidth(LAYOUT["spine_linewidth"])

    ax.set_xlabel("Fraction of reference set", fontsize=FONT["size_label"])
    ax.set_ylabel(
        "Mean |Δ degree mean| per AA type",
        fontsize=FONT["size_label"],
    )
    ax.legend(fontsize=FONT["size_legend"], loc="upper right")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout(pad=LAYOUT["tight_pad"])
    save_fig(fig, output_path_mean)
    plt.close(fig)

    # ── Figure 2: std deviation ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_SIZE["half_page"])

    ax.fill_between(
        fracs,
        agg["std_dev_q25"].values,
        agg["std_dev_q75"].values,
        color=COLORS["orange"],
        alpha=0.20,
        linewidth=0,
        label="IQR across reps",
    )
    ax.plot(
        fracs,
        agg["std_dev_med"].values,
        color=COLORS["orange"],
        linewidth=LAYOUT["line_width"],
        marker="o",
        markersize=LAYOUT["marker_size"],
        label="Median |deviation|",
    )

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(tick_fracs)
    ax2.set_xticklabels(
        [str(n) for n in n_proteins],
        fontsize=FONT["size_tick"],
        rotation=45,
    )
    ax2.set_xlabel("Number of proteins", fontsize=FONT["size_label"])
    ax2.spines["top"].set_visible(True)
    ax2.spines["top"].set_linewidth(LAYOUT["spine_linewidth"])

    ax.set_xlabel("Fraction of reference set", fontsize=FONT["size_label"])
    ax.set_ylabel(
        "Δ degree std per AA type",
        fontsize=FONT["size_label"],
    )
    ax.legend(fontsize=FONT["size_legend"], loc="upper right")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout(pad=LAYOUT["tight_pad"])
    save_fig(fig, output_path_std)
    plt.close(fig)

    print(f"  Saved: {output_path_mean.name}")
    print(f"  Saved: {output_path_std.name}")


def plot_per_aa_curves(
    results_df:  pd.DataFrame,
    baseline_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Per-amino-acid subsampling curves: one panel per AA showing the raw
    degree mean ± 1 std (across reps) vs. subsample fraction, with the
    full-set baseline drawn as a horizontal dashed line.

    Laid out as a 4 × 5 grid (20 canonical AAs).

    Parameters
    ----------
    results_df   : Output of run_subsampling().
    baseline_df  : Full-set per-AA stats from compute_degree_stats().
    output_path  : Save path.
    """
    apply_style()

    baseline = baseline_df.set_index("Residue")

    # Aggregate per (fraction, aa): median and std of deg_mean across reps
    agg = (
        results_df
        .groupby(["fraction", "aa"])["deg_mean"]
        .agg(
            med   ="median",
            q25   =lambda x: np.nanpercentile(x, 25),
            q75   =lambda x: np.nanpercentile(x, 75),
        )
        .reset_index()
        .sort_values(["aa", "fraction"])
    )

    n_cols = 5
    n_rows = 4
    # sharex=False and no shared sharey: each panel autoscales to its own
    # data range so that subsample variation is visible within each AA.
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7.00, 5.5),   # full-width, within PLOS height limit
        sharex=False,
        sharey=False,
    )
    axes_flat = axes.flatten()

    for i, aa in enumerate(CANONICAL_AA):
        ax    = axes_flat[i]
        sub   = agg[agg["aa"] == aa]
        fracs = sub["fraction"].values

        ax.fill_between(
            fracs,
            sub["q25"].values,
            sub["q75"].values,
            color=COLORS["orange"],
            alpha=0.20,
            linewidth=0,
        )
        ax.plot(
            fracs,
            sub["med"].values,
            color=COLORS["orange"],
            linewidth=LAYOUT["line_width"],
            marker="o",
            markersize=2.5,
        )

        # baseline
        b_mean = baseline.loc[aa, "deg_mean"] if aa in baseline.index else np.nan
        ax.axhline(
            b_mean,
            color=COLORS["blue"],
            linewidth=0.8,
            linestyle="--",
            label="Full set" if i == 0 else None,
        )

        ax.set_title(aa, fontsize=FONT["size_tick"], pad=2)
        ax.tick_params(labelsize=6)

        # autoscale y around the actual data range for this AA, with a small
        # margin — do NOT force bottom=0, which would swamp the variation
        all_vals = np.concatenate([
            sub["q25"].values, sub["q75"].values, [b_mean]
        ])
        all_vals = all_vals[~np.isnan(all_vals)]
        if len(all_vals) > 0:
            lo, hi = all_vals.min(), all_vals.max()
            margin = max((hi - lo) * 0.3, 0.1)   # at least 0.1 unit margin
            ax.set_ylim(lo - margin, hi + margin)

        # x-axis: fraction labels, thinned to avoid overlap
        ax.set_xlim(fracs[0] - 0.05, 1.05)
        ax.set_xticks([0.1, 0.5, 1.0])
        ax.set_xticklabels(["0.1", "0.5", "1.0"], fontsize=5)

    # shared axis labels via a hidden outer axis
    for ax in axes_flat[len(CANONICAL_AA):]:
        ax.set_visible(False)

    fig.supxlabel(
        "Fraction of reference set",
        fontsize=FONT["size_label"],
        y=0.01,
    )
    fig.supylabel(
        "Degree mean (raw counts)",
        fontsize=FONT["size_label"],
        x=0.01,
    )

    # single legend for baseline line
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=COLORS["blue"],  linewidth=0.8, linestyle="--", label="Full-set baseline"),
        Line2D([0], [0], color=COLORS["orange"], linewidth=LAYOUT["line_width"], label="Subsample median"),
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        ncol=1,                        # stack vertically
        fontsize=FONT["size_legend"],
        framealpha=LAYOUT["legend_framealpha"],
        bbox_to_anchor=(0.01, 0.01),   # small offset from bottom-left corner
    )

    fig.tight_layout(pad=LAYOUT["tight_pad"], rect=[0, 0.06, 1, 1])
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    pisces_centrality_dir = DATA_DIR_2 / "old_PISCES" / "Centralities",
    output_dir            = DATA_DIR_2 / "sensitivity_analysis",
    fractions             = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00),
    n_reps                = 30,
    random_seed           = 42,
):
    """
    Run the reference-set subsampling sensitivity analysis and save all outputs.

    Parameters
    ----------
    pisces_centrality_dir : Path
        Directory containing the PISCES *_centrality.csv files produced by
        0_3_1_parallel_RIN_analysis_ver3.py. Each CSV must have a Degree column.
    output_dir : Path
        All output figures and the summary CSV are written here.
    fractions : sequence of float
        Fractions of the full protein set to subsample. 1.0 is always included
        and represents the full reference set baseline.
    n_reps : int
        Number of independent random draws per fraction. Higher values reduce
        noise in the IQR band. 30 is a reasonable default; use 10 for a quick
        preview run.
    random_seed : int
        Seed for the top-level RNG. Each replicate draws its own child seed
        deterministically from this, so results are fully reproducible.
    """
    pisces_centrality_dir = Path(pisces_centrality_dir)
    output_dir            = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ensure fraction=1.0 is always present
    fractions = sorted(set(list(fractions) + [1.0]))

    print("=" * 60)
    print("Reference-set subsampling sensitivity analysis")
    print("=" * 60)
    print(f"  PISCES centrality dir : {pisces_centrality_dir}")
    print(f"  Output dir            : {output_dir}")
    print(f"  Fractions             : {fractions}")
    print(f"  Reps per fraction     : {n_reps}")
    print(f"  Random seed           : {random_seed}")
    print()

    # ── 1. load full PISCES set ────────────────────────────────────────────
    print("Loading PISCES centrality CSVs...")
    full_df = load_pisces_centrality(pisces_centrality_dir)

    n_proteins_total = full_df["pdb_id"].nunique()
    n_residues_total = len(full_df)
    print(
        f"  Full reference set: {n_proteins_total} proteins, "
        f"{n_residues_total:,} canonical residues.\n"
    )

    # ── 2. baseline statistics (full set) ─────────────────────────────────
    print("Computing full-set baseline per-AA degree statistics...")
    baseline_df = compute_degree_stats(full_df)
    baseline_df.to_csv(output_dir / "baseline_degree_stats.csv", index=False)
    print("  Saved: baseline_degree_stats.csv")
    print()
    print(baseline_df.to_string(index=False))
    print()

    # ── 3. subsampling experiment ──────────────────────────────────────────
    print("Running subsampling experiment...")
    results_df = run_subsampling(
        full_df     = full_df,
        fractions   = fractions,
        n_reps      = n_reps,
        random_seed = random_seed,
    )
    results_df.to_csv(output_dir / "sensitivity_summary.csv", index=False)
    print("\n  Saved: sensitivity_summary.csv")
    print()

    # ── 4. figures ────────────────────────────────────────────────────────
    print("Generating figures...")

    plot_aggregate_deviation(
        results_df       = results_df,
        baseline_df      = baseline_df,
        output_path_mean = output_dir / "sensitivity_mean_deviation.png",
        output_path_std  = output_dir / "sensitivity_std_deviation.png",
    )

    plot_per_aa_curves(
        results_df  = results_df,
        baseline_df = baseline_df,
        output_path = output_dir / "sensitivity_raw_curves.png",
    )

    print()
    print("=" * 60)
    print("Done. Output files:")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<45}  {size_kb:6.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main(
        pisces_centrality_dir = DATA_DIR_2 / "old_PISCES" / "Centralities_f",
        output_dir            = DATA_DIR_2 / "figures" / "controlled_fig1_recreation_ver0.7" / "sensitivity_analysis",
        fractions             = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00),
        n_reps                = 30,
        random_seed           = 42,
    )
