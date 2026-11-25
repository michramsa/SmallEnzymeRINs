# bin_sensitivity_ver1.py
#
# Purpose: Test whether the standardized centralities depend on the number of
# protein-size bins used in standardization_ver3.py.
#
# For each dataset, the reference and test sets are standardized again with
# several bin counts (N_BINS_LIST), using the SAME functions as
# standardization_ver3.py (imported, not copied, so the method is identical).
# Every run is then compared with the BASELINE run (N_BINS = 5, the value used
# in the paper).
#
# n_bins = 1 means no size adjustment at all (amino-acid type only, as in
# Amitai et al. 2004 for degree). It shows how much the size bins change things.
#
# Statistics per (dataset, centrality, n_bins), written to bin_sensitivity.csv:
#   n_bins_formed               bins actually formed (duplicate quantile edges merge)
#   n_fallback_cells            (amino acid, bin) cells with < MIN_CELL_RESIDUES
#                               reference residues that fell back to all-sizes stats
#   ref_size_rho, test_size_rho Spearman rho between per-protein mean Std value and
#                               protein size (n_nodes). Want ~0: size removed.
#   test_minus_ref_mean         mean Std (test) - mean Std (ref), in SD units
#   spearman_vs_baseline        Spearman rho between each test residue's Std value
#                               in this run and in the baseline run. 1 = same ranking.
#   jaccard_central_vs_baseline overlap of the test residues called "central"
#                               (Std >= CENTRAL_THRESHOLD) in this run vs baseline:
#                               |A and B| / |A or B|. 1 = identical set.
#   frac_central_test           fraction of test residues with Std >= CENTRAL_THRESHOLD
#
# Figure (one file): FIGURE_STAT against the number of bins, one line per
# centrality. The line is the mean over datasets; the shaded band is the
# min-max range over datasets. Change FIGURE_STAT to plot any column above.
#
# Outputs (under OUTPUT_ROOT):
#   bin_sensitivity.csv
#   bin_sensitivity_<FIGURE_STAT>.png
#
# All settings live in the `if __name__ == "__main__":` block at the bottom.

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ── helpers ───────────────────────────────────────────────────────────────────

def load_std_module(std_dir, module_name):
    """Import standardization_ver3.py (or another version) from std_dir."""
    std_dir = str(Path(std_dir).resolve())
    if std_dir not in sys.path:
        sys.path.insert(0, std_dir)
    return importlib.import_module(module_name)


def standardize_with_bins(std, ref_df, test_df, n_bins, col_map, min_cell_residues):
    """One full standardization run with n_bins size bins (same steps as ver3 main)."""
    edges = std.make_interior_edges(ref_df, n_bins)
    ref_b = std.assign_bins(ref_df, edges)
    test_b = std.assign_bins(test_df, edges)
    stats_long, lookup = std.reference_stats(ref_b, list(col_map), min_cell_residues)
    ref_s = std.standardize(ref_b, lookup, col_map)
    test_s = std.standardize(test_b, lookup, col_map)
    return edges, stats_long, ref_s, test_s


def size_rho(df, col):
    """Spearman rho between per-protein mean of col and protein size."""
    p = (df.groupby("PDBid")
         .agg(n_nodes=("n_nodes", "first"), value=(col, "mean"))
         .dropna())
    if len(p) < 3:
        return np.nan
    return stats.spearmanr(p["n_nodes"], p["value"])[0]


def agreement(values, baseline, threshold):
    """Spearman rho and Jaccard overlap of central residues between two runs."""
    ok = np.isfinite(values) & np.isfinite(baseline)
    rho = stats.spearmanr(values[ok], baseline[ok])[0] if ok.sum() > 2 else np.nan
    a = values[ok] >= threshold
    b = baseline[ok] >= threshold
    union = (a | b).sum()
    jaccard = (a & b).sum() / union if union else np.nan
    return rho, jaccard


# ── figure ────────────────────────────────────────────────────────────────────

def plot_stat_vs_bins(results, stat, std_cols, labels, colors, markers,
                      baseline_n_bins, figure_datasets, out_path, dpi):
    df = results if figure_datasets is None else results[results["dataset"].isin(figure_datasets)]
    n_datasets = df["dataset"].nunique()

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8, zorder=0)

    # reference line: 1 for agreement statistics, 0 for size / shift statistics
    ref_y = 1.0 if stat in ("spearman_vs_baseline", "jaccard_central_vs_baseline") else 0.0
    if stat != "frac_central_test":
        ax.axhline(ref_y, color="#8a8984", linewidth=0.8, zorder=1)
    ax.axvline(baseline_n_bins, color="#8a8984", linestyle=":", linewidth=1, zorder=1)

    for col in std_cols:
        g = (df[df["centrality"] == col]
             .groupby("n_bins_requested")[stat]
             .agg(["mean", "min", "max"])
             .sort_index())
        x = g.index.to_numpy(float)
        ax.fill_between(x, g["min"], g["max"], color=colors[col], alpha=0.15,
                        linewidth=0, zorder=2)
        ax.plot(x, g["mean"], color=colors[col], linewidth=2, marker=markers[col],
                markersize=7, markeredgecolor="white", markeredgewidth=1.5,
                label=labels.get(col, col), zorder=3)

    ax.set_xticks(sorted(df["n_bins_requested"].unique()))
    ax.set_xlabel(f"Number of size bins (dotted line = baseline, {baseline_n_bins})")
    ax.set_ylabel(stat.replace("_", " "))
    ax.set_title(f"Sensitivity to size-bin count: {stat.replace('_', ' ')}\n"
                 f"line = mean over {n_datasets} dataset(s); band = min-max",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    std_dir,
    std_module,
    centrality_root,
    output_root,
    datasets: list,
    col_map: dict,
    n_bins_list: list,
    baseline_n_bins: int,
    min_cell_residues: int,
    central_threshold: float,
    figure_stat: str,
    figure_datasets,
    std_labels: dict,
    colors: dict,
    markers: dict,
    dpi: int,
):
    std = load_std_module(std_dir, std_module)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    n_bins_list = sorted(set(n_bins_list) | {baseline_n_bins})
    std_cols = list(col_map.values())
    rows = []

    for dataset in datasets:
        print(f"\n══ {dataset} ══")
        ref_df = std.load_set(centrality_root, dataset, "ref")
        test_df = std.load_set(centrality_root, dataset, "test")

        runs = {}
        for n_bins in n_bins_list:
            print(f"  -- n_bins = {n_bins}")
            runs[n_bins] = standardize_with_bins(std, ref_df, test_df, n_bins,
                                                 col_map, min_cell_residues)

        base_test = runs[baseline_n_bins][3]
        for n_bins, (edges, stats_long, ref_s, test_s) in runs.items():
            for raw, col in col_map.items():
                cell_stats = stats_long[stats_long["centrality"] == raw]
                values = test_s[col].to_numpy(float)
                rho_base, jac = agreement(values, base_test[col].to_numpy(float),
                                          central_threshold)
                finite = values[np.isfinite(values)]
                rows.append({
                    "dataset": dataset,
                    "centrality": col,
                    "n_bins_requested": n_bins,
                    "n_bins_formed": len(edges) + 1,
                    "n_fallback_cells": int(cell_stats["fallback_to_all_sizes"].sum()),
                    "ref_size_rho": size_rho(ref_s, col),
                    "test_size_rho": size_rho(test_s, col),
                    "test_minus_ref_mean": test_s[col].mean() - ref_s[col].mean(),
                    "spearman_vs_baseline": rho_base,
                    "jaccard_central_vs_baseline": jac,
                    "frac_central_test": (finite >= central_threshold).mean() if len(finite) else np.nan,
                })

    results = pd.DataFrame(rows)
    csv_path = output_root / "bin_sensitivity.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nSaved table: {csv_path}")

    print(f"\n── {figure_stat} (mean over datasets) ──")
    print(results.pivot_table(index="centrality", columns="n_bins_requested",
                              values=figure_stat, aggfunc="mean")
          .to_string(float_format=lambda v: f"{v:+.3f}"))

    plot_stat_vs_bins(results, figure_stat, std_cols, std_labels, colors, markers,
                      baseline_n_bins, figure_datasets,
                      output_root / f"bin_sensitivity_{figure_stat}.png", dpi)


if __name__ == "__main__":

    # ── Edit these values ─────────────────────────────────────────────────────
    # Folder that contains standardization_ver3.py, and its module name
    STD_DIR = Path("...0_4_revisions/2_experiments/5_standardization")
    STD_MODULE = "standardization_ver4"  # name this whatever you would want

    # Raw centralities (output of parallel_RIN_centrality_ver1.py)
    CENTRALITY_ROOT = Path("...0_4_revisions/2_experiments/4_centralities")
    # New folder for this test
    OUTPUT_ROOT = Path("...0_4_revisions/2_experiments/5_standardization/bin_sensitivity")

    DATASETS = ["arp_iac", "arp_no_iac", "ring_iac", "ring_no_iac", "pcn5", "pcn6", "pcn7"]

    # Must match COL_MAP in standardization_ver3.py
    COL_MAP = {
        "Degree":                "StdDegree",
        "ClosenessCentrality":   "StdCloseness",
        "BetweennessCentrality": "StdBetweenness",
        "PageRank_xN":           "StdPageRank",
    }

    N_BINS_LIST = [1, 3, 5, 8, 10]   # 1 = no size adjustment (amino acid only)
    BASELINE_N_BINS = 5              # value used in the paper
    MIN_CELL_RESIDUES = 30           # same as standardization_ver3.py
    CENTRAL_THRESHOLD = 2.0          # Std value that counts as "central"

    # Column of bin_sensitivity.csv to plot. Options:
    #   "spearman_vs_baseline", "jaccard_central_vs_baseline",
    #   "test_size_rho", "ref_size_rho", "test_minus_ref_mean",
    #   "frac_central_test", "n_fallback_cells"
    FIGURE_STAT = "spearman_vs_baseline"
    FIGURE_DATASETS = None           # None = all datasets, or e.g. ["ring_iac"]

    STD_LABELS = {
        "StdDegree": "Degree", "StdCloseness": "Closeness",
        "StdBetweenness": "Betweenness", "StdPageRank": "PageRank",
    }
    COLORS = {
        "StdDegree": "#2a78d6", "StdCloseness": "#eb6834",
        "StdBetweenness": "#1baf7a", "StdPageRank": "#eda100",
    }
    MARKERS = {
        "StdDegree": "o", "StdCloseness": "s",
        "StdBetweenness": "^", "StdPageRank": "D",
    }
    DPI = 300
    # ──────────────────────────────────────────────────────────────────────────

    main(
        std_dir=STD_DIR,
        std_module=STD_MODULE,
        centrality_root=CENTRALITY_ROOT,
        output_root=OUTPUT_ROOT,
        datasets=DATASETS,
        col_map=COL_MAP,
        n_bins_list=N_BINS_LIST,
        baseline_n_bins=BASELINE_N_BINS,
        min_cell_residues=MIN_CELL_RESIDUES,
        central_threshold=CENTRAL_THRESHOLD,
        figure_stat=FIGURE_STAT,
        figure_datasets=FIGURE_DATASETS,
        std_labels=STD_LABELS,
        colors=COLORS,
        markers=MARKERS,
        dpi=DPI,
    )
