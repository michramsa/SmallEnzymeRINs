# correlation_heatmaps_ver1.py
#
# Purpose: One correlation heatmap per (method, dataset, set) showing the pairwise
# correlations between the standardized centralities (Degree, Closeness,
# Betweenness, PageRank). Each cell shows r and its bootstrap 95% CI.
# Optionally, one test-minus-reference difference heatmap per (method, dataset).
#
# Input  : output of bootstrap_correlations_ver1.py
#            <BOOT_DIR>/correlations.csv
#            <BOOT_DIR>/diff_test_minus_ref.csv   (only for the difference heatmaps)
#
# Outputs (under OUTPUT_DIR; one figure per file):
#   <method>/heatmap_<dataset>_<set>.<fmt>         r [95% CI], set = ref | test
#   <method>/heatmap_diff_<dataset>.<fmt>          r[test] - r[ref] [95% CI]
#
# Design notes
#   - Only the lower triangle is drawn; the matrix is symmetric and the diagonal
#     is 1 by definition, so that leaves the 6 informative pairs.
#   - All r heatmaps share one fixed colour scale (COLOR_LIMITS) so figures for
#     different datasets and sets can be compared by eye. The difference
#     heatmaps share their own scale, centred on 0 (DIFF_LIMITS).
#   - A cell is marked with an asterisk when the CI excludes 0 (r maps) or when
#     the difference CI excludes 0 (difference maps).
#
# All settings live in the `if __name__ == "__main__":` block at the bottom.

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style_ver2 import FONT, FIG_SIZE, CMAPS, apply_style, save_fig


def short_name(col: str) -> str:
    """StdDegree -> Degree."""
    return col[3:] if col.startswith("Std") else col


def build_matrices(sub: pd.DataFrame, std_cols: list):
    """Square (k x k) arrays of estimate / ci_low / ci_high, NaN where no pair."""
    k = len(std_cols)
    idx = {c: i for i, c in enumerate(std_cols)}
    est, lo, hi = (np.full((k, k), np.nan) for _ in range(3))
    for row in sub.itertuples(index=False):
        i, j = idx[row.var1], idx[row.var2]
        # lower triangle: row = later variable, column = earlier variable
        r, c = max(i, j), min(i, j)
        est[r, c], lo[r, c], hi[r, c] = row.estimate, row.ci_low, row.ci_high
    return est, lo, hi


def draw_heatmap(est, lo, hi, std_cols, title, cbar_label, cmap, vmin, vmax,
                 out_path, figsize, fmt, dpi, decimals):
    k = len(std_cols)
    labels = [short_name(c) for c in std_cols]

    # Lower triangle without the diagonal -> rows 1..k-1, columns 0..k-2
    E = est[1:, :-1]
    mask = np.tril(np.ones((k - 1, k - 1), dtype=bool))      # True where a pair exists
    shown = np.where(mask, E, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    im = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    # cell borders only where a pair exists
    for r in range(k - 1):
        for c in range(k - 1):
            if not mask[r, c]:
                continue
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                       edgecolor="white", linewidth=1.5))
            e, l, h = est[r + 1, c], lo[r + 1, c], hi[r + 1, c]
            if np.isnan(e):
                continue
            sig = "*" if (l > 0 or h < 0) else ""
            # white text on dark cells, black on light ones
            frac = (e - vmin) / (vmax - vmin)
            colour = "white" if (frac < 0.18 or frac > 0.82) else "black"
            ax.text(c, r, f"{e:.{decimals}f}{sig}\n[{l:.{decimals}f}, {h:.{decimals}f}]",
                    ha="center", va="center", fontsize=FONT["size_annot"], color=colour)

    ax.set_xticks(range(k - 1))
    ax.set_xticklabels(labels[:-1], rotation=30, ha="right")
    ax.set_yticks(range(k - 1))
    ax.set_yticklabels(labels[1:])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=FONT["size_title"], fontweight=FONT["weight_title"])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=FONT["size_label"])
    cbar.ax.tick_params(labelsize=FONT["size_tick"])
    cbar.outline.set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_path, dpi=dpi, fmt=fmt)
    plt.close(fig)


def main(
    boot_dir,
    output_dir,
    datasets: list,
    std_cols: list,
    methods: list,
    set_names: list,
    make_diff_heatmaps: bool,
    cmap_r,
    color_limits: tuple,
    cmap_diff,
    diff_limits: tuple,
    decimals: int,
    figsize: tuple,
    file_format: str,
    dpi: int,
):
    boot_dir, output_dir = Path(boot_dir), Path(output_dir)
    apply_style()

    corr = pd.read_csv(boot_dir / "correlations.csv")
    missing = set(std_cols) - set(corr["var1"]) - set(corr["var2"])
    if missing:
        raise ValueError(f"std_cols not found in correlations.csv: {sorted(missing)}")

    n_made = 0
    for m in methods:
        for ds in datasets:
            for s in set_names:
                sub = corr[(corr["method"] == m) & (corr["dataset"] == ds) & (corr["set_name"] == s)]
                if sub.empty:
                    print(f"  SKIP: no rows for {m} / {ds} / {s}")
                    continue
                est, lo, hi = build_matrices(sub, std_cols)
                n_prot = int(sub["n_proteins"].iloc[0])
                name = {"ref": "Reference", "test": "Test"}.get(s, s)
                draw_heatmap(
                    est, lo, hi, std_cols,
                    title=f"{ds}: {name} set (N = {n_prot})",
                    cbar_label=f"{m.capitalize()} r",
                    cmap=cmap_r, vmin=color_limits[0], vmax=color_limits[1],
                    out_path=output_dir / m / f"heatmap_{ds}_{s}.{file_format}",
                    figsize=figsize, fmt=file_format, dpi=dpi, decimals=decimals)
                n_made += 1

    if make_diff_heatmaps:
        diff = pd.read_csv(boot_dir / "diff_test_minus_ref.csv")
        # diff file has no var1/var2 columns, so recover them from the pair label
        pair_vars = (corr[["pair", "var1", "var2"]].drop_duplicates().set_index("pair"))
        diff = diff.join(pair_vars, on="pair")
        for m in methods:
            for ds in datasets:
                sub = diff[(diff["method"] == m) & (diff["dataset"] == ds)]
                if sub.empty:
                    print(f"  SKIP: no difference rows for {m} / {ds}")
                    continue
                est, lo, hi = build_matrices(sub, std_cols)
                draw_heatmap(
                    est, lo, hi, std_cols,
                    title=f"{ds}: test − reference",
                    cbar_label=f"{m.capitalize()} r (test − ref)",
                    cmap=cmap_diff, vmin=diff_limits[0], vmax=diff_limits[1],
                    out_path=output_dir / m / f"heatmap_diff_{ds}.{file_format}",
                    figsize=figsize, fmt=file_format, dpi=dpi, decimals=decimals)
                n_made += 1

    print(f"\nWrote {n_made} heatmap(s) to: {output_dir}")


if __name__ == "__main__":

    # ── Edit these values ─────────────────────────────────────────────────────
    BOOT_DIR = Path("...0_4_revisions/2_experiments/6_bootstrap/std_ver4")        # output of bootstrap_correlations_ver1.py
    OUTPUT_DIR = Path("...0_4_revisions/2_experiments/6_bootstrap/std_ver4/figures_heatmaps")

    DATASETS = ["arp_iac", "arp_no_iac", "ring_iac", "ring_no_iac", "pcn5", "pcn6", "pcn7"]
    STD_COLS = ["StdDegree", "StdCloseness", "StdBetweenness", "StdPageRank"]   # matrix order
    METHODS = ["spearman", "pearson"]                          # either or both
    SET_NAMES = ["ref", "test"]

    MAKE_DIFF_HEATMAPS = True        # extra: test - reference difference per dataset

    CMAP_R = "RdBu_r"                # diverging, centred on 0
    COLOR_LIMITS = (-1.0, 1.0)       # fixed so every r heatmap uses the same scale
    CMAP_DIFF = "PuOr"
    DIFF_LIMITS = (-0.3, 0.3)        # widen if your differences are larger

    DECIMALS = 2
    FIGSIZE = (4.96, 4.2)            # inches; 1.5-column PLOS width
    FILE_FORMAT = "png"             # PLOS accepts tiff or eps; use "png" for quick looks
    DPI = 300
    # ──────────────────────────────────────────────────────────────────────────

    main(
        boot_dir=BOOT_DIR,
        output_dir=OUTPUT_DIR,
        datasets=DATASETS,
        std_cols=STD_COLS,
        methods=METHODS,
        set_names=SET_NAMES,
        make_diff_heatmaps=MAKE_DIFF_HEATMAPS,
        cmap_r=CMAP_R,
        color_limits=COLOR_LIMITS,
        cmap_diff=CMAP_DIFF,
        diff_limits=DIFF_LIMITS,
        decimals=DECIMALS,
        figsize=FIGSIZE,
        file_format=FILE_FORMAT,
        dpi=DPI,
    )
