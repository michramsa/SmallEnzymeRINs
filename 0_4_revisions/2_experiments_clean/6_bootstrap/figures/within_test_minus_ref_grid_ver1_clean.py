# within_test_minus_ref_grid_ver1.py
#
# Purpose: one multi-panel figure per method (Spearman, Pearson) combining the
# six within_test_minus_ref_<pair> figures into a 2 x 3 grid, one panel per
# centrality pair (DC vs CC, DC vs BC, ...).
#
# Uses the same colours, markers, labels and drawing code as
# within_protein_figures_ver1.py, which must sit in the same folder (its
# helpers are imported, so a change there carries over here).
#
# Input : within_test_minus_ref.csv  (written by bootstrap_within_protein.py)
# Output: <FIGURE_DIR>/within_test_minus_ref_grid_<method>.<fmt>
#         one file per method
#
# Layout choices (all switchable in __main__):
#   - shared y axis across the six panels, so the size of the test - reference
#     shift can be compared between pairs by eye
#   - y-axis label only on the left column, x tick labels only on the bottom row
#   - one legend for the whole figure, centred below the grid
#   - panel letters A-F, panel titles = the centrality pair
#
# All settings live in the `if __name__ == "__main__":` block at the bottom.

import string
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from within_protein_figures_ver1 import (  # noqa: E402
    _FALLBACK_FIG_SIZE, draw_series, load_style, ordered_unique,
    series_legend_handles, short_pair,
)


def draw_panel(ax, diff_df, method, pair, datasets, kinds, labels, series_style, cfg):
    """One test - reference panel (same content as the single-file figure)."""
    x = np.arange(len(datasets))
    ax.axhline(0, color=cfg["zero_line_color"], linewidth=cfg["zero_line_width"], zorder=1)
    for off, kind in zip(cfg["series_offsets"], kinds):
        s = diff_df[(diff_df["method"] == method) & (diff_df["pair"] == pair)
                    & (diff_df["kind"] == kind)].set_index("dataset")
        missing = [d for d in datasets if d not in s.index]
        if missing:
            print(f"  NOTE [{method}/{pair}/{kind}]: no rows for {missing}; left blank")
        s = s.reindex(datasets)
        draw_series(ax, x + off, s["estimate"], s["ci_low"], s["ci_high"],
                    series_style[kind], labels["kind"][kind], cfg)
    ax.set_xticks(x)
    ax.set_xticklabels([labels["dataset"].get(d, d) for d in datasets],
                       rotation=cfg["xtick_rotation"],
                       ha="right" if cfg["xtick_rotation"] else "center",
                       rotation_mode="anchor")
    ax.set_xlim(-0.6, len(datasets) - 0.4)
    ax.grid(cfg["show_grid"], axis="y")
    ax.grid(False, axis="x")


def make_grid(diff_df, method, pairs, datasets, kinds, labels, series_style, cfg,
              grid, figsize, out_path, save):
    nrows, ncols = grid
    if len(pairs) > nrows * ncols:
        raise ValueError(f"{len(pairs)} pairs do not fit a {nrows}x{ncols} grid")
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             sharey=cfg["share_y"], layout="constrained")

    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if i >= len(pairs):
            ax.axis("off")
            continue
        pair = pairs[i]
        draw_panel(ax, diff_df, method, pair, datasets, kinds, labels, series_style, cfg)
        if cfg["show_titles"]:
            ax.set_title(short_pair(pair, labels["centrality"]),
                         fontsize=cfg["title_fontsize"], fontweight=cfg["title_weight"])
        if cfg["panel_letters"]:
            ax.set_title(string.ascii_uppercase[i], loc="left",
                         fontsize=cfg["panel_letter_size"], fontweight="bold")
        # outer labels only
        if c == 0:
            ax.set_ylabel(f"{labels['method'][method]}: test − reference\n(95% CI)")
        elif cfg["share_y"]:
            ax.tick_params(labelleft=False)
        bottom_of_column = (r == nrows - 1) or (i + ncols >= len(pairs))
        if cfg["x_labels_bottom_only"] and not bottom_of_column:
            ax.tick_params(labelbottom=False)

    if cfg["y_limits"] is not None:
        for ax in axes.flat:
            ax.set_ylim(*cfg["y_limits"])

    fig.legend(handles=series_legend_handles(kinds, series_style, labels["kind"], cfg),
               loc="outside lower center", ncol=cfg["legend_ncol"], **cfg["legend_kwargs"])
    save(fig, out_path)
    plt.close(fig)


def main(
    input_dir,
    figure_dir,
    methods,
    datasets,
    kinds: list,
    pairs,
    grid: tuple,
    dataset_labels: dict,
    centrality_labels: dict,
    kind_labels: dict,
    method_labels: dict,
    series_style: dict,
    plot_cfg: dict,
    rc_overrides: dict,
    fig_size,
    file_format: str,
    dpi: int,
    style_module_names: list,
    style_dirs: list,
):
    input_dir, figure_dir = Path(input_dir), Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    apply_style, style_save, fig_sizes, style_src = load_style(style_module_names, style_dirs)
    apply_style()
    matplotlib.rcParams.update(rc_overrides)
    print(f"  style: {style_src}")

    fmt = file_format.lower()

    def save(fig, path):
        with warnings.catch_warnings():
            if fmt not in ("tif", "tiff", "eps"):
                warnings.filterwarnings("ignore", message=r".*not accepted by PLOS.*")
            style_save(fig, path, dpi=dpi, fmt=fmt)

    if isinstance(fig_size, (tuple, list)):
        figsize = tuple(fig_size)
    else:
        figsize = fig_sizes.get(fig_size, _FALLBACK_FIG_SIZE.get(fig_size, (6.83, 4.8)))

    labels = {"dataset": dataset_labels, "centrality": centrality_labels,
              "kind": kind_labels, "method": method_labels}
    ext = "tif" if fmt == "tiff" else fmt

    diff_df = pd.read_csv(input_dir / "within_test_minus_ref.csv")
    ds = datasets or ordered_unique(diff_df["dataset"])
    prs = pairs or ordered_unique(diff_df["pair"])
    n = 0
    for m in (methods or ordered_unique(diff_df["method"])):
        make_grid(diff_df, m, prs, ds, kinds, labels, series_style, plot_cfg, grid, figsize,
                  figure_dir / f"within_test_minus_ref_grid_{m}.{ext}", save)
        n += 1
    print(f"\n{n} figure(s) written to: {figure_dir}")


if __name__ == "__main__":

    # ── Edit these values ─────────────────────────────────────────────────────
    INPUT_DIR = Path("...0_4_revisions/2_experiments/6_bootstrap/std_ver4/within_protein")   # OUTPUT_DIR of bootstrap_within_protein.py
    FIGURE_DIR = INPUT_DIR / "figures_revised_std_ver4"

    METHODS = ["spearman", "pearson"]     # one grid figure per method
    DATASETS = ["arp_iac", "arp_no_iac", "ring_iac", "ring_no_iac", "pcn5", "pcn6", "pcn7"]
    KINDS = ["pooled", "within_centered", "mean_per_protein"]
    # Panel order, filled row by row. None = order in the CSV.
    PAIRS = [
        "StdDegree__StdCloseness", "StdDegree__StdBetweenness", "StdDegree__StdPageRank",
        "StdCloseness__StdBetweenness", "StdCloseness__StdPageRank", "StdBetweenness__StdPageRank",
    ]
    GRID = (2, 3)                         # (rows, columns)

    # Labels (same as within_protein_figures_ver1.py)
    DATASET_LABELS = {
        "arp_iac": "Arp+IAC", "arp_no_iac": "Arp", "ring_iac": "RING+IAC", "ring_no_iac": "RING",
        "pcn5": "PCN5", "pcn6": "PCN6", "pcn7": "PCN7",
    }
    CENTRALITY_LABELS = {
        "StdDegree": "DC", "StdCloseness": "CC", "StdBetweenness": "BC", "StdPageRank": "PR",
    }
    KIND_LABELS = {
        "pooled": "Pooled",
        "within_centered": "Within (centered)",
        "mean_per_protein": "Mean per protein",
    }
    METHOD_LABELS = {"spearman": "Spearman ρ", "pearson": "Pearson r"}

    # Point series (same as within_protein_figures_ver1.py)
    SERIES_STYLE = {
        "pooled":           {"face": "#35302B", "edge": "#35302B", "ecolor": "#35302B", "marker": "o"},
        "within_centered":  {"face": "#8E8278", "edge": "#5E554D", "ecolor": "#8E8278", "marker": "s"},
        "mean_per_protein": {"face": "#F2EEE8", "edge": "#35302B", "ecolor": "#35302B", "marker": "^"},
    }

    PLOT_CFG = {
        # same marker / error-bar design as the single figures, slightly smaller for the cells
        "series_offsets": (-0.22, 0.0, 0.22),
        "marker_size": 4,
        "marker_edge_width": 0.8,
        "errorbar_linewidth": 0.8,
        "cap_size": 2,
        "zero_line_color": "#000000",
        "zero_line_width": 0.6,
        "show_grid": False,
        # grid layout
        "share_y": True,                  # same y scale in all panels
        "y_limits": None,                 # e.g. (-0.3, 0.3) to fix the range by hand
        "x_labels_bottom_only": True,     # dataset names only under the bottom row
        "xtick_rotation": 45,             # 7 labels in a narrow cell need rotating
        "show_titles": True,              # panel title = centrality pair
        "title_fontsize": 9,
        "title_weight": "normal",
        "panel_letters": True,
        "panel_letter_size": 11,
        # shared legend below the grid
        "legend_ncol": 3,
        "legend_kwargs": {"frameon": False, "handletextpad": 0.4, "columnspacing": 1.5},
    }

    # Text sizes inside the grid (PLOS: 8-12 pt)
    RC_OVERRIDES = {
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    }

    FIG_SIZE = (6.83, 4.8)                # inches; PLOS full width is 6.83, max height 8.75
    FILE_FORMAT = "png"                   # "tiff" for PLOS, "png" for quick looks
    DPI = 300

    STYLE_MODULE_NAMES = ["figure_style_ver2", "figure_style"]
    STYLE_DIRS = [Path(__file__).resolve().parent]
    # ──────────────────────────────────────────────────────────────────────────

    main(
        input_dir=INPUT_DIR,
        figure_dir=FIGURE_DIR,
        methods=METHODS,
        datasets=DATASETS,
        kinds=KINDS,
        pairs=PAIRS,
        grid=GRID,
        dataset_labels=DATASET_LABELS,
        centrality_labels=CENTRALITY_LABELS,
        kind_labels=KIND_LABELS,
        method_labels=METHOD_LABELS,
        series_style=SERIES_STYLE,
        plot_cfg=PLOT_CFG,
        rc_overrides=RC_OVERRIDES,
        fig_size=FIG_SIZE,
        file_format=FILE_FORMAT,
        dpi=DPI,
        style_module_names=STYLE_MODULE_NAMES,
        style_dirs=STYLE_DIRS,
    )
