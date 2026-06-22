"""
plot_stdcentrality_frequency_by_functional_type.py
===================================================
Drop-in addition for 0_4_fig1_recreation_ver6.py.

Provides one new function:

    plot_stdcentrality_frequency_by_functional_type(
        df, metric, output_path, ...
    )

The figure is a 1 × 4 panel grid — one subplot per functional type
(non-functional, catalytic, binding, dual) — each showing one RSA-category
line per panel, styled identically to the existing plot_stdcloseness_frequency /
plot_stddegree_frequency / plot_stdbetweenness_frequency functions.

Supported metrics
-----------------
  "StdCloseness"   — x-range −4 to 4
  "StdDegree"      — x-range −4 to 4
  "StdBetweenness" — x-range −4 to 4
  "RCA_zscore"     — x-range −4 to 10  (CICD threshold line drawn at z = 2)

The subplot title for each panel includes the functional-type label and its
residue count (n).  A single shared legend appears at the bottom of the figure.

Usage (inside main or __main__ block of ver6):
----------------------------------------------
    from plot_stdcentrality_frequency_by_functional_type import (
        plot_stdcentrality_frequency_by_functional_type,
    )

    for metric in ["StdCloseness", "StdDegree", "StdBetweenness", "RCA_zscore"]:
        plot_stdcentrality_frequency_by_functional_type(
            test_df,
            metric       = metric,
            output_path  = output_dir / f"fig_{metric.lower()}_freq_by_functype.png",
            y_max        = 1.0,
        )

Requirements
------------
The DataFrame must already contain:
  - the requested metric column
  - RSA_category   (from assign_rsa_category())
  - functional_type (from add_functional_type())
  - RCA_zscore only required when metric == "RCA_zscore"
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from figure_style_ver2 import (
    apply_style, save_fig,
    FIG_SIZE, FONT, LAYOUT,
)

# ── shared axis / bin constants (mirrors ver6) ────────────────────────────────
_DIST_X_MIN  = -4.0
_DIST_X_MAX  =  4.0
_DIST_X_STEP =  0.5
_DIST_Y_STEP =  0.1

# Per-metric x-axis configuration
# Each entry: (x_min, x_max, x_step, short_label)
_METRIC_CONFIG = {
    "StdCloseness":   (-4.0,  4.0, 0.5, "Std CC"),
    "StdDegree":      (-4.0,  4.0, 0.5, "Std DC"),
    "StdBetweenness": (-4.0,  4.0, 0.5, "Std BC"),
    "RCA_zscore":     (-4.0, 10.0, 1.0, "RCA Z-score"),
}

# CICD threshold drawn only for RCA_zscore (del Sol et al. 2006)
_RCA_CICD_THRESHOLD = 2.0

# RSA colours — matches RSA_COLORS from the functional breakdown scripts
_DIST_RSA_COLORS = {
    "exposed": "#35302B",   # dark brown
    "core":    "#E8E4DF",   # off-white / light warm
    "other":   "#8E8278",   # medium warm grey
}
_DIST_RSA_ORDER  = ["exposed", "core", "other"]
_DIST_RSA_LABELS = {
    "exposed": "Highly exposed (RSA > 0.5)",
    "core":    "Core (RSA = 0.0)",
    "other":   "Partial Exposed (0 < RSA ≤ 0.5)",
}

# Functional-type draw order, colours, and labels
_FUNC_DRAW_ORDER = ["non_functional", "catalytic", "binding", "dual"]

_FUNC_LABELS = {
    "non_functional": "Non-functional",
    "catalytic":      "Catalytic",
    "binding":        "Binding",
    "dual":           "Dual",
}


# ── helper ────────────────────────────────────────────────────────────────────

def _compute_freq_distribution(
    series:  pd.Series,
    x_min:   float,
    x_max:   float,
    x_step:  float,
) -> tuple:
    """
    Bin *series* and return (bin_centres, percentage_frequencies).
    Each RSA category is normalised independently so all lines sum to 1.
    """
    bin_edges = np.arange(x_min, x_max + x_step, x_step)
    counts, _ = np.histogram(series.dropna(), bins=bin_edges)
    total     = counts.sum()
    pct_freq  = (counts / total) if total > 0 else np.zeros_like(counts, dtype=float)
    centres   = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centres, pct_freq


def _sparse_tick_labels(ticks: np.ndarray) -> list:
    """
    Return a label list that shows only the leftmost, centre, and rightmost
    tick values — all others are blank strings.  This avoids over-crowding on
    narrow panels.

    The centre tick is the one closest to the midpoint of the tick array.
    """
    n = len(ticks)
    if n == 0:
        return []
    if n == 1:
        return [f"{ticks[0]:.4g}"]
    if n == 2:
        return [f"{ticks[0]:.4g}", f"{ticks[-1]:.4g}"]

    mid_idx = n // 2
    labels  = [""] * n
    labels[0]       = f"{ticks[0]:.4g}"
    labels[mid_idx] = f"{ticks[mid_idx]:.4g}"
    labels[-1]      = f"{ticks[-1]:.4g}"
    return labels


# ── main function ─────────────────────────────────────────────────────────────

def plot_stdcentrality_frequency_by_functional_type(
    df:          pd.DataFrame,
    metric:      str,
    output_path: Path,
    x_min:       float = None,
    x_max:       float = None,
    x_step:      float = None,
    y_min:       float = 0.0,
    y_max:       float = None,
    y_step:      float = _DIST_Y_STEP,
    func_order:  list  = None,
) -> None:
    """
    1 × 4 panel figure: one subplot per functional type, each showing the
    percentage-frequency distribution of *metric* coloured by RSA category.

    Parameters
    ----------
    df          : pd.DataFrame
        Must contain columns: *metric*, 'RSA_category', 'functional_type'.
    metric      : str
        One of 'StdCloseness', 'StdDegree', 'StdBetweenness', 'RCA_zscore'.
    output_path : Path
        Destination file (PNG / TIFF / EPS).
    x_min, x_max, x_step : float, optional
        Override the default x-axis range / bin width for the chosen metric.
    y_min       : float
        Y-axis lower bound (default 0.0).
    y_max       : float
        Y-axis upper bound.  If None, inferred from the data across all panels
        so every subplot shares the same y-scale.
    y_step      : float
        Y-axis tick interval (default 0.1).
    func_order  : list, optional
        Order of functional-type panels.  Defaults to
        ['non_functional', 'catalytic', 'binding', 'dual'].
    """
    # ── validate metric ───────────────────────────────────────────────────────
    if metric not in _METRIC_CONFIG:
        raise ValueError(
            f"Unsupported metric '{metric}'. "
            f"Choose from: {list(_METRIC_CONFIG.keys())}"
        )

    cfg_x_min, cfg_x_max, cfg_x_step, xlabel = _METRIC_CONFIG[metric]
    if x_min  is None: x_min  = cfg_x_min
    if x_max  is None: x_max  = cfg_x_max
    if x_step is None: x_step = cfg_x_step

    # ── validate required columns ─────────────────────────────────────────────
    for required_col in [metric, "RSA_category", "functional_type"]:
        if required_col not in df.columns:
            raise ValueError(
                f"Column '{required_col}' not found in DataFrame. "
                "Ensure all standardisation and labelling steps have been run."
            )

    if func_order is None:
        func_order = _FUNC_DRAW_ORDER

    # ── auto-detect y_max across all panels if not supplied ───────────────────
    if y_max is None:
        global_max = 0.0
        for ftype in func_order:
            sub = df[df["functional_type"] == ftype]
            for cat in _DIST_RSA_ORDER:
                series = sub[sub["RSA_category"] == cat][metric].dropna()
                if series.empty:
                    continue
                _, pct = _compute_freq_distribution(series, x_min, x_max, x_step)
                if pct.max() > global_max:
                    global_max = pct.max()
        y_max = (int(global_max / y_step) + 2) * y_step

    # ── layout ────────────────────────────────────────────────────────────────
    apply_style()

    n_panels  = len(func_order)
    fig_w     = max(FIG_SIZE["full_width"][0], n_panels * 1.8)
    fig_h     = FIG_SIZE["full_width"][1] + 0.8   # extra room for bottom legend
    fig, axes = plt.subplots(
        nrows   = 1,
        ncols   = n_panels,
        figsize = (fig_w, fig_h),
        sharey  = True,
    )

    # build tick arrays once
    x_ticks = np.arange(x_min, x_max + x_step / 2, x_step)
    y_ticks = np.arange(y_min, y_max + y_step  / 2, y_step)
    x_tick_labels = _sparse_tick_labels(x_ticks)

    for col_idx, ftype in enumerate(func_order):
        ax  = axes[col_idx]
        sub = df[df["functional_type"] == ftype]
        n   = len(sub[metric].dropna())

        ax.set_facecolor("white")
        ax.grid(False)

        # ── RSA-category lines ────────────────────────────────────────────
        for cat in _DIST_RSA_ORDER:
            series = sub[sub["RSA_category"] == cat][metric].dropna()
            if series.empty:
                continue
            centres, pct = _compute_freq_distribution(series, x_min, x_max, x_step)
            ax.plot(
                centres, pct,
                color     = _DIST_RSA_COLORS[cat],
                linewidth = LAYOUT["line_width"],
            )

        # ── CICD threshold line (RCA only) ────────────────────────────────
        if metric == "RCA_zscore":
            ax.axvline(
                _RCA_CICD_THRESHOLD,
                color     = "black",
                linewidth = 0.8,
                linestyle = ":",
                zorder    = 3,
            )

        # ── panel title: functional-type label + n ────────────────────────
        ax.set_title(
            f"{_FUNC_LABELS[ftype]}\n(n = {n:,})",
            fontsize   = FONT["size_title"],
            fontweight = "bold",
            color      = "black",
            pad        = 6,
        )

        # ── x-axis ───────────────────────────────────────────────────────
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            x_tick_labels,
            rotation = 0,          # no rotation needed with sparse labels
            ha       = "center",
            fontsize = FONT["size_tick"],
        )
        ax.set_xlabel(xlabel, fontsize=FONT["size_label"])

        # ── y-axis: label only on leftmost panel ──────────────────────────
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        if col_idx == 0:
            ax.set_ylabel("Frequency (%)", fontsize=FONT["size_label"])
        else:
            ax.tick_params(labelleft=False)

    # ── shared RSA legend below all panels ───────────────────────────────────
    legend_handles = [
        mlines.Line2D(
            [], [],
            color     = _DIST_RSA_COLORS[cat],
            linewidth = LAYOUT["line_width"],
            label     = _DIST_RSA_LABELS[cat],
        )
        for cat in _DIST_RSA_ORDER
    ]

    # Add CICD marker to legend when plotting RCA
    if metric == "RCA_zscore":
        legend_handles.append(
            mlines.Line2D(
                [], [],
                color     = "black",
                linewidth = 0.8,
                linestyle = ":",
                label     = f"CICD threshold (z = {_RCA_CICD_THRESHOLD:.0f})",
            )
        )

    fig.legend(
        handles        = legend_handles,
        loc            = "lower center",
        ncol           = len(legend_handles),
        fontsize       = FONT["size_legend"],
        frameon        = False,
        bbox_to_anchor = (0.5, -0.01),
    )

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── __main__ entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── configurable parameters — edit these directly ─────────────────────────
    standardized_csv = Path(
        "path/to/test_standardized.csv"     # <- change me
    )
    output_dir = Path(
        "path/to/output/figures"            # <- change me
    )

    # Metrics to plot — remove any that are absent from your CSV
    metrics = ["StdCloseness", "StdDegree", "StdBetweenness", "RCA_zscore"]

    y_max  = 1.0   # shared y-axis upper bound; set None for auto-detect
    y_step = 0.1

    # ── load ──────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(standardized_csv)

    print(f"Loaded {len(df):,} rows from {standardized_csv}")
    print(f"Columns: {df.columns.tolist()}")

    # ── produce one figure per metric ─────────────────────────────────────────
    for metric in metrics:
        if metric not in df.columns:
            print(f"  WARNING: '{metric}' not found in CSV — skipping.")
            continue

        out_path = output_dir / f"fig_{metric.lower()}_freq_by_functype.png"
        plot_stdcentrality_frequency_by_functional_type(
            df          = df,
            metric      = metric,
            output_path = out_path,
            y_max       = y_max,
            y_step      = y_step,
        )

    print("\nDone.")
