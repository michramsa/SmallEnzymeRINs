# Purpose: Generate fig1_test_functional_breakdown figures.
#
# test_standardized.csv arrives fully pre-computed — StdCloseness, StdDegree,
# StdBetweenness, RCA_zscore, StdPageRank, and functional_type are all already
# columns, so no merge/standardize/label-loading step is needed before plotting.
#
# Every figure below puts StdDegree on the y-axis and the other centrality
# measure on the x-axis:
#   - plot_test_only_functional_breakdown       — StdCloseness (x) vs StdDegree (y)
#   - plot_functional_breakdown_degree_vs_betweenness — StdBetweenness (x) vs StdDegree (y)
#   - plot_functional_breakdown_degree_vs_rca         — RCA_zscore (x) vs StdDegree (y)
#   - plot_functional_breakdown_degree_vs_pagerank    — StdPageRank (x) vs StdDegree (y)
#
# Functional type colour scheme:
#       binding        → #8E8278
#       catalytic      → #35302B
#       dual           → #E8E4DF
#       non-functional → #CCCCCC

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figure_style_ver2 import apply_style, save_fig, FIG_SIZE, FONT

# ── constants ─────────────────────────────────────────────────────────────────

# Colours for the breakdown plot
FUNCTIONAL_TYPE_COLORS = {
    "binding":        "#8E8278",  
    "catalytic":      "#35302B",   
    "dual":           "#E8E4DF",   
    "non_functional": "#CCCCCC",   
}

FUNCTIONAL_TYPE_LABELS = {
    "binding":        "Binding",
    "catalytic":      "Catalytic",
    "dual":           "Dual (binding + catalytic)",
    "non_functional": "Non-functional",
}

# Draw order: non-functional first (background), then each functional type on top
FUNCTIONAL_TYPE_DRAW_ORDER = ["non_functional", "binding", "catalytic", "dual"]

AXIS_LIM = (-4, 4)


# ── axis helpers (carried over) ───────────────────────────────────────────────

def _format_axes_no_title(ax,
                 xlabel="Standardized Closeness Centrality",
                 ylabel="Standardized Degree Centrality"):
    ax.set_xlim(AXIS_LIM)
    ax.set_ylim(AXIS_LIM)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel(xlabel, fontsize=FONT["size_label"])
    ax.set_ylabel(ylabel, fontsize=FONT["size_label"])


# ── new plot function ──────────────────────────────────────────────────────────

def plot_test_only_functional_breakdown(test_df: pd.DataFrame,
                                        output_path: Path) -> None:
    """
    Scatter of the test set coloured by functional residue type:
        binding        → blue   (#4472C4)
        catalytic      → orange (#E8781A)
        dual           → red    (#CC2222)
        non-functional → light grey (#CCCCCC)

    Requires 'functional_type' column (already present in test_standardized.csv).

    Draw order: non-functional rendered first (background), then binding,
    catalytic, and dual on top so the rarest category (dual) is always visible.

    Parameters
    ----------
    test_df     : DataFrame with columns StdCloseness, StdDegree, functional_type.
    output_path : Destination PNG file.
    """
    if "functional_type" not in test_df.columns:
        raise ValueError(
            "'functional_type' column not found. "
            "test_standardized.csv should already contain this column."
        )

    counts = test_df["functional_type"].value_counts()
    print(f"plot_test_only_functional_breakdown received: {len(test_df):,} rows")
    for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
        print(f"  {ftype:20s}: {counts.get(ftype, 0):,}")

    # ── scatter parameters per type ───────────────────────────────────────
    scatter_params = {
        "non_functional": dict(alpha=0.45, s=12, zorder=2, edgecolors="none"),
        "binding":        dict(alpha=0.90, s=30, zorder=3, edgecolors="white", linewidths=0.5),
        "catalytic":      dict(alpha=0.90, s=30, zorder=4, edgecolors="white", linewidths=0.5),
        "dual":           dict(alpha=1.00, s=45, zorder=5, edgecolors="white", linewidths=0.5),
    }

    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])

    for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
        sub = test_df[test_df["functional_type"] == ftype]
        if sub.empty:
            continue
        n   = len(sub)
        ax.scatter(
            sub["StdCloseness"],
            sub["StdDegree"],
            c=FUNCTIONAL_TYPE_COLORS[ftype],
            label=f"{FUNCTIONAL_TYPE_LABELS[ftype]} (n={n:,})",
            **scatter_params[ftype],
        )

    ax.legend(fontsize=FONT["size_legend"], loc="upper left")
    _format_axes_no_title(ax)
    fig.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_functional_breakdown_degree_vs_betweenness(test_df: pd.DataFrame,
                                                    output_path: Path,
                                                    x_lim: tuple = (-4, 10),
                                                    y_lim: tuple = AXIS_LIM) -> None:
    """
    Scatter of the test set: StdBetweenness (x) vs. StdDegree (y),
    coloured by functional residue type. Mirrors plot_test_only_functional_breakdown().

    Parameters
    ----------
    test_df     : DataFrame with columns StdBetweenness, StdDegree, functional_type.
    output_path : Destination PNG file.
    x_lim       : X-axis (StdBetweenness) limits. Default (-4, 10).
    y_lim       : Y-axis (StdDegree) limits. Default (-4, 4).
    """
    if "functional_type" not in test_df.columns:
        raise ValueError(
            "'functional_type' column not found. "
            "test_standardized.csv should already contain this column."
        )
    if "StdBetweenness" not in test_df.columns:
        raise ValueError(
            "'StdBetweenness' column not found. "
            "test_standardized.csv should already contain this column."
        )

    counts = test_df["functional_type"].value_counts()
    print(f"plot_functional_breakdown_degree_vs_betweenness received: {len(test_df):,} rows")
    for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
        print(f"  {ftype:20s}: {counts.get(ftype, 0):,}")

    scatter_params = {
        "non_functional": dict(alpha=0.45, s=12, zorder=2, edgecolors="none"),
        "binding":        dict(alpha=0.90, s=30, zorder=3, edgecolors="white", linewidths=0.5),
        "catalytic":      dict(alpha=0.90, s=30, zorder=4, edgecolors="white", linewidths=0.5),
        "dual":           dict(alpha=1.00, s=45, zorder=5, edgecolors="white", linewidths=0.6),
    }

    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])

    for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
        sub = test_df[test_df["functional_type"] == ftype]
        if sub.empty:
            continue
        n = len(sub)
        ax.scatter(
            sub["StdBetweenness"],
            sub["StdDegree"],
            c=FUNCTIONAL_TYPE_COLORS[ftype],
            label=f"{FUNCTIONAL_TYPE_LABELS[ftype]} (n={n:,})",
            **scatter_params[ftype],
        )

    ax.legend(fontsize=FONT["size_legend"], loc="lower right")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Standardized Betweenness Centrality", fontsize=FONT["size_label"])
    ax.set_ylabel("Standardized Degree Centrality",      fontsize=FONT["size_label"])
    fig.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_functional_breakdown_degree_vs_rca(test_df: pd.DataFrame,
                                            output_path: Path,
                                            x_lim: tuple = (-4, 10),
                                            y_lim: tuple = AXIS_LIM) -> None:
    """
    Scatter of the test set: RCA_zscore (x) vs. StdDegree (y),
    coloured by functional residue type. No CICD threshold line is drawn.

    Draw order: non-functional rendered first (background), then binding,
    catalytic, and dual on top so the rarest category (dual) is always visible.

    Parameters
    ----------
    test_df     : DataFrame with columns RCA_zscore, StdDegree, functional_type.
    output_path : Destination PNG file.
    x_lim       : X-axis (RCA_zscore) limits. Default (-4, 10).
    y_lim       : Y-axis (StdDegree) limits. Default (-4, 4).
    """
    if "functional_type" not in test_df.columns:
        raise ValueError(
            "'functional_type' column not found. "
            "test_standardized.csv should already contain this column."
        )
    if "RCA_zscore" not in test_df.columns:
        raise ValueError(
            "'RCA_zscore' column not found. "
            "test_standardized.csv should already contain this column."
        )

    counts = test_df["functional_type"].value_counts()
    print(f"plot_functional_breakdown_degree_vs_rca received: {len(test_df):,} rows")
    for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
        print(f"  {ftype:20s}: {counts.get(ftype, 0):,}")

    scatter_params = {
        "non_functional": dict(alpha=0.45, s=12, zorder=2, edgecolors="none"),
        "binding":        dict(alpha=0.90, s=30, zorder=3, edgecolors="white", linewidths=0.5),
        "catalytic":      dict(alpha=0.90, s=30, zorder=4, edgecolors="white", linewidths=0.5),
        "dual":           dict(alpha=1.00, s=45, zorder=5, edgecolors="white", linewidths=0.6),
    }

    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])

    for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
        sub = test_df[test_df["functional_type"] == ftype].dropna(
            subset=["RCA_zscore", "StdDegree"]
        )
        if sub.empty:
            continue
        n = len(sub)
        ax.scatter(
            sub["RCA_zscore"],
            sub["StdDegree"],
            c=FUNCTIONAL_TYPE_COLORS[ftype],
            label=f"{FUNCTIONAL_TYPE_LABELS[ftype]} (n={n:,})",
            **scatter_params[ftype],
        )

    ax.legend(fontsize=FONT["size_legend"], loc="lower right")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel("RCA Z-score",                    fontsize=FONT["size_label"])
    ax.set_ylabel("Standardized Degree Centrality", fontsize=FONT["size_label"])
    fig.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_functional_breakdown_degree_vs_pagerank(test_df: pd.DataFrame,
                                                  output_path: Path,
                                                  x_lim: tuple = AXIS_LIM,
                                                  y_lim: tuple = AXIS_LIM) -> None:
    """
    Scatter of the test set: StdPageRank (x) vs. StdDegree (y),
    coloured by functional residue type. Mirrors plot_functional_breakdown_degree_vs_betweenness()
    and plot_functional_breakdown_degree_vs_rca(), but for PageRank.

    No threshold line is drawn. The optimal StdPageRank threshold is
    determined empirically by the LOO jackknife procedure, not by a fixed
    literature value.

    Draw order: non-functional rendered first (background), then binding,
    catalytic, and dual on top so the rarest category (dual) is always visible.

    Parameters
    ----------
    test_df     : DataFrame with columns StdPageRank, StdDegree, functional_type.
    output_path : Destination PNG file.
    x_lim       : X-axis (StdPageRank) limits. Default (-4, 4).
    y_lim       : Y-axis (StdDegree) limits. Default (-4, 4).
    """
    if "functional_type" not in test_df.columns:
        raise ValueError(
            "'functional_type' column not found. "
            "test_standardized.csv should already contain this column."
        )
    if "StdPageRank" not in test_df.columns:
        raise ValueError(
            "'StdPageRank' column not found. "
            "test_standardized.csv should already contain this column."
        )

    counts = test_df["functional_type"].value_counts()
    print(f"plot_functional_breakdown_degree_vs_pagerank received: {len(test_df):,} rows")
    for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
        print(f"  {ftype:20s}: {counts.get(ftype, 0):,}")

    scatter_params = {
        "non_functional": dict(alpha=0.45, s=12, zorder=2, edgecolors="none"),
        "binding":        dict(alpha=0.90, s=30, zorder=3, edgecolors="white", linewidths=0.5),
        "catalytic":      dict(alpha=0.90, s=30, zorder=4, edgecolors="white", linewidths=0.5),
        "dual":           dict(alpha=1.00, s=45, zorder=5, edgecolors="white", linewidths=0.6),
    }

    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])

    for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
        sub = test_df[test_df["functional_type"] == ftype].dropna(
            subset=["StdPageRank", "StdDegree"]
        )
        if sub.empty:
            continue
        n = len(sub)
        ax.scatter(
            sub["StdPageRank"],
            sub["StdDegree"],
            c=FUNCTIONAL_TYPE_COLORS[ftype],
            label=f"{FUNCTIONAL_TYPE_LABELS[ftype]} (n={n:,})",
            **scatter_params[ftype],
        )

    ax.legend(fontsize=FONT["size_legend"], loc="upper left")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Standardized PageRank",           fontsize=FONT["size_label"])
    ax.set_ylabel("Standardized Degree Centrality",  fontsize=FONT["size_label"])
    fig.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    ver_dir  = FIGS_DIR / "breakdown"
    test_csv = FIGS_DIR / "test_standardized.csv"

    # 1. Load standardized test data. Already contains RCA_deltaL, RCA_zscore,
    #    IsCICD, PageRank, StdPageRank, StdCloseness, StdDegree, StdBetweenness,
    #    RSA_category, functional_type, and is_functional.
    test_df = pd.read_csv(test_csv)

    # 2. Closeness vs. Degree breakdown (original)
    plot_test_only_functional_breakdown(
        test_df,
        ver_dir / "fig1_test_functional_breakdown.png",
    )

    # 3. StdDegree (y) vs. StdBetweenness (x) breakdown
    plot_functional_breakdown_degree_vs_betweenness(
        test_df,
        ver_dir / "fig1_test_funct_breakdown_deg_vs_bet.png",
        x_lim=(-4, 10),
        y_lim=(-4, 4),
    )

    # 4. StdDegree (y) vs. RCA z-score (x) breakdown (no CICD threshold line)
    plot_functional_breakdown_degree_vs_rca(
        test_df,
        ver_dir / "fig1_test_funct_breakdown_deg_vs_rca.png",
        x_lim=(-4, 10),
        y_lim=(-4, 4),
    )

    # 5. StdDegree (y) vs. StdPageRank (x) breakdown
    plot_functional_breakdown_degree_vs_pagerank(
        test_df,
        ver_dir / "fig1_test_funct_breakdown_deg_vs_page.png",
    )

    print("\nDone.")
