# figures.py
#
# Purpose: Recreate Amitai et al. Figure 1 (standardized closeness vs. standardized
# degree) using the old PISCES set as the reference population, with the small
# enzyme test set overlaid, plus a family of related scatter / frequency / boxplot
# figures built on the same standardized centrality data.
#
# Functional-type coloring, used in the categorical scatter plots:
#   non_functional, binding, catalytic, dual (drawn in that order so rarer
#   categories stay visible on top).

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

from figure_style_ver2 import apply_style, save_fig, FIG_SIZE, FONT, LAYOUT, DPI, COLORS
from standardization import (
    standardize_closeness_per_protein,
    compute_degree_stats_by_aa,
    standardize_degree_by_aa,
    standardize_betweenness_per_protein,
    rename_pagerank_zscore,
    assign_rsa_category,
)

# 20 canonical amino acids — only these are included in analysis
CANONICAL_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}

# ── RSA category styling ────────────────────────────────────────────────────

RSA_COLORS = {
    "exposed": "#35302B",
    "core":    "#E8E4DF",
    "other":   "#8E8278",
    "unknown": "#888888",
}

RSA_ORDER = ["other", "core", "exposed", "unknown"]   # draw exposed on top

RSA_LABELS = {
    "exposed": "Exposed (RSA > 0.5)",
    "core":    "Core (RSA = 0)",
    "other":   "Partial Exposed (0 < RSA ≤ 0.5)",
    "unknown": "RSA not available",
}

OVERLAY_COLORS = {
    "exposed": "#35302B",
    "core":    "#E8E4DF",
    "other":   "#8E8278",
    "unknown": "#8D8D8D",
}

AXIS_LIM   = (-4, 4)
AXIS_LIM_2 = (-4, 10)

# ── functional-type styling (shared with biolip_residue_labels.py output) ──

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

# Draw order: non-functional background first, rarest (dual) on top
FUNCTIONAL_TYPE_DRAW_ORDER = ["non_functional", "binding", "catalytic", "dual"]

# ── frequency-distribution styling ──────────────────────────────────────────

_DIST_X_MIN   = -4.0
_DIST_X_MAX   =  4.0
_DIST_X_STEP  =  0.5
_DIST_Y_STEP  =  0.1

_DIST_RSA_COLORS = {
    "exposed": "#E8781A",   # orange
    "core":    "#2E8B2E",   # green
    "other":   "#4472C4",   # blue
}

_DIST_RSA_ORDER  = ["exposed", "core", "other"]
_DIST_RSA_LABELS = {
    "exposed": "Highly exposed (RSA > 0.5)",
    "core":    "Core (RSA = 0.0)",
    "other":   "Other (0 < RSA ≤ 0.5)",
}


# ── data loading ──────────────────────────────────────────────────────────────

def load_centrality_dir(centrality_dir: Path,
                        exclude_pdb_ids: set = None) -> pd.DataFrame:
    """
    Load all *_centrality.csv files from a directory into one concatenated DataFrame.
    Filters to canonical amino acids only.

    RSA-empirical is expected to already be present as a column in each
    *_centrality.csv (written upstream by merge_rsa_into_centrality() in
    move_SASA_files.py) — no SASA-file join is performed here.

    Parameters
    ----------
    centrality_dir  : Path  Directory containing *_centrality.csv files.
    exclude_pdb_ids : set of str, optional
        PDB IDs to skip entirely before reading their CSV files.
        Matching is case-insensitive (e.g. {'7ara', '4q95'} or {'7ARA', '4Q95'}).

    Returns
    -------
    pd.DataFrame — concatenated centrality data with a 'pdb_id' column added.
    """
    centrality_dir  = Path(centrality_dir)
    exclude_pdb_ids = {p.lower() for p in exclude_pdb_ids} if exclude_pdb_ids else set()
    frames = []

    for csv_path in sorted(centrality_dir.glob("*_centrality.csv")):
        pdb_id = csv_path.stem.replace("_centrality", "")

        if pdb_id.lower() in exclude_pdb_ids:
            print(f"  Skipping excluded PDB: {pdb_id}")
            continue

        df = pd.read_csv(csv_path)
        df["pdb_id"] = pdb_id

        # keep only canonical amino acids
        df = df[df["Residue"].isin(CANONICAL_AA)].copy()

        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No centrality CSVs found in {centrality_dir}")

    combined = pd.concat(frames, ignore_index=True)

    # validate required columns
    for col in ["Degree", "ClosenessCentrality", "Residue"]:
        if col not in combined.columns:
            raise ValueError(
                f"Required column '{col}' not found. "
                f"Columns present: {list(combined.columns)}"
            )

    return combined


def parse_fasta_chains(fasta_path: Path) -> set:
    """
    Return a set of (pdb_id_uppercase, chain) tuples from a FASTA file.
    e.g. '>207lA' -> ('207L', 'A')
    """
    chains = set()
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                raw    = line[1:].strip().split()[0]
                pdb_id = raw[:4].upper()
                chain  = raw[4:]
                chains.add((pdb_id, chain))
    return chains


def standardize_if_missing(df: pd.DataFrame, aa_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Apply each standardization step only if its output column is not already
    present in df. Useful for the test set, whose centrality CSVs may already
    have Std* columns precomputed (e.g. by append_standardized_to_centrality_dir()
    in standardization.py) — in that case this is a no-op pass-through that
    just verifies the columns are there.

    StdDegree, if computed, always uses aa_stats from the reference population
    (never recomputed from the test set itself), consistent with the rest of
    this pipeline.

    Parameters
    ----------
    df       : pd.DataFrame  Test-set centrality data.
    aa_stats : pd.DataFrame  Output of compute_degree_stats_by_aa() on the
                             reference set. Only used if StdDegree is missing.

    Returns
    -------
    pd.DataFrame with StdCloseness, StdDegree, StdBetweenness, StdPageRank
    (where derivable), and RSA_category present.
    """
    if "StdCloseness" not in df.columns:
        print("  StdCloseness not found — computing.")
        df = standardize_closeness_per_protein(df)
    else:
        print("  StdCloseness already present — skipping.")

    if "StdDegree" not in df.columns:
        print("  StdDegree not found — computing from reference aa_stats.")
        df = standardize_degree_by_aa(df, aa_stats)
    else:
        print("  StdDegree already present — skipping.")

    if "StdBetweenness" not in df.columns:
        print("  StdBetweenness not found — computing.")
        df = standardize_betweenness_per_protein(df)
    else:
        print("  StdBetweenness already present — skipping.")

    if "StdPageRank" not in df.columns and "PageRank_zscore" in df.columns:
        print("  StdPageRank not found — renaming from PageRank_zscore.")
        df = rename_pagerank_zscore(df)
    elif "StdPageRank" in df.columns:
        print("  StdPageRank already present — skipping.")

    if "RSA_category" not in df.columns:
        print("  RSA_category not found — computing.")
        df = assign_rsa_category(df)
    else:
        print("  RSA_category already present — skipping.")

    return df


def load_functional_labels(labels_csv: Path) -> pd.DataFrame:
    """
    Load biolip_residue_labels.csv (built by biolip_residue_labels.py).

    Expected columns: pdb_id (lowercase), chain, seq_num (int),
    label ('catalytic' | 'binding' | 'dual').

    Returns
    -------
    pd.DataFrame with columns: pdb_id, chain, seq_num, label.
    """
    labels_csv = Path(labels_csv)
    df = pd.read_csv(labels_csv)
    df["pdb_id"] = df["pdb_id"].str.lower()
    df["chain"]  = df["chain"].astype(str).str.strip()
    return df


def add_functional_labels(df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join functional-residue labels onto a residue-level DataFrame, adding both
    a binary 'is_functional' column and a 4-category 'functional_type' column.

    Join key: pdb_id (lowercase) + Chain + Residue number  <->
              pdb_id (lowercase) + chain + seq_num.

    Residues absent from label_df get is_functional=False and
    functional_type='non_functional'.

    Parameters
    ----------
    df       : pd.DataFrame  Must contain 'pdb_id', 'Chain', 'Residue number'.
    label_df : pd.DataFrame  Output of load_functional_labels().

    Returns
    -------
    pd.DataFrame with 'is_functional' and 'functional_type' columns added.
    """
    df = df.copy()
    df["pdb_id_lower"]   = df["pdb_id"].str.lower()
    df["Residue number"] = df["Residue number"].astype(int)

    merged = df.merge(
        label_df[["pdb_id", "chain", "seq_num", "label"]],
        left_on  = ["pdb_id_lower", "Chain", "Residue number"],
        right_on = ["pdb_id",       "chain", "seq_num"],
        how      = "left",
    )

    df["functional_type"] = merged["label"].fillna("non_functional").values
    df["is_functional"]   = df["functional_type"] != "non_functional"
    df.drop(columns=["pdb_id_lower"], inplace=True)

    counts = df["functional_type"].value_counts()
    print("\n── Functional type breakdown ─────────────────────────────────")
    for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
        print(f"  {ftype:20s}: {counts.get(ftype, 0):,}")
    print("─────────────────────────────────────────────────────────────\n")

    return df


# ── plotting: shared helpers ────────────────────────────────────────────────

def _format_axes(ax, xlabel, ylabel, title=None, axis_lim=AXIS_LIM):
    """
    Apply shared axis formatting: dashed zero lines, axis labels, and an
    optional title. If axis_lim is given, applies it to both x and y
    (matching the old square-axis behavior); pass axis_lim=None to leave
    limits untouched (e.g. when the caller sets x_lim/y_lim separately).
    """
    if axis_lim is not None:
        ax.set_xlim(axis_lim)
        ax.set_ylim(axis_lim)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel(xlabel, fontsize=FONT["size_label"])
    ax.set_ylabel(ylabel, fontsize=FONT["size_label"])
    if title is not None:
        ax.set_title(title, fontsize=FONT["size_title"])


def _scatter_by_rsa(ax, df, color_map, x_col, y_col,
                    alpha=0.4, s=6, zorder=2, edgecolors="white"):
    """
    Scatter helper: x_col (x) vs. y_col (y), colored by RSA category.
    Generic replacement for the old per-column-pair _scatter_by_rsa_* helpers.
    """
    for cat in RSA_ORDER:
        sub = df[df["RSA_category"] == cat]
        if sub.empty:
            continue
        ax.scatter(
            sub[x_col], sub[y_col],
            c=color_map[cat], alpha=alpha, s=s,
            zorder=zorder, edgecolors=edgecolors, label=None,
        )


# ── plotting: RSA-category scatter plots ────────────────────────────────────

def plot_test_only_scatter(test_df: pd.DataFrame,
                           output_path: Path,
                           x_col: str,
                           y_col: str,
                           x_label: str,
                           y_label: str,
                           x_lim: tuple = AXIS_LIM,
                           y_lim: tuple = AXIS_LIM_2,
                           legend_loc: str = "upper left",
                           ref_df: pd.DataFrame = None,
                           threshold_line: float = None):
    """
    Scatter of the test set: x_col (x) vs. y_col (y), colored by RSA category.

    Generic replacement for the old plot_test_only(), plot_reference_only(),
    and the family of plot_test_only_<metric>_vs_<metric>() functions.

    Parameters
    ----------
    test_df       : pd.DataFrame  Must contain x_col, y_col, and 'RSA_category'.
    output_path   : Path          Destination file (PNG / TIFF).
    x_col, y_col  : str           Column names to plot.
    x_label,
    y_label       : str           Axis labels.
    x_lim, y_lim  : tuple         Axis limits.
    legend_loc    : str           Legend placement.
    ref_df        : pd.DataFrame, optional
        If provided, drawn first as a muted background layer (same role as the
        old plot_reference_only(), without the foreground test set). If None,
        only test_df is plotted (matches the old plot_test_only()).
    threshold_line : float, optional
        If provided, draws a horizontal dotted line at this y-value (e.g. the
        CICD threshold at y=2 used by the RCA-vs-closeness plot).
    """
    for col in (x_col, y_col, "RSA_category"):
        if col not in test_df.columns:
            raise ValueError(
                f"'{col}' column not found in test_df. "
                "Ensure the DataFrame has been standardized and "
                "assign_rsa_category() has been run."
            )

    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])

    if ref_df is not None:
        _scatter_by_rsa(ax, ref_df, RSA_COLORS, x_col, y_col, alpha=0.35, s=5)

    _scatter_by_rsa(
        ax, test_df, OVERLAY_COLORS, x_col, y_col,
        alpha=0.70, s=18, zorder=2, edgecolors="white",
    )

    patches = [
        mpatches.Patch(color=OVERLAY_COLORS[cat], label=RSA_LABELS[cat])
        for cat in RSA_ORDER
        if not test_df[test_df["RSA_category"] == cat].empty
    ]
    ax.legend(handles=patches, fontsize=FONT["size_legend"], loc=legend_loc)

    if threshold_line is not None:
        ax.axhline(threshold_line, color="black", linewidth=0.8, linestyle=":",
                  label=f"CICD threshold (z = {threshold_line})")

    _format_axes(ax, x_label, y_label, axis_lim=None)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    fig.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_overlay_scatter(ref_df: pd.DataFrame,
                         test_df: pd.DataFrame,
                         output_path: Path,
                         x_col: str,
                         y_col: str,
                         x_label: str,
                         y_label: str,
                         x_lim: tuple = AXIS_LIM,
                         y_lim: tuple = AXIS_LIM_2,
                         legend_loc: str = "upper left",
                         threshold_line: float = None):
    """
    Overlay scatter: x_col (x) vs. y_col (y). Reference set as muted
    background, test set as prominent foreground.

    Generic replacement for the old plot_overlay() and the family of
    plot_overlay_<metric>_vs_<metric>() functions.

    Parameters
    ----------
    ref_df, test_df : pd.DataFrame  Must each contain x_col, y_col, and
                                    'RSA_category'.
    output_path     : Path          Destination file.
    x_col, y_col    : str           Column names to plot.
    x_label, y_label: str           Axis labels.
    x_lim, y_lim    : tuple         Axis limits.
    legend_loc      : str           Legend placement.
    threshold_line  : float, optional
        If provided, draws a horizontal dotted line at this y-value (e.g. the
        CICD threshold at y=2 used by the RCA-vs-closeness overlay plot).
    """
    for col in (x_col, y_col, "RSA_category"):
        for label, df in [("ref_df", ref_df), ("test_df", test_df)]:
            if col not in df.columns:
                raise ValueError(
                    f"'{col}' column not found in {label}. "
                    "Ensure both DataFrames have been standardized and "
                    "assign_rsa_category() has been run."
                )

    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])

    # reference set — muted background
    _scatter_by_rsa(ax, ref_df, RSA_COLORS, x_col, y_col, alpha=0.20, s=4, zorder=2)
    # test set — prominent foreground
    _scatter_by_rsa(
        ax, test_df, OVERLAY_COLORS, x_col, y_col,
        alpha=0.70, s=18, zorder=3, edgecolors="white",
    )

    ref_patches = [
        mpatches.Patch(color=RSA_COLORS[cat], alpha=0.5,
                       label=f"Ref: {RSA_LABELS[cat]}")
        for cat in RSA_ORDER
        if not ref_df[ref_df["RSA_category"] == cat].empty
    ]
    test_patches = [
        mpatches.Patch(color=OVERLAY_COLORS[cat],
                       label=f"Test: {RSA_LABELS[cat]}")
        for cat in RSA_ORDER
        if not test_df[test_df["RSA_category"] == cat].empty
    ]
    ax.legend(
        handles=ref_patches + test_patches,
        fontsize=FONT["size_legend"],
        loc=legend_loc,
        title="Reference set  /  Small enzyme test set",
        title_fontsize=FONT["size_legend"],
    )

    if threshold_line is not None:
        ax.axhline(threshold_line, color="black", linewidth=0.8, linestyle=":",
                  label=f"CICD threshold (z = {threshold_line})")

    _format_axes(ax, x_label, y_label, axis_lim=None)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    fig.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")

# ── plotting: frequency distributions ───────────────────────────────────────

def _compute_freq_distribution(series: pd.Series,
                                x_min: float = _DIST_X_MIN,
                                x_max: float = _DIST_X_MAX,
                                x_step: float = _DIST_X_STEP) -> tuple:
    """
    Bin a numeric series into histogram bins and return bin centres with
    percentage frequencies (each category sums to 100 independently).
    """
    bin_edges = np.arange(x_min, x_max + x_step, x_step)
    counts, _ = np.histogram(series.dropna(), bins=bin_edges)
    total      = counts.sum()
    pct_freq   = (counts / total) if total > 0 else np.zeros_like(counts, dtype=float)
    centres    = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centres, pct_freq


def plot_std_frequency(df: pd.DataFrame,
                       output_path: Path,
                       value_col: str,
                       x_label: str,
                       legend_loc: str = "upper left",
                       x_min: float = _DIST_X_MIN,
                       x_max: float = _DIST_X_MAX,
                       x_step: float = _DIST_X_STEP,
                       y_min: float = 0.0,
                       y_max: float = None,
                       y_step: float = _DIST_Y_STEP):
    """
    Line plot of percentage-frequency vs. value_col, with one line per RSA
    category (exposed = orange, core = green, other = blue).

    Generic replacement for plot_stdcloseness_frequency(),
    plot_stddegree_frequency(), plot_stdbetweenness_frequency(), and
    plot_stdpagerank_frequency().

    Parameters
    ----------
    df                    : pd.DataFrame  Must contain value_col and 'RSA_category'.
    output_path           : Path          Destination file (PNG / TIFF / PDF).
    value_col             : str           Column to bin and plot.
    x_label               : str           X-axis label.
    legend_loc            : str           Legend placement.
    x_min, x_max, x_step  : float         X-axis range and bin width.
    y_min                 : float         Y-axis lower bound (default 0.0).
    y_max                 : float         Y-axis upper bound (default: auto from data).
    y_step                : float         Y-axis tick interval (as % frequency).
    """
    if value_col not in df.columns:
        raise ValueError(
            f"'{value_col}' column not found. Ensure the DataFrame has been "
            "standardized before calling this function."
        )
    if "RSA_category" not in df.columns:
        raise ValueError(
            "'RSA_category' column not found. Run assign_rsa_category() first."
        )

    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["half_page"])

    ax.set_facecolor("white")
    ax.grid(False)

    for cat in _DIST_RSA_ORDER:
        subset = df[df["RSA_category"] == cat][value_col]
        if subset.dropna().empty:
            continue
        centres, pct = _compute_freq_distribution(subset, x_min, x_max, x_step)
        ax.plot(
            centres, pct,
            color     = _DIST_RSA_COLORS[cat],
            linewidth = LAYOUT["line_width"],
            label     = _DIST_RSA_LABELS[cat],
        )

    # X-axis
    x_ticks = np.arange(x_min, x_max + x_step / 2, x_step)
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Y-axis
    if y_max is None:
        y_max_data = max(
            _compute_freq_distribution(
                df[df["RSA_category"] == cat][value_col], x_min, x_max, x_step
            )[1].max()
            for cat in _DIST_RSA_ORDER
            if not df[df["RSA_category"] == cat][value_col].dropna().empty
        )
        y_max = (int(y_max_data / y_step) + 2) * y_step
    y_ticks = np.arange(y_min, y_max + y_step / 2, y_step)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)

    ax.set_xlabel(x_label,         fontsize=FONT["size_label"])
    ax.set_ylabel("Frequency (%)", fontsize=FONT["size_label"])
    ax.legend(fontsize=FONT["size_legend"], loc=legend_loc)

    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")

# ── plotting: functional-residue categorical scatter ────────────────────────

def plot_test_only_categorical_scatter(test_df: pd.DataFrame,
                                       output_path: Path,
                                       x_col: str,
                                       y_col: str,
                                       x_label: str,
                                       y_label: str,
                                       mode: str = "functional_type",
                                       x_lim: tuple = (0.0, 1.0),
                                       y_lim: tuple = AXIS_LIM,
                                       legend_loc: str = "upper right"):
    """
    Scatter of the test set: x_col (x) vs. y_col (y), colored by functional
    residue status. Two modes are supported:

      mode="binary"          : grey/black scatter using 'is_functional'.
                               (Generic replacement for the old
                               plot_test_only_rsa_vs_closeness() and
                               plot_test_only_functional().)
      mode="functional_type" : 4-category scatter using 'functional_type'
                               (non_functional / binding / catalytic / dual),
                               drawn in FUNCTIONAL_TYPE_DRAW_ORDER so rarer
                               categories stay visible on top.
                               (Generic replacement for
                               plot_test_only_rsa_vs_closeness_2(),
                               plot_test_only_rsa_vs_betweenness(),
                               plot_test_only_rsa_vs_rca_zscore(), and
                               plot_test_only_rsa_vs_pagerank().)

    Parameters
    ----------
    test_df     : pd.DataFrame  Must contain x_col, y_col, and either
                                'is_functional' (mode="binary") or
                                'functional_type' (mode="functional_type").
                                Run add_functional_labels() before calling.
    output_path : Path          Destination file (PNG / TIFF).
    x_col, y_col: str           Column names to plot.
    x_label,
    y_label     : str           Axis labels.
    mode        : str           "binary" or "functional_type".
    x_lim, y_lim: tuple         Axis limits.
    legend_loc  : str           Legend placement.
    """
    if mode not in ("binary", "functional_type"):
        raise ValueError(f"mode must be 'binary' or 'functional_type', got {mode!r}")

    status_col = "is_functional" if mode == "binary" else "functional_type"
    for col in (x_col, y_col, status_col):
        if col not in test_df.columns:
            raise ValueError(
                f"'{col}' column not found. "
                "Ensure the DataFrame has been processed by "
                "add_functional_labels() and the relevant standardize_* function."
            )

    plot_df = test_df[[x_col, y_col, status_col]].dropna(subset=[x_col, y_col])
    n_dropped = len(test_df) - len(plot_df)
    if n_dropped > 0:
        print(f"  plot_test_only_categorical_scatter: dropped {n_dropped} rows "
              f"with NaN in {x_col} or {y_col}.")

    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])

    if mode == "binary":
        non_func = plot_df[~plot_df["is_functional"]]
        func     = plot_df[plot_df["is_functional"]]

        ax.scatter(
            non_func[x_col], non_func[y_col],
            c="#CCCCCC", alpha=0.5, s=12, zorder=2,
            edgecolors="white",
            label=f"Non-functional (n={len(non_func):,})",
        )
        ax.scatter(
            func[x_col], func[y_col],
            c="black", alpha=0.9, s=30, zorder=3,
            edgecolors="white", linewidths=0.5,
            label=f"Functional residues (n={len(func):,})",
        )

    else:  # mode == "functional_type"
        counts = plot_df["functional_type"].value_counts()
        print(f"plot_test_only_categorical_scatter received: {len(plot_df):,} rows")
        for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
            print(f"  {ftype:20s}: {counts.get(ftype, 0):,}")

        scatter_params = {
            "non_functional": dict(alpha=0.45, s=12, zorder=2, edgecolors="none"),
            "binding":        dict(alpha=0.90, s=30, zorder=3, edgecolors="white", linewidths=0.5),
            "catalytic":      dict(alpha=0.90, s=30, zorder=4, edgecolors="white", linewidths=0.5),
            "dual":           dict(alpha=1.00, s=45, zorder=5, edgecolors="white", linewidths=0.6),
        }

        for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
            sub = plot_df[plot_df["functional_type"] == ftype]
            if sub.empty:
                continue
            n = len(sub)
            ax.scatter(
                sub[x_col], sub[y_col],
                c=FUNCTIONAL_TYPE_COLORS[ftype],
                label=f"{FUNCTIONAL_TYPE_LABELS[ftype]} (n={n:,})",
                **scatter_params[ftype],
            )

    ax.legend(fontsize=FONT["size_legend"], loc=legend_loc)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel(x_label, fontsize=FONT["size_label"])
    ax.set_ylabel(y_label, fontsize=FONT["size_label"])

    fig.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")

# ── standalone diagnostics ───────────────────────────────────────────────────

def compute_centrality_correlation(df: pd.DataFrame, label: str = "") -> float:
    """
    Compute and print the Pearson correlation between StdCloseness and StdDegree.
    Drops rows with NaN in either column before computing.
    Returns the correlation coefficient r.
    """
    valid = df[["StdCloseness", "StdDegree"]].dropna()
    n_dropped = len(df) - len(valid)
    if n_dropped > 0:
        print(f"  Note: dropped {n_dropped} rows with NaN values.")

    r, p = stats.pearsonr(valid["StdCloseness"], valid["StdDegree"])
    print(f"\n── Centrality Correlation ({label}) ──────────────────────────")
    print(f"  n residues : {len(valid):,}")
    print(f"  Pearson r  : {r:.4f}")
    print(f"  p-value    : {p:.2e}")
    print("─────────────────────────────────────────────────────────────\n")
    return r


def plot_chain_length_distribution(
    ref_csv,
    output_path,
    test_csv=None,
    max_chain_length_line=None,
    bin_width=10,
):
    """
    Plot the distribution of chain lengths (canonical residues per chain)
    from one or two standardized CSVs.

    Chain length is computed as the number of rows per (pdb_id, Chain) group,
    which reflects canonical-AA residue count after filtering in
    load_centrality_dir().

    Parameters
    ----------
    ref_csv               : str or Path  Path to ref_standardized.csv (or any
                                         standardized CSV with pdb_id + Chain columns).
    output_path           : str or Path  Destination PNG file.
    test_csv               : str or Path, optional  Path to test_standardized.csv.
                                         If provided, overlaid as a second histogram.
    max_chain_length_line : int, optional  If provided, draws a vertical dashed
                                         line at this value for reference (e.g.
                                         a size cutoff applied upstream).
    bin_width              : int          Width of histogram bins in residues. Default 10.
    """
    apply_style()

    def _chain_lengths(csv_path, label):
        df = pd.read_csv(csv_path)
        lengths = (
            df.groupby(["pdb_id", "Chain"])
            .size()
            .reset_index(name="chain_length")
        )
        print(f"\n── Chain length summary: {label} ──────────────────────────")
        print(f"  Chains (unique pdb_id+Chain): {len(lengths)}")
        print(f"  Proteins (unique pdb_id):     {df['pdb_id'].nunique()}")
        print(f"  Min    : {lengths['chain_length'].min()}")
        print(f"  Max    : {lengths['chain_length'].max()}")
        print(f"  Mean   : {lengths['chain_length'].mean():.1f}")
        print(f"  Median : {lengths['chain_length'].median():.1f}")
        print(f"  Q1     : {lengths['chain_length'].quantile(0.25):.1f}")
        print(f"  Q3     : {lengths['chain_length'].quantile(0.75):.1f}")
        print("─────────────────────────────────────────────────────────────")
        return lengths["chain_length"]

    ref_lengths  = _chain_lengths(ref_csv,  "Reference set")
    test_lengths = _chain_lengths(test_csv, "Test set") if test_csv is not None else None

    # shared bin edges across both datasets so bars are comparable
    all_lengths = pd.concat([ref_lengths] + ([test_lengths] if test_lengths is not None else []))
    x_min = 0
    x_max = int(np.ceil(all_lengths.max() / bin_width) * bin_width)
    bins  = np.arange(x_min, x_max + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=FIG_SIZE["half_page"])

    ax.hist(
        ref_lengths,
        bins=bins,
        color=COLORS["blue"],
        alpha=0.6,
        label=f"Reference (n={len(ref_lengths)} chains)",
        edgecolor="white",
        linewidth=0.4,
    )

    if test_lengths is not None:
        ax.hist(
            test_lengths,
            bins=bins,
            color=COLORS["orange"],
            alpha=0.7,
            label=f"Test (n={len(test_lengths)} chains)",
            edgecolor="white",
            linewidth=0.4,
        )

    if max_chain_length_line is not None:
        ax.axvline(
            max_chain_length_line,
            color=COLORS["red"],
            linewidth=1.2,
            linestyle="--",
            label=f"Size reference ({max_chain_length_line} aa)",
        )

    ax.set_xlabel("Chain length (canonical residues)", fontsize=FONT["size_label"])
    ax.set_ylabel("Number of chains",                  fontsize=FONT["size_label"])
    ax.set_xlim(x_min, x_max)
    ax.legend(fontsize=FONT["size_legend"])

    fig.tight_layout(pad=LAYOUT["tight_pad"])
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"\nSaved: {output_path}")

# ── figure config tables ─────────────────────────────────────────────────────
#
# Each entry drives one call into plot_test_only_scatter() / plot_overlay_scatter().
# "name" is used to build the output filename; everything else maps directly to
# the function's keyword arguments.

SCATTER_SPECS = [
    dict(name="closeness_vs_degree",
         x_col="StdCloseness", y_col="StdDegree",
         x_label="Standardized Closeness Centrality",
         y_label="Standardized Degree Centrality",
         x_lim=AXIS_LIM, y_lim=AXIS_LIM, legend_loc="upper left"),
    dict(name="betweenness_vs_closeness",
         x_col="StdCloseness", y_col="StdBetweenness",
         x_label="Standardized Closeness Centrality",
         y_label="Standardized Betweenness Centrality",
         x_lim=AXIS_LIM, y_lim=AXIS_LIM_2, legend_loc="upper left"),
    dict(name="betweenness_vs_degree",
         x_col="StdDegree", y_col="StdBetweenness",
         x_label="Standardized Degree Centrality",
         y_label="Standardized Betweenness Centrality",
         x_lim=AXIS_LIM, y_lim=AXIS_LIM_2, legend_loc="upper left"),
    dict(name="degree_vs_betweenness",
         x_col="StdBetweenness", y_col="StdDegree",
         x_label="Standardized Betweenness Centrality",
         y_label="Standardized Degree Centrality",
         x_lim=AXIS_LIM_2, y_lim=AXIS_LIM, legend_loc="upper right"),
    dict(name="degree_vs_rca",
         x_col="RCA_zscore", y_col="StdDegree",
         x_label="RCA Z-score",
         y_label="Standardized Degree Centrality",
         x_lim=AXIS_LIM_2, y_lim=AXIS_LIM, legend_loc="upper right"),
    dict(name="pagerank_vs_closeness",
         x_col="StdCloseness", y_col="StdPageRank",
         x_label="Standardized Closeness Centrality",
         y_label="Standardized PageRank",
         x_lim=AXIS_LIM, y_lim=AXIS_LIM_2, legend_loc="lower right"),
    dict(name="rca_zscore_vs_closeness",
         x_col="StdCloseness", y_col="RCA_zscore",
         x_label="Standardized Closeness Centrality",
         y_label="RCA Z-score",
         x_lim=AXIS_LIM, y_lim=AXIS_LIM_2, legend_loc="upper left",
         threshold_line=2),
]

# Each entry drives one call into plot_std_frequency().
FREQUENCY_SPECS = [
    dict(name="stdcloseness_frequency",   value_col="StdCloseness",
         x_label="Standardized Closeness Centrality",   legend_loc="upper left"),
    dict(name="stddegree_frequency",      value_col="StdDegree",
         x_label="Standardized Degree Centrality",      legend_loc="upper left"),
    dict(name="stdbetweenness_frequency", value_col="StdBetweenness",
         x_label="Standardized Betweenness Centrality", legend_loc="upper right"),
    dict(name="stdpagerank_frequency",    value_col="StdPageRank",
         x_label="Standardized PageRank",                legend_loc="upper right"),
]

# Each entry drives one call into plot_test_only_categorical_scatter() with
# mode="functional_type". The binary-mode plots (RSA-vs-closeness, and
# closeness-vs-degree highlighting functional residues) are called directly
# in run_pipeline() since there are only two of them.
CATEGORICAL_SPECS = [
    dict(name="rsa_vs_closeness",
         x_col="RSA-empirical", y_col="StdCloseness",
         x_label="RSA (empirical)", y_label="Standardized Closeness Centrality",
         y_lim=AXIS_LIM, legend_loc="upper right"),
    dict(name="rsa_vs_betweenness",
         x_col="RSA-empirical", y_col="StdBetweenness",
         x_label="RSA (empirical)", y_label="Standardized Betweenness Centrality",
         y_lim=AXIS_LIM_2, legend_loc="upper right"),
    dict(name="rsa_vs_rca_zscore",
         x_col="RSA-empirical", y_col="RCA_zscore",
         x_label="RSA (empirical)", y_label="RCA Z-score",
         y_lim=AXIS_LIM_2, legend_loc="upper right"),
    dict(name="rsa_vs_pagerank",
         x_col="RSA-empirical", y_col="StdPageRank",
         x_label="RSA (empirical)", y_label="Standardized PageRank",
         y_lim=AXIS_LIM_2, legend_loc="lower right"),
]

# ── pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    ref_centrality_dir,
    test_centrality_dir,
    output_dir,
    biolip_labels_csv=None,
    exclude_test_pdb_ids=None,
    fasta_path=None,
):
    """
    Full pipeline: load reference + test centrality data (already filtered and
    annotated with RSA-empirical, RCA, and PageRank upstream), standardize,
    join functional-residue labels, save the standardized CSVs, and generate
    the full set of figures.

    Parameters
    ----------
    ref_centrality_dir   : str or Path  Directory of reference (PISCES)
                                        *_centrality.csv files. Assumed already
                                        filtered to the desired population.
    test_centrality_dir  : str or Path  Directory of test-set *_centrality.csv files.
    output_dir            : str or Path  Destination directory for CSVs and figures.
    biolip_labels_csv     : str or Path, optional
        Path to biolip_residue_labels.csv. If provided, functional-residue
        figures are generated; if None, they are skipped.
    exclude_test_pdb_ids  : set of str, optional  PDB IDs to exclude from the test set.
    fasta_path             : str or Path, optional
        If provided, restricts the test set to (pdb_id, chain) pairs present
        in this FASTA file.
    """
    ref_centrality_dir  = Path(ref_centrality_dir)
    test_centrality_dir = Path(test_centrality_dir)
    output_dir           = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. load reference set ─────────────────────────────────────────────
    print("Loading reference (old PISCES) centrality CSVs...")
    ref_df = load_centrality_dir(ref_centrality_dir)
    print(f"  Loaded {len(ref_df):,} residues from {ref_df['pdb_id'].nunique()} proteins.")

    # ── 2. compute per-AA-type degree stats from reference set ────────────
    aa_stats = compute_degree_stats_by_aa(ref_df)
    aa_stats.to_csv(output_dir / "degree_stats_by_aa_type.csv", index=False)

    # ── 3. standardize reference set ───────────────────────────────────────
    ref_df = standardize_closeness_per_protein(ref_df)
    ref_df = standardize_degree_by_aa(ref_df, aa_stats)
    ref_df = standardize_betweenness_per_protein(ref_df)
    if "PageRank_zscore" in ref_df.columns:
        ref_df = rename_pagerank_zscore(ref_df)
    ref_df = assign_rsa_category(ref_df)

    # ── 4. load test set ────────────────────────────────────────────────────
    print("Loading small enzyme test set centrality CSVs...")
    test_df = load_centrality_dir(test_centrality_dir, exclude_pdb_ids=exclude_test_pdb_ids)
    print(f"  Loaded {len(test_df):,} residues from {test_df['pdb_id'].nunique()} proteins.")

    # ── 4b. filter test_df to FASTA chains only ────────────────────────────
    if fasta_path is not None:
        fasta_chains = parse_fasta_chains(fasta_path)
        before = len(test_df)
        test_df["pdb_id_upper"] = test_df["pdb_id"].str.upper()
        test_df = test_df[
            test_df.apply(
                lambda r: (r["pdb_id_upper"], r["Chain"]) in fasta_chains,
                axis=1
            )
        ].copy()
        test_df.drop(columns=["pdb_id_upper"], inplace=True)
        print(f"  FASTA chain filter: {before:,} -> {len(test_df):,} residues "
              f"({test_df['pdb_id'].nunique()} proteins, "
              f"{test_df.groupby(['pdb_id','Chain']).ngroups} chains)")
    else:
        print("  NOTE: fasta_path is None — FASTA chain filter was not applied.")

    # ── 5. standardize test set (skip steps already present in the CSVs) ───
    # closeness: per-protein z-score (each test protein uses its own mean/std)
    # degree: per-AA-type z-score using reference population stats
    test_df = standardize_if_missing(test_df, aa_stats)

    # ── 5b. functional-residue labels (test set only) ──────────────────────
    if biolip_labels_csv is not None:
        label_df = load_functional_labels(biolip_labels_csv)
        test_df  = add_functional_labels(test_df, label_df)

    # diagnostic prints
    print(test_df.columns.tolist())
    print(test_df.shape)
    print(test_df["pdb_id"].nunique())
    print(test_df["ClosenessCentrality"].isna().sum())

    # ── 6. save standardized CSVs ───────────────────────────────────────────
    ref_df.to_csv(output_dir  / "ref_standardized.csv",  index=False)
    test_df.to_csv(output_dir / "test_standardized.csv", index=False)
    print("Saved standardized CSVs.")

    # ── 7. core closeness-vs-degree figures ─────────────────────────────────
    closeness_vs_degree = SCATTER_SPECS[0]
    plot_test_only_scatter(
        test_df=ref_df, output_path=output_dir / "fig1_reference_only.png",
        **{k: v for k, v in closeness_vs_degree.items() if k != "name"},
    )
    plot_test_only_scatter(
        test_df=test_df, output_path=output_dir / "fig1_test_only.png",
        **{k: v for k, v in closeness_vs_degree.items() if k != "name"},
    )
    plot_overlay_scatter(
        ref_df=ref_df, test_df=test_df, output_path=output_dir / "fig1_overlay.png",
        **{k: v for k, v in closeness_vs_degree.items() if k != "name"},
    )

    # ── 8. remaining scatter figures (test-only + overlay where columns exist) ─
    for spec in SCATTER_SPECS[1:]:
        kwargs = {k: v for k, v in spec.items() if k != "name"}
        required = {spec["x_col"], spec["y_col"], "RSA_category"}

        if required.issubset(test_df.columns):
            plot_test_only_scatter(
                test_df=test_df,
                output_path=output_dir / f"fig_{spec['name']}_test.png",
                **kwargs,
            )
        else:
            print(f"  Skipping test-only '{spec['name']}': missing columns.")

        if required.issubset(ref_df.columns) and required.issubset(test_df.columns):
            plot_overlay_scatter(
                ref_df=ref_df, test_df=test_df,
                output_path=output_dir / f"fig_{spec['name']}_overlay.png",
                **kwargs,
            )
        else:
            print(f"  Skipping overlay '{spec['name']}': missing columns.")

    # ── 9. frequency distributions (reference set) ──────────────────────────
    for spec in FREQUENCY_SPECS:
        kwargs = {k: v for k, v in spec.items() if k != "name"}
        if spec["value_col"] in ref_df.columns:
            plot_std_frequency(
                ref_df,
                output_dir / f"fig_{spec['name']}.png",
                y_max=1.0,
                **kwargs,
            )
        else:
            print(f"  Skipping frequency plot '{spec['name']}': missing column.")

    # ── 10. functional-residue figures (requires biolip_labels_csv) ─────────
    if biolip_labels_csv is not None:
        # binary mode: closeness vs degree, functional residues highlighted
        plot_test_only_categorical_scatter(
            test_df, output_dir / "fig1_test_functional.png",
            x_col="StdCloseness", y_col="StdDegree",
            x_label="Standardized Closeness Centrality",
            y_label="Standardized Degree Centrality",
            mode="binary", x_lim=AXIS_LIM, y_lim=AXIS_LIM, legend_loc="upper left",
        )
        # binary mode: RSA vs closeness, functional residues highlighted
        plot_test_only_categorical_scatter(
            test_df, output_dir / "fig_rsa_vs_closeness_binary.png",
            x_col="RSA-empirical", y_col="StdCloseness",
            x_label="RSA (empirical)", y_label="Standardized Closeness Centrality",
            mode="binary", x_lim=(0.0, 1.0), y_lim=AXIS_LIM, legend_loc="upper right",
        )
        # 4-category functional_type mode for the RSA-vs-X family
        for spec in CATEGORICAL_SPECS:
            kwargs = {k: v for k, v in spec.items() if k != "name"}
            required = {spec["x_col"], spec["y_col"], "functional_type"}
            if required.issubset(test_df.columns):
                plot_test_only_categorical_scatter(
                    test_df, output_dir / f"fig_{spec['name']}.png",
                    mode="functional_type", x_lim=(0.0, 1.0),
                    **kwargs,
                )
            else:
                print(f"  Skipping categorical '{spec['name']}': missing columns.")

    print("\nDone.")


if __name__ == "__main__":
    # Edit these paths/parameters as needed, then run:  python figures.py

    # ref_centrality_dir   = DATA_DIR_3 / "ref" / "Centralities"
    # test_centrality_dir  = DATA_DIR_3 / "test" / "Centralities"
    # output_dir           = FINAL_DIR / "figures"
    # biolip_labels_csv    = DATA_DIR_3 / "test" / "biolip_lookup" / "biolip_residue_labels.csv"

    # # # 6/24/26 - excludes disconnected graphs / pdb-parsing issues
    # # exclude_test_pdb_ids = {
    # #     '7ara', '4q95', '1fa5', '1g5q', '1qzu', '2xki', '3d6e', '3f21',
    # #     '4bbq', '4fzy', '4rkg', '5ztm', '7ydw', '8uc6',
    # # }

    # run_pipeline(
    #     ref_centrality_dir    = ref_centrality_dir,
    #     test_centrality_dir   = test_centrality_dir,
    #     output_dir             = output_dir,
    #     biolip_labels_csv      = biolip_labels_csv,
    #     exclude_test_pdb_ids   = None,
    #     fasta_path              = None,
    # )

    # 7/3/26 need pagerank graphs - fix legend locations

    ref_standardized_csv  = FIGS_DIR / "ref_standardized.csv"
    test_standardized_csv = FIGS_DIR / "test_standardized.csv"

    pagerank_vs_degree_x_col      = "StdPageRank"
    pagerank_vs_degree_y_col      = "StdDegree"
    pagerank_vs_degree_x_label    = "Standardized PageRank"
    pagerank_vs_degree_y_label    = "Standardized Degree Centrality"
    pagerank_vs_degree_x_lim      = AXIS_LIM
    pagerank_vs_degree_y_lim      = AXIS_LIM
    pagerank_vs_degree_legend_loc = "lower right"

    pagerank_degree_test_output_path    = FIGS_DIR / "pagerank" / "fig_pagerank_vs_degree_test.png"


    ref_df_reload  = pd.read_csv(ref_standardized_csv)
    test_df_reload = pd.read_csv(test_standardized_csv)

    # plot_test_only_scatter(
    #     test_df=test_df_reload,
    #     output_path=pagerank_degree_test_output_path,
    #     x_col=pagerank_vs_degree_x_col, y_col=pagerank_vs_degree_y_col,
    #     x_label=pagerank_vs_degree_x_label, y_label=pagerank_vs_degree_y_label,
    #     x_lim=pagerank_vs_degree_x_lim, y_lim=pagerank_vs_degree_y_lim,
    #     legend_loc=pagerank_vs_degree_legend_loc,
    # )

    pagerank_rsa_test_output_path    = FIGS_DIR / "pagerank" / "fig_pagerank_vs_rsa_test.png"

    plot_test_only_categorical_scatter(
        test_df=test_df_reload,
        output_path=pagerank_rsa_test_output_path,
        x_col="RSA-empirical", y_col="StdPageRank",
        x_label="RSA (empirical)", y_label="Standardized PageRank Centrality",
        mode="functional_type", x_lim=(0.0, 1.0), y_lim=AXIS_LIM, legend_loc="upper right",
    )
