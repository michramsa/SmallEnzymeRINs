# Purpose: Generate functional breakdown figures (closeness vs degree,
#          betweenness vs closeness, betweenness vs degree) separately for
#          each EC class, using only enzyme chains that share their EC class.
#
# Extends 0_4_fig1_recreation_functional_breakdown_ver3.py with:
#   - load_ec_classifications()  — parses test_set_ec_classifications.csv
#   - filter_to_ec_class()       — filters test_df to a single EC class
#   - Three plot wrappers that accept an ec_class_name label for titling /
#     output naming.
#
# Coloring scheme is identical to 0_4_fig1_recreation_functional_breakdown_ver3.py:
#   binding        → #8E8278
#   catalytic      → #35302B
#   dual           → #E8E4DF
#   non_functional → #CCCCCC
#
# Output files (one per EC class per plot type), e.g.:
#   fig1_enzyme_breakdown_Hydrolases_closeness_vs_degree.png
#   fig1_enzyme_breakdown_Hydrolases_betweenness_vs_closeness.png
#   fig1_enzyme_breakdown_Hydrolases_betweenness_vs_degree.png

import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from figure_style_ver2 import apply_style, save_fig, validate_figure, FIG_SIZE, FONT, LAYOUT, DPI, COLORS
from scipy import stats

# ── constants ─────────────────────────────────────────────────────────────────

CANONICAL_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

EXPECTED_TOKEN  = re.compile(r'^[A-Z]\d+$')
INSERTION_TOKEN = re.compile(r'^[A-Z]\d+[A-Za-z]+$')

FUNCTIONAL_TYPES = ["catalytic", "binding", "dual"]

# Colours carried over from 0_4_fig1_recreation_functional_breakdown_ver3.py
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


# ── data loading ──────────────────────────────────────────────────────────────

def load_functional_type_labels(detailed_csv: Path) -> pd.DataFrame:
    """
    Parse biolip_functional_residues_detailed.csv and return a long-form
    DataFrame with one row per (pdb_id, chain, residue_num, functional_type).

    functional_type is one of: 'catalytic', 'binding', 'dual'.

    Priority when a residue appears under multiple types:
        dual > catalytic > binding
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
        print(f"Warning: insertion code tokens skipped: {n_insertion}. "
              f"Examples: {insertion_examples}")
    if n_unexpected > 0:
        print(f"Warning: unexpected token formats skipped: {n_unexpected}. "
              f"Examples: {unexpected_examples}")

    label_df = pd.DataFrame(records)
    label_df = label_df.drop_duplicates(
        subset=["pdb_id", "chain", "residue_num", "functional_type"]
    )

    print(f"Functional type labels loaded: {len(label_df):,} tokens across "
          f"{label_df['pdb_id'].nunique()} proteins.")
    return label_df


def add_functional_type(df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join functional_type labels onto a residue-level DataFrame.

    Join key: pdb_id (lowercase) + Chain + Residue number.
    Residues not found in label_df receive functional_type = 'non_functional'.

    When a residue appears under multiple functional types in label_df,
    the highest-priority type is kept:
        dual > catalytic > binding
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

    if "functional_type_y" in merged.columns:
        ft_col = "functional_type_y"
    else:
        ft_col = "functional_type"
    merged[ft_col] = merged[ft_col].fillna("non_functional")
    df["functional_type"] = merged[ft_col].values
    df.drop(columns=["pdb_id_lower"], inplace=True)

    test_ids  = set(df["pdb_id"].str.lower().unique())
    label_ids = set(label_dedup["pdb_id"].str.lower().unique())
    missing   = test_ids - label_ids
    if missing:
        import warnings
        warnings.warn(
            f"{len(missing)} test-set protein(s) have no entry in the detailed CSV — "
            f"all their residues will be labeled non_functional. "
            f"Missing: {sorted(missing)}"
        )

    counts = df["functional_type"].value_counts()
    print("\n── Functional type breakdown ─────────────────────────────────")
    for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
        print(f"  {ftype:20s}: {counts.get(ftype, 0):,}")
    print("─────────────────────────────────────────────────────────────\n")

    return df


def standardize_betweenness_per_protein(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score BetweennessCentrality within each protein individually.
    Adds StdBetweenness column.
    Proteins with std == 0 (degenerate case) receive StdBetweenness = 0.
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


# ── EC classification loading and filtering ───────────────────────────────────

def load_ec_classifications(ec_csv: Path) -> pd.DataFrame:
    """
    Parse test_set_ec_classifications.csv and return a DataFrame with columns:
        pdb_id (lowercase), prio_ec_number, ec_class_name.

    Expected CSV format:
        pdb_id,prio_ec_number,ec_class_name
        2a11,3.1.26.3,Hydrolases
    """
    ec_df = pd.read_csv(ec_csv)
    ec_df["pdb_id"] = ec_df["pdb_id"].str.lower().str.strip()
    ec_df["ec_class_name"] = ec_df["ec_class_name"].str.strip()

    print(f"EC classifications loaded: {len(ec_df):,} entries, "
          f"{ec_df['ec_class_name'].nunique()} unique EC class(es).")
    print(f"  Classes found: {sorted(ec_df['ec_class_name'].unique())}")
    return ec_df


def filter_to_ec_class(test_df: pd.DataFrame,
                       ec_df: pd.DataFrame,
                       ec_class_name: str) -> pd.DataFrame:
    """
    Filter test_df to only residues belonging to enzyme chains whose EC class
    matches ec_class_name.

    Only pdb_ids that appear in ec_df with the given ec_class_name are kept.
    The match is case-insensitive on ec_class_name.

    Parameters
    ----------
    test_df       : Full residue-level DataFrame (already has functional_type).
    ec_df         : Output of load_ec_classifications().
    ec_class_name : EC class to filter to (e.g. 'Hydrolases').

    Returns
    -------
    Filtered DataFrame containing only residues from matching enzyme chains.
    """
    mask = ec_df["ec_class_name"].str.lower() == ec_class_name.lower()
    matching_ids = set(ec_df.loc[mask, "pdb_id"].str.lower())

    if not matching_ids:
        raise ValueError(
            f"No pdb_ids found for EC class '{ec_class_name}'. "
            f"Available classes: {sorted(ec_df['ec_class_name'].unique())}"
        )

    filtered = test_df[test_df["pdb_id"].str.lower().isin(matching_ids)].copy()

    print(f"\n── EC class filter: '{ec_class_name}' ───────────────────────")
    print(f"  Matching pdb_ids : {len(matching_ids)}")
    print(f"  Residues retained: {len(filtered):,} "
          f"(from {filtered['pdb_id'].nunique()} proteins)")
    if filtered.empty:
        import warnings
        warnings.warn(
            f"No residues remain after filtering to EC class '{ec_class_name}'. "
            "Check that pdb_id casing matches between test_df and ec_df."
        )
    return filtered


# ── axis helpers ──────────────────────────────────────────────────────────────

def _format_axes_no_title(ax,
                          xlabel="Standardized Closeness Centrality",
                          ylabel="Standardized Degree Centrality"):
    ax.set_xlim(AXIS_LIM)
    ax.set_ylim(AXIS_LIM)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel(xlabel, fontsize=FONT["size_label"])
    ax.set_ylabel(ylabel, fontsize=FONT["size_label"])


# ── per-EC-class plot functions ───────────────────────────────────────────────

def plot_enzyme_breakdown_closeness_vs_degree(ec_df_subset: pd.DataFrame,
                                              output_path: Path,
                                              x_lim: tuple = AXIS_LIM,
                                              y_lim: tuple = AXIS_LIM) -> None:
    """
    Scatter of StdCloseness (x) vs StdDegree (y) for one EC class,
    coloured by functional residue type.

    Parameters
    ----------
    ec_df_subset : Residue-level DataFrame already filtered to one EC class,
                   with columns StdCloseness, StdDegree, functional_type.
    output_path  : Destination PNG file.
    x_lim        : X-axis limits. Default (-4, 4).
    y_lim        : Y-axis limits. Default (-4, 4).
    """
    if "functional_type" not in ec_df_subset.columns:
        raise ValueError(
            "'functional_type' column not found. "
            "Run add_functional_type() before calling this function."
        )

    counts = ec_df_subset["functional_type"].value_counts()
    print(f"plot_enzyme_breakdown_closeness_vs_degree received: {len(ec_df_subset):,} rows")
    for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
        print(f"  {ftype:20s}: {counts.get(ftype, 0):,}")

    scatter_params = {
        "non_functional": dict(alpha=0.45, s=12, zorder=2, edgecolors="none"),
        "binding":        dict(alpha=0.90, s=30, zorder=3, edgecolors="white", linewidths=0.5),
        "catalytic":      dict(alpha=0.90, s=30, zorder=4, edgecolors="white", linewidths=0.5),
        "dual":           dict(alpha=1.00, s=45, zorder=5, edgecolors="grey", linewidths=0.5),
    }

    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])

    for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
        sub = ec_df_subset[ec_df_subset["functional_type"] == ftype]
        if sub.empty:
            continue
        n = len(sub)
        ax.scatter(
            sub["StdCloseness"],
            sub["StdDegree"],
            c=FUNCTIONAL_TYPE_COLORS[ftype],
            label=f"{FUNCTIONAL_TYPE_LABELS[ftype]} (n={n:,})",
            **scatter_params[ftype],
        )

    ax.legend(fontsize=FONT["size_legend"], loc="upper left")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    _format_axes_no_title(ax)
    fig.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_enzyme_breakdown_betweenness_vs_closeness(ec_df_subset: pd.DataFrame,
                                                   output_path: Path,
                                                   x_lim: tuple = AXIS_LIM,
                                                   y_lim: tuple = (-4, 10)) -> None:
    """
    Scatter of StdCloseness (x) vs StdBetweenness (y) for one EC class,
    coloured by functional residue type.

    Parameters
    ----------
    ec_df_subset : Residue-level DataFrame already filtered to one EC class,
                   with columns StdCloseness, StdBetweenness, functional_type.
    output_path  : Destination PNG file.
    x_lim        : X-axis limits. Default (-4, 4).
    y_lim        : Y-axis limits. Default (-4, 10).
    """
    if "functional_type" not in ec_df_subset.columns:
        raise ValueError(
            "'functional_type' column not found. "
            "Run add_functional_type() before calling this function."
        )
    if "StdBetweenness" not in ec_df_subset.columns:
        raise ValueError(
            "'StdBetweenness' column not found. "
            "Run standardize_betweenness_per_protein() before calling this function."
        )

    counts = ec_df_subset["functional_type"].value_counts()
    print(f"plot_enzyme_breakdown_betweenness_vs_closeness received: {len(ec_df_subset):,} rows")
    for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
        print(f"  {ftype:20s}: {counts.get(ftype, 0):,}")

    scatter_params = {
        "non_functional": dict(alpha=0.45, s=12, zorder=2, edgecolors="none"),
        "binding":        dict(alpha=0.90, s=30, zorder=3, edgecolors="white", linewidths=0.5),
        "catalytic":      dict(alpha=0.90, s=30, zorder=4, edgecolors="white", linewidths=0.5),
        "dual":           dict(alpha=1.00, s=45, zorder=5, edgecolors="grey", linewidths=0.6),
    }

    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])

    for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
        sub = ec_df_subset[ec_df_subset["functional_type"] == ftype]
        if sub.empty:
            continue
        n = len(sub)
        ax.scatter(
            sub["StdCloseness"],
            sub["StdBetweenness"],
            c=FUNCTIONAL_TYPE_COLORS[ftype],
            label=f"{FUNCTIONAL_TYPE_LABELS[ftype]} (n={n:,})",
            **scatter_params[ftype],
        )

    ax.legend(fontsize=FONT["size_legend"], loc="upper left")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Standardized Closeness Centrality",   fontsize=FONT["size_label"])
    ax.set_ylabel("Standardized Betweenness Centrality", fontsize=FONT["size_label"])
    fig.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_enzyme_breakdown_betweenness_vs_degree(ec_df_subset: pd.DataFrame,
                                                output_path: Path,
                                                x_lim: tuple = AXIS_LIM,
                                                y_lim: tuple = (-4, 10)) -> None:
    """
    Scatter of StdDegree (x) vs StdBetweenness (y) for one EC class,
    coloured by functional residue type.

    Parameters
    ----------
    ec_df_subset : Residue-level DataFrame already filtered to one EC class,
                   with columns StdDegree, StdBetweenness, functional_type.
    output_path  : Destination PNG file.
    x_lim        : X-axis limits. Default (-4, 4).
    y_lim        : Y-axis limits. Default (-4, 10).
    """
    if "functional_type" not in ec_df_subset.columns:
        raise ValueError(
            "'functional_type' column not found. "
            "Run add_functional_type() before calling this function."
        )
    if "StdBetweenness" not in ec_df_subset.columns:
        raise ValueError(
            "'StdBetweenness' column not found. "
            "Run standardize_betweenness_per_protein() before calling this function."
        )

    counts = ec_df_subset["functional_type"].value_counts()
    print(f"plot_enzyme_breakdown_betweenness_vs_degree received: {len(ec_df_subset):,} rows")
    for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
        print(f"  {ftype:20s}: {counts.get(ftype, 0):,}")

    scatter_params = {
        "non_functional": dict(alpha=0.45, s=12, zorder=2, edgecolors="none"),
        "binding":        dict(alpha=0.90, s=30, zorder=3, edgecolors="white", linewidths=0.5),
        "catalytic":      dict(alpha=0.90, s=30, zorder=4, edgecolors="white", linewidths=0.5),
        "dual":           dict(alpha=1.00, s=45, zorder=5, edgecolors="grey", linewidths=0.6),
    }

    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])

    for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
        sub = ec_df_subset[ec_df_subset["functional_type"] == ftype]
        if sub.empty:
            continue
        n = len(sub)
        ax.scatter(
            sub["StdDegree"],
            sub["StdBetweenness"],
            c=FUNCTIONAL_TYPE_COLORS[ftype],
            label=f"{FUNCTIONAL_TYPE_LABELS[ftype]} (n={n:,})",
            **scatter_params[ftype],
        )

    ax.legend(fontsize=FONT["size_legend"], loc="upper left")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Standardized Degree Centrality",      fontsize=FONT["size_label"])
    ax.set_ylabel("Standardized Betweenness Centrality", fontsize=FONT["size_label"])
    fig.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── paths — edit these as needed ──────────────────────────────────────
    ver_dir      = DATA_DIR_2 / "figures" / "controlled_fig1_recreation_ver0.7"
    test_csv     = ver_dir / "test_standardized.csv"
    detailed_csv = DATA_DIR_2 / "biolip_functional_residues_detailed.csv"
    ec_csv       = DATA_DIR_2 / "figures" / "controlled_fig1_recreation_ver0.7" / "test_set_ec_classifications.csv"
    out_dir      = ver_dir / "enzyme_breakdown_by_ec_class"
    # ─────────────────────────────────────────────────────────────────────

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load standardized test data
    test_df = pd.read_csv(test_csv)

    # 2. Standardize betweenness per protein
    test_df = standardize_betweenness_per_protein(test_df)

    # 3. Load functional type labels from detailed BioLiP CSV
    label_df = load_functional_type_labels(detailed_csv)

    # 4. Join functional_type onto test_df
    test_df = add_functional_type(test_df, label_df)

    # 5. Load EC classifications
    ec_df = load_ec_classifications(ec_csv)

    # 6. Iterate over every unique EC class and produce three figures each
    ec_classes = sorted(ec_df["ec_class_name"].unique())
    print(f"\nGenerating figures for {len(ec_classes)} EC class(es): {ec_classes}\n")

    for ec_class in ec_classes:
        # Sanitize class name for use in filenames (replace spaces with underscores)
        safe_name = ec_class.replace(" ", "_")

        print(f"\n{'='*60}")
        print(f"EC class: {ec_class}")
        print(f"{'='*60}")

        ec_subset = filter_to_ec_class(test_df, ec_df, ec_class)

        if ec_subset.empty:
            print(f"  Skipping '{ec_class}' — no residues after filtering.")
            continue

        # Figure A: Closeness vs Degree
        plot_enzyme_breakdown_closeness_vs_degree(
            ec_subset,
            out_dir / f"fig1_enzyme_breakdown_{safe_name}_closeness_vs_degree.png",
        )

        # Figure B: Betweenness vs Closeness
        plot_enzyme_breakdown_betweenness_vs_closeness(
            ec_subset,
            out_dir / f"fig1_enzyme_breakdown_{safe_name}_betweenness_vs_closeness.png",
        )

        # Figure C: Betweenness vs Degree
        plot_enzyme_breakdown_betweenness_vs_degree(
            ec_subset,
            out_dir / f"fig1_enzyme_breakdown_{safe_name}_betweenness_vs_degree.png",
        )

    print("\nDone.")
