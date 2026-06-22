# Purpose: Generate fig1_test_functional_breakdown figures.
#
# Extends 0_4_fig1_recreation_functional_breakdown_ver3.py with:
#   - merge_rca_into_standardized()          — patches RCA columns from raw centrality CSVs
#   - plot_functional_breakdown_rca_vs_closeness() — RCA z-score (y) vs StdCloseness (x)
#         coloured by functional residue type, with dotted threshold line at y=2 (del Sol 2006)
#
# Functional type colour scheme:
#       binding        → #8E8278
#       catalytic      → #35302B
#       dual           → #E8E4DF
#       non-functional → #CCCCCC
#
# All other functions are carried over unchanged from ver3.

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

# Residue token validation patterns (from 0_5_loo_experiment_ver6.py)
EXPECTED_TOKEN  = re.compile(r'^[A-Z]\d+$')
INSERTION_TOKEN = re.compile(r'^[A-Z]\d+[A-Za-z]+$')

FUNCTIONAL_TYPES = ["catalytic", "binding", "dual"]

# Colours for the breakdown plot
FUNCTIONAL_TYPE_COLORS = {
    "binding":        "#8E8278",   # blue
    "catalytic":      "#35302B",   # orange
    "dual":           "#E8E4DF",   # red
    "non_functional": "#CCCCCC",   # light grey
}

FUNCTIONAL_TYPE_LABELS = {
    "binding":        "Binding",
    "catalytic":      "Catalytic",
    "dual":           "Dual (binding + catalytic)",
    "non_functional": "Non-functional",
}

# Draw order: non-functional first (background), then each functional type on top
FUNCTIONAL_TYPE_DRAW_ORDER = ["non_functional", "binding", "catalytic", "dual"]

RSA_COLORS = {
    "exposed": "#35302B",
    "core":    "#E8E4DF",
    "other":   "#8E8278",
    "unknown": "#8D8D8D",
}

RSA_ORDER  = ["other", "core", "exposed", "unknown"]
RSA_LABELS = {
    "exposed": "Highly exposed (RSA > 0.5)",
    "core":    "Core (RSA = 0)",
    "other":   "Other (0 < RSA ≤ 0.5)",
    "unknown": "RSA not available",
}

OVERLAY_COLORS = {
    "exposed": "#35302B",   
    "core":    "#E8E4DF",   
    "other":   "#8E8278",   
    "unknown": "#8D8D8D",  
}

AXIS_LIM = (-4, 4)


# ── data loading ──────────────────────────────────────────────────────────────

def load_biolip_functional_residues(biolip_csv: Path) -> pd.DataFrame:
    """
    Parse the BioLiP CSV and return a DataFrame of functional residue join keys.
    (Carried over from 0_4_fig1_recreation.py — used by mark_functional_residues.)
    """
    biolip_df = pd.read_csv(biolip_csv)
    records = []

    for _, row in biolip_df.iterrows():
        pdb_id = str(row["pdb_id"]).upper()
        chain  = str(row["chain"])
        tokens = str(row["functional_residues"]).split()

        for token in tokens:
            if not token:
                continue
            aa_one = token[0]
            try:
                res_num = int(token[1:])
            except ValueError:
                print(f"Warning: could not parse residue number from token '{token}' "
                      f"in {pdb_id} — skipping.")
                continue
            aa_three = ONE_TO_THREE.get(aa_one, "UNK")
            records.append({
                "pdb_id":      pdb_id,
                "chain":       chain,
                "residue_num": res_num,
                "aa_one":      aa_one,
                "aa_three":    aa_three,
            })

    return pd.DataFrame(records)


def load_functional_type_labels(detailed_csv: Path) -> pd.DataFrame:
    """
    Parse biolip_functional_residues_detailed.csv and return a long-form
    DataFrame with one row per (pdb_id, chain, residue_num, functional_type).

    functional_type is one of: 'catalytic', 'binding', 'dual'.

    Residue tokens are validated at load time:
      - Expected format : [single uppercase letter][integer]  e.g. C95
      - Insertion codes : [letter][digits][trailing letter]   e.g. C95A  → skipped
      - Other unexpected formats are flagged and skipped.

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

    # pandas suffixes the column 'functional_type_y' only when test_df already
    # contains a 'functional_type' column; otherwise it stays 'functional_type'.
    if "functional_type_y" in merged.columns:
        ft_col = "functional_type_y"
    else:
        ft_col = "functional_type"
    merged[ft_col] = merged[ft_col].fillna("non_functional")
    df["functional_type"] = merged[ft_col].values
    df.drop(columns=["pdb_id_lower"], inplace=True)

    # warn about test proteins absent from label_df
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

    # count breakdown
    counts = df["functional_type"].value_counts()
    print("\n── Functional type breakdown ─────────────────────────────────")
    for ftype in FUNCTIONAL_TYPE_DRAW_ORDER:
        print(f"  {ftype:20s}: {counts.get(ftype, 0):,}")
    print("─────────────────────────────────────────────────────────────\n")

    return df


# ── standardization (carried over) ────────────────────────────────────────────

def assign_rsa_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "RSA-empirical" not in df.columns:
        df["RSA_category"] = "unknown"
        return df
    conditions = [df["RSA-empirical"] > 0.5, df["RSA-empirical"] == 0.0]
    choices    = ["exposed", "core"]
    df["RSA_category"] = np.select(conditions, choices, default="other")
    return df


# ── axis helpers (carried over) ───────────────────────────────────────────────

def _format_axes(ax, title,
                 xlabel="Standardized Closeness Centrality",
                 ylabel="Standardized Degree Centrality"):
    ax.set_xlim(AXIS_LIM)
    ax.set_ylim(AXIS_LIM)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel(xlabel, fontsize=FONT["size_label"])
    ax.set_ylabel(ylabel, fontsize=FONT["size_label"])
    ax.set_title(title,   fontsize=FONT["size_title"])

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

    Requires 'functional_type' column (added by add_functional_type).

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
            "Run add_functional_type() before calling this function."
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
    # _format_axes(
    #     ax,
    #     f"Small Enzyme Test Set: Functional Residue Types\n"
    #     f"(n = {len(test_df):,} residues, {test_df['pdb_id'].nunique()} proteins)\n"
    #     "Closeness: per-protein z-score  |  Degree: per-AA-type z-score (ref population)",
    #     xlabel="Standardized Closeness",
    #     ylabel="Standardized Degree",
    # )
    _format_axes_no_title(ax)
    fig.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


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


def plot_functional_breakdown_betweenness_vs_closeness(test_df: pd.DataFrame,
                                                       output_path: Path,
                                                       x_lim: tuple = AXIS_LIM,
                                                       y_lim: tuple = (-4, 10)) -> None:
    """
    Scatter of the test set: StdCloseness (x) vs. StdBetweenness (y),
    coloured by functional residue type. Mirrors plot_test_only_functional_breakdown().

    Parameters
    ----------
    test_df     : DataFrame with columns StdCloseness, StdBetweenness, functional_type.
    output_path : Destination PNG file.
    x_lim       : X-axis (StdCloseness) limits. Default (-4, 4).
    y_lim       : Y-axis (StdBetweenness) limits. Default (-4, 10).
    """
    if "functional_type" not in test_df.columns:
        raise ValueError(
            "'functional_type' column not found. "
            "Run add_functional_type() before calling this function."
        )
    if "StdBetweenness" not in test_df.columns:
        raise ValueError(
            "'StdBetweenness' column not found. "
            "Run standardize_betweenness_per_protein() before calling this function."
        )

    counts = test_df["functional_type"].value_counts()
    print(f"plot_functional_breakdown_betweenness_vs_closeness received: {len(test_df):,} rows")
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


def plot_functional_breakdown_betweenness_vs_degree(test_df: pd.DataFrame,
                                                    output_path: Path,
                                                    x_lim: tuple = AXIS_LIM,
                                                    y_lim: tuple = (-4, 10)) -> None:
    """
    Scatter of the test set: StdDegree (x) vs. StdBetweenness (y),
    coloured by functional residue type. Mirrors plot_test_only_functional_breakdown().

    Parameters
    ----------
    test_df     : DataFrame with columns StdDegree, StdBetweenness, functional_type.
    output_path : Destination PNG file.
    x_lim       : X-axis (StdDegree) limits. Default (-4, 4).
    y_lim       : Y-axis (StdBetweenness) limits. Default (-4, 10).
    """
    if "functional_type" not in test_df.columns:
        raise ValueError(
            "'functional_type' column not found. "
            "Run add_functional_type() before calling this function."
        )
    if "StdBetweenness" not in test_df.columns:
        raise ValueError(
            "'StdBetweenness' column not found. "
            "Run standardize_betweenness_per_protein() before calling this function."
        )

    counts = test_df["functional_type"].value_counts()
    print(f"plot_functional_breakdown_betweenness_vs_degree received: {len(test_df):,} rows")
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
            "Run add_functional_type() before calling this function."
        )
    if "StdBetweenness" not in test_df.columns:
        raise ValueError(
            "'StdBetweenness' column not found. "
            "Run standardize_betweenness_per_protein() before calling this function."
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
            "Run add_functional_type() before calling this function."
        )
    if "RCA_zscore" not in test_df.columns:
        raise ValueError(
            "'RCA_zscore' column not found. "
            "Run merge_rca_into_standardized() before calling this function."
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


# ── RCA helpers (ported from 0_4_fig1_recreation_ver4.py) ─────────────────────

def merge_rca_into_standardized(std_df: pd.DataFrame,
                                centrality_dir: Path,
                                exclude_pdb_ids: set = None) -> pd.DataFrame:
    """
    Patch RCA_zscore (and RCA_deltaL, IsCICD) into an already-standardized
    DataFrame by re-reading only those columns from the raw centrality CSVs.

    Use this when append_rca_to_centrality() was run *after* the last
    main_size_controlled() call, so the saved *_standardized.csv lacks the
    RCA columns even though the raw centrality CSVs already have them.

    Join key: pdb_id + Node (both present in std_df and in every
    *_centrality.csv file; Node is globally unique within a protein).

    Parameters
    ----------
    std_df          : pd.DataFrame
        Output of pd.read_csv(ver_dir / "test_standardized.csv").
        Must contain columns 'pdb_id' and 'Node'.
    centrality_dir  : Path
        Directory that contains the raw *_centrality.csv files.
    exclude_pdb_ids : set of str, optional
        PDB IDs to skip (case-insensitive). Pass the same set used when
        loading the standardized CSV so the two DataFrames stay in sync.

    Returns
    -------
    pd.DataFrame — std_df with RCA_deltaL, RCA_zscore, IsCICD columns
    added (or overwritten). Rows with no match receive NaN / False.
    """
    centrality_dir  = Path(centrality_dir)
    exclude_pdb_ids = {p.lower() for p in exclude_pdb_ids} if exclude_pdb_ids else set()

    rca_cols = ["RCA_deltaL", "RCA_zscore", "IsCICD"]

    frames = []
    for csv_path in sorted(centrality_dir.glob("*_centrality.csv")):
        pdb_id = csv_path.stem.replace("_centrality", "")
        if pdb_id.lower() in exclude_pdb_ids:
            continue

        raw = pd.read_csv(csv_path, usecols=lambda c: c in (["Node"] + rca_cols))

        missing = [c for c in rca_cols if c not in raw.columns]
        if missing:
            print(f"  WARNING: {pdb_id} centrality CSV missing {missing} — "
                  "run append_rca_to_centrality() first.")
            continue

        raw["pdb_id"] = pdb_id
        frames.append(raw[["pdb_id", "Node"] + rca_cols])

    if not frames:
        raise FileNotFoundError(
            f"No centrality CSVs with RCA columns found in {centrality_dir}. "
            "Run append_rca_to_centrality() to populate them."
        )

    rca_df = pd.concat(frames, ignore_index=True)

    # drop any pre-existing RCA columns in std_df before merging
    std_df = std_df.drop(columns=[c for c in rca_cols if c in std_df.columns],
                         errors="ignore").copy()

    merged = std_df.merge(rca_df, on=["pdb_id", "Node"], how="left")

    n_matched = merged["RCA_zscore"].notna().sum()
    n_total   = len(merged)
    print(f"  merge_rca_into_standardized: matched {n_matched:,} / {n_total:,} rows "
          f"({100 * n_matched / n_total:.1f}%)")
    if n_matched < n_total:
        print(f"  WARNING: {n_total - n_matched:,} rows have no RCA match.")

    return merged


def plot_functional_breakdown_rca_vs_closeness(test_df: pd.DataFrame,
                                               output_path: Path,
                                               x_lim: tuple = AXIS_LIM,
                                               y_lim: tuple = (-4, 10)) -> None:
    """
    Scatter of the test set: StdCloseness (x) vs. RCA_zscore (y),
    coloured by functional residue type.

    A dotted horizontal line at y = 2 marks the CICD threshold
    (del Sol et al. 2006): residues above this line are classified as
    Centrally Interconnected Conserved Determinants.

    Draw order: non-functional rendered first (background), then binding,
    catalytic, and dual on top so the rarest category (dual) is always visible.

    Parameters
    ----------
    test_df     : DataFrame with columns StdCloseness, RCA_zscore, functional_type.
    output_path : Destination PNG file.
    x_lim       : X-axis (StdCloseness) limits. Default (-4, 4).
    y_lim       : Y-axis (RCA_zscore) limits. Default (-4, 10).
    """
    if "functional_type" not in test_df.columns:
        raise ValueError(
            "'functional_type' column not found. "
            "Run add_functional_type() before calling this function."
        )
    if "RCA_zscore" not in test_df.columns:
        raise ValueError(
            "'RCA_zscore' column not found. "
            "Run merge_rca_into_standardized() before calling this function."
        )

    counts = test_df["functional_type"].value_counts()
    print(f"plot_functional_breakdown_rca_vs_closeness received: {len(test_df):,} rows")
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
            subset=["StdCloseness", "RCA_zscore"]
        )
        if sub.empty:
            continue
        n = len(sub)
        ax.scatter(
            sub["StdCloseness"],
            sub["RCA_zscore"],
            c=FUNCTIONAL_TYPE_COLORS[ftype],
            label=f"{FUNCTIONAL_TYPE_LABELS[ftype]} (n={n:,})",
            **scatter_params[ftype],
        )

    ax.legend(fontsize=FONT["size_legend"], loc="upper left")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.axhline(0, color="grey",  linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey",  linewidth=0.6, linestyle="--")
    # del Sol et al. (2006) CICD threshold
    ax.axhline(2, color="black", linewidth=0.8, linestyle=":",
               label="CICD threshold (z = 2)")
    ax.set_xlabel("Standardized Closeness Centrality", fontsize=FONT["size_label"])
    ax.set_ylabel("RCA Z-score",                       fontsize=FONT["size_label"])
    fig.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── paths — edit these as needed ──────────────────────────────────────
    ver_dir              = DATA_DIR_2 / "figures" / "controlled_fig1_recreation_ver0.7"
    test_csv             = ver_dir / "test_standardized.csv"
    detailed_csv         = DATA_DIR_2 / "biolip_functional_residues_detailed.csv"
    test_centrality_dir  = DATA_DIR_2 / "Centralities"
    exclude_test_pdb_ids = {'7ara', '4q95'}
    # ─────────────────────────────────────────────────────────────────────

    # 1. Load standardized test data
    test_df = pd.read_csv(test_csv)

    # 2. Standardize betweenness (not saved in the CSV by main_size_controlled)
    test_df = standardize_betweenness_per_protein(test_df)

    # 3. Patch RCA columns from raw centrality CSVs
    test_df = merge_rca_into_standardized(
        test_df,
        test_centrality_dir,
        exclude_pdb_ids=exclude_test_pdb_ids,
    )

    # 4. Load functional type labels from detailed BioLiP CSV
    label_df = load_functional_type_labels(detailed_csv)

    # 5. Join functional_type onto test_df
    test_df = add_functional_type(test_df, label_df)

    # # 6. Closeness vs. Degree breakdown (original)
    # plot_test_only_functional_breakdown(
    #     test_df,
    #     ver_dir / "fig1_test_functional_breakdown.png",
    # )

    # # 7. Betweenness vs. Closeness breakdown
    # plot_functional_breakdown_betweenness_vs_closeness(
    #     test_df,
    #     ver_dir / "fig1_test_functional_breakdown_betweenness_vs_closeness.png",
    # )

    # # 8. Betweenness vs. Degree breakdown
    # plot_functional_breakdown_betweenness_vs_degree(
    #     test_df,
    #     ver_dir / "fig1_test_functional_breakdown_betweenness_vs_degree.png",
    # )

    # # 9. RCA z-score vs. StdCloseness breakdown (coloured by functional type)
    # plot_functional_breakdown_rca_vs_closeness(
    #     test_df,
    #     ver_dir / "fig1_test_functional_breakdown_rca_vs_closeness.png",
    #     y_lim=(-4, 10),
    # )

    # 10. StdDegree (y) vs. StdBetweenness (x) breakdown
    plot_functional_breakdown_degree_vs_betweenness(
        test_df,
        ver_dir / "fig1_test_functional_breakdown_degree_vs_betweenness.png",
        x_lim=(-4, 10),
        y_lim=(-4, 4),
    )

    # 11. StdDegree (y) vs. RCA z-score (x) breakdown (no CICD threshold line)
    plot_functional_breakdown_degree_vs_rca(
        test_df,
        ver_dir / "fig1_test_functional_breakdown_degree_vs_rca.png",
        x_lim=(-4, 10),
        y_lim=(-4, 4),
    )

    print("\nDone.")
