# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *
import matplotlib.pyplot as plt
import numpy as np
import warnings

from figure_style_ver2 import apply_style, save_fig, FIG_SIZE, FONT, LAYOUT, COLORS

# ---------------------------------------------------------------------------
# EC class labels — matches 0_4_enzyme_distribution_ver4.py
# ---------------------------------------------------------------------------

EC_CLASSES = {
    1: "EC 1\nOxidoreductases",
    2: "EC 2\nTransferases",
    3: "EC 3\nHydrolases",
    4: "EC 4\nLyases",
    5: "EC 5\nIsomerases",
    6: "EC 6\nLigases",
    7: "EC 7\nTranslocases",
}

# ---------------------------------------------------------------------------
# Functional-type colors/labels — matches 0_4_fig1_recreation_functional_breakdown_ver6.py
# ---------------------------------------------------------------------------

FUNCTIONAL_TYPES = ["binding", "catalytic", "dual"]

FUNCTIONAL_TYPE_COLORS = {
    "binding":   "#8E8278",
    "catalytic": "#35302B",
    "dual":      "#D4A017",   # brightened from the near-white project default —
                              # dual is the rarest category and gets lost against
                              # white paper/background otherwise.
}

FUNCTIONAL_TYPE_LABELS = {
    "binding":   "Binding",
    "catalytic": "Catalytic",
    "dual":      "Dual (binding + catalytic)",
}


# ===========================================================================
# EC-class assignment
# ===========================================================================

def add_ec_class(
    biolip_summary_df: pd.DataFrame,
    ec_col: str = "ec_numbers",
) -> pd.DataFrame:
    """
    Derive a single EC main-class (1-7) per protein from biolip_per_pdb_summary.csv.

    ec_numbers holds a ';'-separated, alphabetically-sorted list of every
    distinct EC number BioLiP annotated for that PDB entry (see
    0_6_biolip_lookup_ver1.py). A protein is assigned the main class of the
    FIRST (alphabetically-lowest) EC number in that list. Proteins whose
    listed EC numbers span more than one main class are flagged to stdout
    so you can decide whether to exclude them upstream.

    Parameters
    ----------
    biolip_summary_df : pd.DataFrame  Must contain 'pdb_id' and ec_col.
    ec_col             : str          Column holding ';'-joined EC numbers.

    Returns
    -------
    pd.DataFrame  Copy of biolip_summary_df with an added integer 'ec_class'
                  column (NaN where no valid EC number was found).
    """
    df = biolip_summary_df.copy()

    def _first_class(cell) -> float:
        if pd.isna(cell) or str(cell).strip() == "":
            return np.nan
        ecs = [e.strip() for e in str(cell).split(";") if e.strip()]
        if not ecs:
            return np.nan
        first_digit = ecs[0].split(".")[0]
        try:
            val = int(first_digit)
        except ValueError:
            return np.nan
        return val if 1 <= val <= 7 else np.nan

    df["ec_class"] = df[ec_col].apply(_first_class)

    # flag proteins whose EC numbers span multiple main classes
    def _n_classes(cell) -> int:
        if pd.isna(cell) or str(cell).strip() == "":
            return 0
        classes = set()
        for e in str(cell).split(";"):
            e = e.strip()
            if not e:
                continue
            try:
                classes.add(int(e.split(".")[0]))
            except ValueError:
                continue
        return len(classes)

    multi_class = df[df[ec_col].apply(_n_classes) > 1]
    if not multi_class.empty:
        print(f"NOTE: {len(multi_class)} protein(s) have EC numbers spanning "
              f"multiple main classes; assigned to the first-listed class only:")
        for _, row in multi_class.iterrows():
            print(f"  {row['pdb_id']}: {row[ec_col]}  -> assigned EC {int(row['ec_class'])}")

    n_missing = df["ec_class"].isna().sum()
    if n_missing > 0:
        print(f"NOTE: {n_missing} protein(s) have no parseable EC number and "
              f"will be excluded from the EC-class breakdown.")

    return df


# ===========================================================================
# Residue-level breakdown
# ===========================================================================

def build_functional_breakdown_by_ec(
    residue_df: pd.DataFrame,
    biolip_summary_df: pd.DataFrame,
    functional_type_col: str = "functional_type",
    ec_col: str = "ec_numbers",
) -> pd.DataFrame:
    """
    Join residue-level functional_type labels onto protein-level EC class
    and return a pivoted count table (EC class x functional type).

    Join key: pdb_id, matched case-insensitively (BioLiP stores lower-case
    IDs; test_standardized.csv commonly stores upper-case IDs).

    Parameters
    ----------
    residue_df         : DataFrame with 'pdb_id' and functional_type_col.
                          Rows with functional_type == 'non_functional' are
                          dropped before counting.
    biolip_summary_df  : biolip_per_pdb_summary.csv, loaded as a DataFrame.
    functional_type_col: Column name holding 'binding'/'catalytic'/'dual'/
                          'non_functional'.
    ec_col              : Column in biolip_summary_df holding ';'-joined EC
                          numbers (default 'ec_numbers').

    Returns
    -------
    pd.DataFrame  Index = EC class (1-7), columns = FUNCTIONAL_TYPES,
                  values = residue counts. Missing combinations are 0.
    """
    ec_df = add_ec_class(biolip_summary_df, ec_col=ec_col)
    ec_lookup = ec_df.dropna(subset=["ec_class"]).copy()
    ec_lookup["pdb_id_lower"] = ec_lookup["pdb_id"].astype(str).str.strip().str.lower()
    ec_lookup = ec_lookup.drop_duplicates(subset=["pdb_id_lower"])

    res = residue_df.copy()
    res["pdb_id_lower"] = res["pdb_id"].astype(str).str.strip().str.lower()

    merged = res.merge(
        ec_lookup[["pdb_id_lower", "ec_class"]],
        on="pdb_id_lower",
        how="left",
    )

    n_no_ec = merged["ec_class"].isna().sum()
    if n_no_ec > 0:
        missing_pdbs = merged.loc[merged["ec_class"].isna(), "pdb_id"].nunique()
        print(f"NOTE: {n_no_ec} residue(s) across {missing_pdbs} protein(s) have "
              f"no EC-class match and are excluded from this figure.")

    functional = merged[
        merged[functional_type_col].isin(FUNCTIONAL_TYPES)
        & merged["ec_class"].notna()
    ].copy()
    functional["ec_class"] = functional["ec_class"].astype(int)

    counts = (
        functional
        .groupby(["ec_class", functional_type_col])
        .size()
        .unstack(fill_value=0)
    )

    # ensure all 7 EC classes and all 3 functional types are present
    counts = counts.reindex(index=range(1, 8), columns=FUNCTIONAL_TYPES, fill_value=0)
    return counts


# ===========================================================================
# Plot
# ===========================================================================

def plot_functional_breakdown_by_ec(
    counts: pd.DataFrame,
    output_fig: Path,
    title: str = "Binding, Catalytic, and Dual Residues by Enzyme Class",
) -> None:
    """
    Grouped bar chart: one cluster per EC main class (1-7), with one bar
    per functional type (binding / catalytic / dual) within each cluster.

    Parameters
    ----------
    counts     : pd.DataFrame  Index = EC class (1-7), columns = FUNCTIONAL_TYPES.
                 As returned by build_functional_breakdown_by_ec().
    output_fig : Path  Full output path including filename.
    title      : str   Figure title. Pass None to suppress.
    """
    apply_style()

    ec_indices = list(counts.index)
    n_types = len(FUNCTIONAL_TYPES)
    bar_width = 0.8 / n_types
    x = np.arange(len(ec_indices))

    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])

    for i, ftype in enumerate(FUNCTIONAL_TYPES):
        offsets = x + (i - (n_types - 1) / 2) * bar_width
        vals = counts[ftype].values
        bars = ax.bar(
            offsets,
            vals,
            width=bar_width,
            color=FUNCTIONAL_TYPE_COLORS[ftype],
            edgecolor=COLORS["white"],
            linewidth=LAYOUT["spine_linewidth"],
            label=FUNCTIONAL_TYPE_LABELS[ftype],
            zorder=3,
        )
        max_val = counts.values.max() if counts.values.max() > 0 else 1
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_val * 0.015,
                    str(int(val)),
                    ha="center", va="bottom",
                    fontsize=FONT["size_annot"],
                    color=COLORS["dark_gray"],
                )

    ax.set_xticks(x)
    ax.set_xticklabels([EC_CLASSES[i] for i in ec_indices], fontsize=FONT["size_tick"])
    ax.set_xlabel("Enzyme Commission (EC) Class", fontsize=FONT["size_label"])
    ax.set_ylabel("Number of Residues", fontsize=FONT["size_label"])

    if title:
        ax.set_title(title, fontsize=FONT["size_title"], fontweight=FONT["weight_title"])

    ax.legend(fontsize=FONT["size_legend"], loc="upper right")

    n_total = int(counts.values.sum())
    ax.text(
        0.01, 0.97,
        f"n = {n_total:,} residues",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=FONT["size_legend"],
        color=COLORS["mid_gray"],
        style="italic",
    )

    fig.tight_layout(pad=LAYOUT["tight_pad"])
    output_fig = Path(output_fig)
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_fig)
    plt.close(fig)

    print(f"Figure saved to: {output_fig}")
    print("\nResidue counts by EC class and functional type:")
    print(counts.to_string())
    print(f"\nColumn totals — binding: {counts['binding'].sum():,}  "
          f"catalytic: {counts['catalytic'].sum():,}  "
          f"dual: {counts['dual'].sum():,}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Configurable paths  (edit these to match your directory structure)
    # ------------------------------------------------------------------

    TEST_STANDARDIZED_CSV = FIGS_DIR / "test_standardized.csv"
    BIOLIP_SUMMARY_CSV    = DATA_DIR_3 / "test" / "biolip_lookup" / "biolip_per_pdb_summary.csv"

    OUTPUT_FIG  = FIGS_DIR / "enzyme_dist_no_mc" / "functional_breakdown_by_ec_class.png"

    # Column names — change if yours differ
    FUNCTIONAL_TYPE_COL = "functional_type"
    EC_COL               = "ec_numbers"

    # ------------------------------------------------------------------

    print("=" * 60)
    print("FUNCTIONAL RESIDUE BREAKDOWN BY EC CLASS")
    print("=" * 60)

    residue_df = pd.read_csv(TEST_STANDARDIZED_CSV)
    biolip_summary_df = pd.read_csv(BIOLIP_SUMMARY_CSV)

    if FUNCTIONAL_TYPE_COL not in residue_df.columns:
        raise ValueError(
            f"'{FUNCTIONAL_TYPE_COL}' column not found in {TEST_STANDARDIZED_CSV}. "
            f"Run add_functional_type() (0_4_fig1_recreation_functional_breakdown_ver6.py) "
            f"before using this script."
        )

    counts = build_functional_breakdown_by_ec(
        residue_df=residue_df,
        biolip_summary_df=biolip_summary_df,
        functional_type_col=FUNCTIONAL_TYPE_COL,
        ec_col=EC_COL,
    )

    plot_functional_breakdown_by_ec(
        counts=counts,
        output_fig=OUTPUT_FIG,
    )
