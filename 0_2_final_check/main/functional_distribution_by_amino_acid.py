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
# Canonical amino-acid order — matches STANDARD_AA / CANONICAL_AA elsewhere
# in the project (0_4_enzyme_distribution_ver4.py, functional_breakdown scripts)
# ---------------------------------------------------------------------------

CANONICAL_AA_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]

# ---------------------------------------------------------------------------
# Colors — non_functional reuses the project-standard gray; functional
# reuses the darkest tone from FUNCTIONAL_TYPE_COLORS (catalytic) so it
# stays visually consistent with the other functional-breakdown figures.
# ---------------------------------------------------------------------------

BINARY_COLORS = {
    "functional":     "#35302B",
    "non_functional": "#CCCCCC",
}

BINARY_LABELS = {
    "functional":     "Functional (binding + catalytic + dual)",
    "non_functional": "Non-functional",
}


# ===========================================================================
# Data prep
# ===========================================================================

def build_functional_distribution_by_aa(
    residue_df: pd.DataFrame,
    functional_type_col: str = "functional_type",
    residue_col: str = "Residue",
) -> pd.DataFrame:
    """
    Count functional vs. non-functional residues for each of the 20 standard
    amino acid types.

    'Functional' collapses binding/catalytic/dual into a single category;
    'non_functional' is taken directly from functional_type_col. Residues
    whose Residue value is not one of the 20 canonical three-letter codes
    (e.g. SEC, PYL, or HETATM leftovers) are dropped and reported.

    Parameters
    ----------
    residue_df           : DataFrame with residue_col and functional_type_col.
    functional_type_col  : Column holding 'binding'/'catalytic'/'dual'/'non_functional'.
    residue_col           : Column holding the three-letter amino acid code.

    Returns
    -------
    pd.DataFrame  Index = CANONICAL_AA_ORDER, columns = ['functional', 'non_functional'],
                  values = residue counts.
    """
    df = residue_df.copy()
    df[residue_col] = df[residue_col].astype(str).str.strip().str.upper()

    non_canonical = df[~df[residue_col].isin(CANONICAL_AA_ORDER)]
    if not non_canonical.empty:
        dropped_types = non_canonical[residue_col].value_counts()
        print(f"NOTE: {len(non_canonical)} residue(s) with non-canonical "
              f"Residue codes excluded from this figure:")
        for code, n in dropped_types.items():
            print(f"  {code}: {n}")
        df = df[df[residue_col].isin(CANONICAL_AA_ORDER)]

    df["is_functional_binary"] = np.where(
        df[functional_type_col] == "non_functional", "non_functional", "functional"
    )

    counts = (
        df
        .groupby([residue_col, "is_functional_binary"])
        .size()
        .unstack(fill_value=0)
    )
    counts = counts.reindex(index=CANONICAL_AA_ORDER, columns=["functional", "non_functional"], fill_value=0)
    return counts


# ===========================================================================
# Plot
# ===========================================================================

def plot_functional_distribution_by_aa(
    counts: pd.DataFrame,
    output_fig: Path,
    log_scale: bool = True,
) -> None:
    """
    Grouped bar chart: one cluster per amino acid type (20 total), with a
    'functional' bar and a 'non_functional' bar in each cluster.

    No on-graph title is drawn (add one in a caption instead, if needed).
    The legend is placed below the plot (not overlapping the bars, and not
    widening the figure the way a right-side legend would).

    Parameters
    ----------
    counts     : pd.DataFrame  Index = amino acid codes, columns = ['functional', 'non_functional'].
                 As returned by build_functional_distribution_by_aa().
    output_fig : Path  Full output path including filename.
    log_scale  : bool  Use a log-scaled y-axis (default True). Functional
                 residues are typically 10-50x rarer than non-functional
                 ones for every amino acid type, so a linear axis makes the
                 functional bars nearly invisible. Set False for a linear
                 axis showing true proportional heights instead.
    """
    apply_style()

    aa_labels = list(counts.index)
    categories = ["functional", "non_functional"]
    n_cats = len(categories)
    bar_width = 0.55 / n_cats
    bar_pitch = bar_width * 1.15
    x = np.arange(len(aa_labels))

    # Taller than the standard full_width figure to leave room below the
    # x-axis (rotated tick labels) for the legend, without widening the plot.
    figsize = (FIG_SIZE["full_width"][0], FIG_SIZE["full_width"][1] + 1.3)
    fig, ax = plt.subplots(figsize=figsize)

    for i, cat in enumerate(categories):
        offsets = x + (i - (n_cats - 1) / 2) * bar_pitch
        vals = counts[cat].values
        ax.bar(
            offsets,
            vals,
            width=bar_width,
            color=BINARY_COLORS[cat],
            edgecolor=COLORS["white"],
            linewidth=LAYOUT["spine_linewidth"],
            label=BINARY_LABELS[cat],
            zorder=3,
        )

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Number of Residues (log scale)", fontsize=FONT["size_label"])
    else:
        ax.set_ylabel("Number of Residues", fontsize=FONT["size_label"])

    ax.set_xticks(x)
    ax.set_xticklabels(aa_labels, fontsize=FONT["size_tick"], rotation=45, ha="right")
    ax.set_xlabel("Amino Acid Type", fontsize=FONT["size_label"])

    n_total = int(counts.values.sum())
    ax.legend(
        fontsize=FONT["size_legend"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=2,
        title=f"n = {n_total:,} residues",
        title_fontsize=FONT["size_legend"],
        frameon=True,
    )

    fig.tight_layout(pad=LAYOUT["tight_pad"])
    output_fig = Path(output_fig)
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_fig)
    plt.close(fig)

    print(f"Figure saved to: {output_fig}")
    print("\nResidue counts by amino acid type:")
    print(counts.to_string())
    print(f"\nColumn totals — functional: {counts['functional'].sum():,}  "
          f"non_functional: {counts['non_functional'].sum():,}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Configurable paths  (edit these to match your directory structure)
    # ------------------------------------------------------------------

    TEST_STANDARDIZED_CSV = FIGS_DIR / "test_standardized.csv"
    OUTPUT_FIG = FIGS_DIR / "enzyme_dist_no_mc" / "functional_distribution_by_amino_acid.png"

    # Column names — change if yours differ
    FUNCTIONAL_TYPE_COL = "functional_type"
    RESIDUE_COL = "Residue"

    # Use a log-scaled y-axis (recommended — see log_scale docstring above)
    LOG_SCALE = True

    # ------------------------------------------------------------------

    print("=" * 60)
    print("FUNCTIONAL RESIDUE DISTRIBUTION BY AMINO ACID TYPE")
    print("=" * 60)

    residue_df = pd.read_csv(TEST_STANDARDIZED_CSV)

    if FUNCTIONAL_TYPE_COL not in residue_df.columns:
        raise ValueError(
            f"'{FUNCTIONAL_TYPE_COL}' column not found in {TEST_STANDARDIZED_CSV}. "
            f"Run add_functional_type() (0_4_fig1_recreation_functional_breakdown_ver6.py) "
            f"before using this script."
        )

    counts = build_functional_distribution_by_aa(
        residue_df=residue_df,
        functional_type_col=FUNCTIONAL_TYPE_COL,
        residue_col=RESIDUE_COL,
    )

    plot_functional_distribution_by_aa(
        counts=counts,
        output_fig=OUTPUT_FIG,
        log_scale=LOG_SCALE,
    )
