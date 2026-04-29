# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Shared constants
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

EC_COLORS = {
    1: "#E69F00",
    2: "#56B4E9",
    3: "#009E73",
    4: "#F0E442",
    5: "#0072B2",
    6: "#D55E00",
    7: "#CC79A7",
}


# ---------------------------------------------------------------------------
# Reusable plotting function
# ---------------------------------------------------------------------------

def plot_ec_distribution(
    df: "pd.DataFrame",
    ec_col: str,
    output_fig: "Path",
    title: str = "Distribution of Enzyme Commission (EC) Main Classes",
) -> None:
    """
    Plot a bar chart of EC main-class distribution and save to disk.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the EC number column.
    ec_col : str
        Name of the column holding EC numbers (e.g. "ec_number" or
        "prio_ec_number").
    output_fig : Path
        Full path (including filename) where the figure will be saved.
    title : str
        Title displayed on the figure.
    """
    # Extract the first digit of the EC number (e.g. "3.5.4.13" → 3)
    ec_class = (
        df[ec_col]
        .astype(str)
        .str.strip()
        .str.split(".")
        .str[0]
    )
    ec_class = pd.to_numeric(ec_class, errors="coerce")
    valid_mask = ec_class.notna() & ec_class.between(1, 7)
    ec_class = ec_class[valid_mask].astype(int)

    # Count occurrences per class; fill any missing classes with 0
    counts = ec_class.value_counts().sort_index()
    all_classes = pd.Series(0, index=range(1, 8))
    counts = (all_classes + counts).fillna(0).astype(int)

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    bars = ax.bar(
        counts.index,
        counts.values,
        color=[EC_COLORS[i] for i in counts.index],
        edgecolor="white",
        linewidth=1.2,
        width=0.65,
        zorder=3,
    )

    # Value labels on top of each bar
    for bar, val in zip(bars, counts.values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.015,
                str(val),
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#333333",
            )

    ax.set_xticks(range(1, 8))
    ax.set_xticklabels(
        [EC_CLASSES[i] for i in range(1, 8)],
        fontsize=10, color="#333333",
    )
    ax.set_ylabel("Number of Entries", fontsize=12, color="#333333", labelpad=10)
    ax.set_xlabel("Enzyme Commission (EC) Class", fontsize=12, color="#333333", labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", color="#222222", pad=16)

    # Grid on y-axis only
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="#CCCCCC", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.tick_params(colors="#555555")

    # Total count annotation
    total = counts.sum()
    ax.text(
        0.99, 0.97,
        f"n = {total:,} entries",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10, color="#666666",
        style="italic",
    )

    plt.tight_layout()
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_fig, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Figure saved to: {output_fig}")
    print(f"\nEC Class Counts ({ec_col}):")
    for ec, count in counts.items():
        label = EC_CLASSES[ec].replace("\n", " - ")
        print(f"  {label}: {count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    FIGURES_DIR = BASE_DIR / "5_PDB_info_exploration" / "figures"

    # ------------------------------------------------------------------
    # Figure 1 – original ec_number distribution (pdb_ids_with_ec.csv)
    # ------------------------------------------------------------------
    # 4/21/26
    ## only want entries with an EC (enzymes only)
    # INPUT_CSV  = DATA_DIR / "FASTA" / "oligo_bio_func_residues_with_complex_type.csv"
    # OUTPUT_CSV = DATA_DIR / "FASTA" / "pdb_ids_with_ec.csv"

    # df = pd.read_csv(INPUT_CSV)
    # has_ec = (
    #     df["ec_number"].notna() & 
    #     df["ec_number"].astype(str).str.strip().ne("") & 
    #     df["ec_number"].astype(str).str.strip().ne("nan") 
    # )
    # filtered_df = df[has_ec].reset_index(drop=True)
    # filtered_df.to_csv(OUTPUT_CSV, index=False)

    # print(f"Total rows       : {len(df)}")
    # print(f"Rows with EC     : {len(filtered_df)}")
    # print(f"Saved to         : {OUTPUT_CSV}")

    # df_ec = pd.read_csv(DATA_DIR / "FASTA" / "pdb_ids_with_ec.csv")

    # plot_ec_distribution(
    #     df=df_ec,
    #     ec_col="ec_number",
    #     output_fig=FIGURES_DIR / "enzyme_distribution.png",
    #     title="Distribution of Enzyme Commission (EC) Main Classes",
    # )

    # ------------------------------------------------------------------
    # Figure 2 – priority EC number distribution
    #            (CSV with prio_ec_number / prio_ec_source columns)
    # ------------------------------------------------------------------
    # 4/21/26
    ## Uses the prioritised EC assignment:
    ##   - PDB Primary Data assignment is used when available
    ##   - UniProt assignment used otherwise
    ## Majority of papers defer to the UniProt assignment; inconsistencies
    ## between the two sources are possible and tracked in prio_ec_source.

    INPUT_CSV_PRIO = DATA_DIR / "enzymes" / "resolved_prio_ec.csv"

    # df_prio = pd.read_csv(INPUT_CSV_PRIO)

    # plot_ec_distribution(
    #     df=df_prio,
    #     ec_col="prio_ec_number",
    #     output_fig= FIGURES_DIR / "enzyme_distribution_prio_ec.png",
    #     title="Distribution of EC Main Classes\n(Priority EC Assignment)",
    # )

    # # Optional: show breakdown by source
    # if "prio_ec_source" in df_prio.columns:
    #     print("\nEntries by priority EC source:")
    #     print(df_prio["prio_ec_source"].value_counts().to_string())

    ## only looking at monomers

    # df_prio = pd.read_csv(INPUT_CSV_PRIO)

    # # Filter to monomers only
    # df_prio_mono = df_prio[df_prio["oligomeric_state"] == "monomeric"].reset_index(drop=True)

    # plot_ec_distribution(
    #     df=df_prio_mono,
    #     ec_col="prio_ec_number",
    #     output_fig=FIGURES_DIR / "enzyme_distribution_prio_ec_monomers.png",
    #     title="Distribution of EC Main Classes\n(Priority EC, Monomers Only)",
    # )
    ## Figure 3 - 4/21/ - non-mutants only
    filename = DATA_DIR / "enzymes" / "enzymes_no_mutants.csv"

    df_prio = pd.read_csv(filename)

    plot_ec_distribution(
        df=df_prio,
        ec_col="prio_ec_number",
        output_fig=FIGURES_DIR / "enzyme_distribution_prio_ec_no_mutants.png",
        title="Distribution of EC Main Classes\n(Priority EC and No Mutants)",
    )
