# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

import matplotlib.pyplot as plt

if __name__ == "__main__":
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

    # 4/21/2026
    # figures to show distribution within dataset
    INPUT_CSV  = DATA_DIR / "FASTA" / "pdb_ids_with_ec.csv"
    OUTPUT_FIG = BASE_DIR / "5_PDB_info_exploration" / "figures" / "enzyme_distribution.png"
    
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
    
    df = pd.read_csv(INPUT_CSV)
    
    # Extract the first digit of the EC number (e.g. "3.5.4.13" → 3)
    df["ec_class"] = (
        df["ec_number"]
        .astype(str)
        .str.strip()
        .str.split(".")
        .str[0]
    )
    
    # Keep only valid integer EC classes 1–7
    df["ec_class"] = pd.to_numeric(df["ec_class"], errors="coerce")
    df = df.dropna(subset=["ec_class"])
    df["ec_class"] = df["ec_class"].astype(int)
    df = df[df["ec_class"].between(1, 7)]
    
    # Count occurrences per class
    counts = df["ec_class"].value_counts().sort_index()
    
    # Fill in any missing EC classes with 0 so all 7 are shown
    all_classes = pd.Series(0, index=range(1, 8))
    counts = (all_classes + counts).fillna(0).astype(int)
    
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
    ax.set_title(
        "Distribution of Enzyme Commission (EC) Main Classes",
        fontsize=14, fontweight="bold", color="#222222", pad=16,
    )
    
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
    plt.savefig(OUTPUT_FIG, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Figure saved to: {OUTPUT_FIG}")
    print("\nEC Class Counts:")
    for ec, count in counts.items():
        label = EC_CLASSES[ec].replace("\n", " - ")
        print(f"  {label}: {count}")

    ## inconsistencies possible between PDB primary data EC assignment and the UniProt assignment
    ## majority of papers tend to defer to the UniProt assignment 
    