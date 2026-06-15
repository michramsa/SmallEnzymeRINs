# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *
import matplotlib.pyplot as plt

from figure_style_ver2 import apply_style, save_fig, COLOR_CYCLE, FIG_SIZE, FONT, LAYOUT, DPI, COLORS


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

# Colors sourced from figure_style.py COLOR_CYCLE for consistency
EC_COLORS = {
    1: COLOR_CYCLE[0],   # blue
    2: COLOR_CYCLE[1],   # orange
    3: COLOR_CYCLE[2],   # green
    4: COLOR_CYCLE[3],   # purple
    5: COLOR_CYCLE[4],   # teal
    6: COLOR_CYCLE[5],   # red
    7: COLOR_CYCLE[6],   # yellow
}


# ---------------------------------------------------------------------------
# Helper: extract EC main class counts from a DataFrame
# ---------------------------------------------------------------------------

def _get_ec_counts(df: pd.DataFrame, ec_col: str) -> pd.Series:
    """
    Extract the first digit of each EC number and return a count Series
    indexed 1–7, with 0 for any missing classes.

    Entries whose EC number does not parse to a valid integer in 1–7
    (e.g. partial EC numbers like "3.-.-.-" or missing values) are
    dropped from the counts but reported to stdout.

    Parameters
    ----------
    df     : pd.DataFrame  Must contain ec_col.
    ec_col : str           Column holding full EC numbers (e.g. "3.5.4.13").

    Returns
    -------
    pd.Series  Integer counts indexed 1–7.
    """
    ec_class = (
        df[ec_col]
        .astype(str)
        .str.strip()
        .str.split(".")
        .str[0]
    )
    ec_class = pd.to_numeric(ec_class, errors="coerce")
    valid_mask = ec_class.notna() & ec_class.between(1, 7)

    # Report any entries that will be excluded from the bar chart
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        cols = ["pdb_id", ec_col] if "pdb_id" in df.columns else [ec_col]
        dropped_df = df.loc[~valid_mask, cols]
        print(f"NOTE: {n_dropped} entry/entries have unparseable EC numbers "
              f"and are excluded from bar counts:")
        for _, row in dropped_df.iterrows():
            print(f"  {row.to_dict()}")

    ec_class = ec_class[valid_mask].astype(int)
    counts = ec_class.value_counts().sort_index()
    all_classes = pd.Series(0, index=range(1, 8))
    return (all_classes + counts).fillna(0).astype(int)


# ---------------------------------------------------------------------------
# Reusable plotting function
# ---------------------------------------------------------------------------

def plot_ec_distribution(
    df: pd.DataFrame,
    ec_col: str,
    output_fig: Path,
    title: str = "Distribution of Enzyme Commission (EC) Main Classes",
    n_total: int = None,
) -> None:
    """
    Plot a bar chart of EC main-class distribution and save to disk.
    Styling follows figure_style.py conventions.

    Parameters
    ----------
    df         : pd.DataFrame  Must contain ec_col.
    ec_col     : str           Column holding EC numbers (e.g. "prio_ec_number").
    output_fig : Path          Full output path including filename.
    title      : str           Figure title.
    n_total    : int, optional
        The true number of proteins to display in the "n = " annotation.
        Defaults to len(df) when not provided. Pass this explicitly when
        the DataFrame has already been deduplicated upstream, to ensure
        the annotation matches the actual sample size rather than the
        number of parseable EC entries.
    """
    apply_style()

    counts = _get_ec_counts(df, ec_col)

    # Use the explicitly supplied total if provided, otherwise fall back
    # to len(df) so that entries with unparseable EC numbers are still
    # counted in the annotation.
    if n_total is None:
        n_total = len(df)

    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])

    bars = ax.bar(
        counts.index,
        counts.values,
        color=[EC_COLORS[i] for i in counts.index],
        edgecolor=COLORS["white"],
        linewidth=LAYOUT["spine_linewidth"],
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
                fontsize=FONT["size_tick"],
                fontweight="bold",
                color=COLORS["dark_gray"],
            )

    ax.set_xticks(range(1, 8))
    ax.set_xticklabels(
        [EC_CLASSES[i] for i in range(1, 8)],
        fontsize=FONT["size_tick"],
    )
    ax.set_ylabel("Number of Entries", fontsize=FONT["size_label"])
    ax.set_xlabel("Enzyme Commission (EC) Class", fontsize=FONT["size_label"])
    ax.set_title(title, fontsize=FONT["size_title"], fontweight=FONT["weight_title"])

    # Total count annotation — uses n_total, not counts.sum(), so entries
    # with unparseable EC numbers are included in the displayed total.
    ax.text(
        0.99, 0.97,
        f"n = {n_total:,} proteins",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=FONT["size_legend"],
        color=COLORS["mid_gray"],
        style="italic",
    )

    fig.tight_layout(pad=LAYOUT["tight_pad"])
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_fig)
    plt.close(fig)

    print(f"Figure saved to: {output_fig}")
    print(f"\nEC Class Counts ({ec_col}):")
    for ec, count in counts.items():
        label = EC_CLASSES[ec].replace("\n", " - ")
        print(f"  {label}: {count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    FIGURES_DIR = DATA_DIR_2 / "figures" / "controlled_fig1_recreation_ver0.7"

    # Shared input CSV containing priority EC assignments for all enzymes
    RESOLVED_PRIO_EC_CSV = DATA_DIR_2 / "resolved_prio_ec.csv"

    # ------------------------------------------------------------------
    # Figure 1 – non-mutants only (full BioLiP enzyme set)
    # ------------------------------------------------------------------
    # NO_MUTANTS_CSV = DATA_DIR / "enzymes" / "enzymes_no_mutants.csv"

    # df_no_mutants = pd.read_csv(NO_MUTANTS_CSV)

    # plot_ec_distribution(
    #     df=df_no_mutants,
    #     ec_col="prio_ec_number",
    #     output_fig=FIGURES_DIR / "enzyme_distribution_prio_ec_no_mutants.png",
    #     title="Distribution of EC Main Classes\n(Priority EC, No Mutants)",
    # )

    # ------------------------------------------------------------------
    # Figure 2 – test set (small enzyme set used in RIN analysis)
    #
    # PDB IDs are drawn from the unique pdb_id values in
    # test_standardized_ver4.csv — the verified, processed residue-level
    # table produced by 8_fig1_recreation_ver5.py.  One row per residue,
    # so we deduplicate before joining to resolved_prio_ec.csv.
    # ------------------------------------------------------------------

    # ## updated 5/20/26
    # TEST_STANDARDIZED_CSV = DATA_DIR_2 / "figures" / "fig1_recreation" / "test_standardized.csv"

    # # Collect the unique PDB IDs present in the test set (uppercase for matching)
    # df_test_std = pd.read_csv(TEST_STANDARDIZED_CSV, usecols=["pdb_id"])
    # test_pdb_ids = set(df_test_std["pdb_id"].str.upper().unique())
    # print(f"\nUnique PDB IDs in test set: {len(test_pdb_ids)}")

    # # Filter the priority EC table to only those PDB IDs, then deduplicate
    # # on pdb_id so each protein is counted once regardless of chain count.
    # # keep="first" retains the first chain row, which carries the EC assignment.
    # df_prio = pd.read_csv(RESOLVED_PRIO_EC_CSV)
    # prio_ids = set(df_prio["pdb_id"].str.upper().unique())

    # # Diagnose any test set IDs that have no match in resolved_prio_ec.csv
    # unmatched = test_pdb_ids - prio_ids
    # if unmatched:
    #     print(f"WARNING: {len(unmatched)} test set PDB ID(s) not found in resolved_prio_ec.csv:")
    #     for uid in sorted(unmatched):
    #         print(f"  {uid}")

    # df_test = (
    #     df_prio[df_prio["pdb_id"].str.upper().isin(test_pdb_ids)]
    #     .drop_duplicates(subset="pdb_id", keep="first")
    #     .reset_index(drop=True)
    # )
    # print(f"Matched proteins in resolved_prio_ec.csv: {len(df_test)}")

    # plot_ec_distribution(
    #     df=df_test,
    #     ec_col="prio_ec_number",
    #     output_fig=FIGURES_DIR / "enzyme_distribution_test_set.tiff",
    #     title=None,
    #     n_total=len(test_pdb_ids),
    # )


    ## updated 6/1/26
    TEST_STANDARDIZED_CSV = DATA_DIR_2 / "figures" / "controlled_fig1_recreation_ver0.7" / "test_standardized.csv"

    # Collect the unique PDB IDs present in the test set (uppercase for matching)
    df_test_std = pd.read_csv(TEST_STANDARDIZED_CSV, usecols=["pdb_id"])
    test_pdb_ids = set(df_test_std["pdb_id"].str.upper().unique())
    print(f"\nUnique PDB IDs in test set: {len(test_pdb_ids)}")

    # Filter the priority EC table to only those PDB IDs, then deduplicate
    # on pdb_id so each protein is counted once regardless of chain count.
    # keep="first" retains the first chain row, which carries the EC assignment.
    df_prio = pd.read_csv(RESOLVED_PRIO_EC_CSV)
    prio_ids = set(df_prio["pdb_id"].str.upper().unique())

    # Diagnose any test set IDs that have no match in resolved_prio_ec.csv
    unmatched = test_pdb_ids - prio_ids
    if unmatched:
        print(f"WARNING: {len(unmatched)} test set PDB ID(s) not found in resolved_prio_ec.csv:")
        for uid in sorted(unmatched):
            print(f"  {uid}")

    df_test = (
        df_prio[df_prio["pdb_id"].str.upper().isin(test_pdb_ids)]
        .drop_duplicates(subset="pdb_id", keep="first")
        .reset_index(drop=True)
    )
    print(f"Matched proteins in resolved_prio_ec.csv: {len(df_test)}")

    plot_ec_distribution(
        df=df_test,
        ec_col="prio_ec_number",
        output_fig=FIGURES_DIR / "enzyme_distribution_test_set.png",
        title=None,
        n_total=len(test_pdb_ids),
    )
