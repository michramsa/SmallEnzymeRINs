# Purpose: Recreate Amitai et al. Figure 1 (standardized closeness vs. standardized degree)
# using the old PISCES set as the reference population, with the small enzyme test set overlaid.
#
# Standardization follows Amitai et al. Methods exactly:
#
#   CLOSENESS: z-scored per protein.
#       std_closeness(residue) = (closeness - protein_mean_closeness) / protein_std_closeness
#
#   DEGREE: z-scored per amino acid type, using reference population statistics.
#       std_degree(residue) = (raw_degree - aa_type_mean) / aa_type_std
#       where mean/std are computed across ALL residues of that amino acid type
#       in the reference (PISCES) set.
#       NOTE: uses raw integer Degree counts (not normalized DegreeCentrality).
#
# Only the 20 canonical amino acids are included; non-standard residues are dropped.
#
# RSA join: SASA CSVs have columns Model, Chain, Residue, Residue ID, RSA-empirical.
#           Centrality CSVs have columns Chain, Residue number.
#           Join key: Chain + Residue number (centrality) == Chain + Residue ID (sasa).
#
# RSA coloring (RSA-empirical):
#   Highly exposed: RSA > 0.5  (orange)
#   Core:           RSA == 0.0 (green)
#   Other:          0.0 < RSA <= 0.5 (blue)
#
# Output: one figure file per plot type (reference only, overlay).
#         one CSV of per-amino-acid-type degree stats from the reference set.

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
# 20 canonical amino acids — only these are included in analysis
CANONICAL_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}


# ── data loading ──────────────────────────────────────────────────────────────

def load_centrality_dir(centrality_dir: Path, sasa_dir: Path = None,
                        exclude_pdb_ids: set = None) -> pd.DataFrame:
    """
    Load all *_centrality.csv files from a directory into one concatenated DataFrame.
    Filters to canonical amino acids only.
    Optionally joins RSA-empirical values from matching *_sasa.csv files.

    Centrality CSV columns used: Node, Residue, Chain, Residue number, Degree, ClosenessCentrality
    SASA CSV columns used:       Chain, Residue ID, RSA-empirical
    Join key: Chain + Residue number (centrality) <-> Chain + Residue ID (sasa)

    Parameters
    ----------
    exclude_pdb_ids : set of str, optional
        PDB IDs to skip entirely before reading their CSV files.
        Matching is case-insensitive (e.g. {'7ara', '4q95'} or {'7ARA', '4Q95'}).
    """
    centrality_dir  = Path(centrality_dir)
    exclude_pdb_ids = {p.lower() for p in exclude_pdb_ids} if exclude_pdb_ids else set()
    frames = []

    for csv_path in sorted(centrality_dir.glob("*_centrality.csv")):
        # derive pdb_id from filename BEFORE reading the file so we can skip early
        pdb_id = csv_path.stem.replace("_centrality", "")

        if pdb_id.lower() in exclude_pdb_ids:
            print(f"  Skipping excluded PDB: {pdb_id}")
            continue

        df = pd.read_csv(csv_path)
        df["pdb_id"] = pdb_id

        # keep only canonical amino acids
        df = df[df["Residue"].isin(CANONICAL_AA)].copy()

        # optional RSA join
        if sasa_dir is not None:
            sasa_path = Path(sasa_dir) / f"{pdb_id}_sasa.csv"
            if sasa_path.exists():
                sasa_df = pd.read_csv(sasa_path)

                # keep only canonical AAs in SASA too
                sasa_df = sasa_df[sasa_df["Residue"].isin(CANONICAL_AA)].copy()
            
                # # ── diagnostic: check for duplicate residue IDs in SASA ──────────
                # dupes = sasa_df.duplicated(subset=["Chain", "Residue ID"], keep=False)
                # if dupes.any():
                #     print(f"  WARNING: {pdb_id} SASA has {dupes.sum()} duplicate "
                #         f"(Chain, Residue ID) rows — likely multiple models.")
                #     print(sasa_df[dupes][["Model", "Chain", "Residue ID"]].head(10).to_string(index=False))
                # # ─────────────────────────────────────────────────────────────────

                # keep only the first model to avoid row duplication on merge
                sasa_df = sasa_df[sasa_df["Model"] == sasa_df["Model"].min()].copy()

                # normalise join key types
                df["Residue number"]  = df["Residue number"].astype(int)
                sasa_df["Residue ID"] = sasa_df["Residue ID"].astype(int)

                df = df.merge(
                    sasa_df[["Chain", "Residue ID", "RSA-empirical"]],
                    left_on=["Chain", "Residue number"],
                    right_on=["Chain", "Residue ID"],
                    how="left",
                )
                df.drop(columns=["Residue ID"], inplace=True)
            else:
                print(f"  No SASA file found for {pdb_id}, RSA will be absent.")

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


# ── standardization ───────────────────────────────────────────────────────────

def standardize_closeness_per_protein(df: pd.DataFrame) -> pd.DataFrame:
    """
    Amitai method for closeness: z-score within each protein individually.
    Adds StdCloseness column.
    Proteins with std == 0 (degenerate case) get StdCloseness = 0.
    """
    df = df.copy()

    def _zscore(grp):
        mu  = grp["ClosenessCentrality"].mean()
        sig = grp["ClosenessCentrality"].std(ddof=0)
        grp = grp.copy()
        grp["StdCloseness"] = 0.0 if sig == 0 else (grp["ClosenessCentrality"] - mu) / sig
        return grp

    return df.groupby("pdb_id", group_keys=False).apply(_zscore)


def compute_degree_stats_by_aa(ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    Amitai method for degree: compute mean and std of raw Degree
    grouped by amino acid type across the entire reference population.

    Returns a DataFrame with columns: Residue, deg_mean, deg_std.
    """
    stats = (
        ref_df.groupby("Residue")["Degree"]
        .agg(deg_mean="mean", deg_std=lambda x: x.std(ddof=0))
        .reset_index()
    )

    print("\n── Per-Amino-Acid-Type Degree Statistics (reference set) ────")
    print(stats.to_string(index=False))
    print(f"\nTotal residues in reference set: {len(ref_df):,}")
    print("─────────────────────────────────────────────────────────────\n")

    return stats


def standardize_degree_by_aa(df: pd.DataFrame, aa_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Amitai method for degree: z-score each residue against its amino acid
    type's mean and std from the reference population.
    Adds StdDegree column.
    """
    df = df.copy()
    df = df.merge(aa_stats, on="Residue", how="left")

    missing = df["deg_mean"].isna().sum()
    if missing > 0:
        unknown = df.loc[df["deg_mean"].isna(), "Residue"].unique()
        print(f"Warning: {missing} residues have no degree stats "
              f"(unexpected non-canonical AAs): {unknown}.")

    df["StdDegree"] = (df["Degree"] - df["deg_mean"]) / df["deg_std"]
    df.drop(columns=["deg_mean", "deg_std"], inplace=True)
    return df


def standardize_betweenness_per_protein(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score BetweennessCentrality within each protein individually,
    following the same approach used for closeness (Amitai et al. Methods).
 
    Adds a StdBetweenness column.
    Proteins with std == 0 (degenerate case) receive StdBetweenness = 0.
 
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'BetweennessCentrality' and 'pdb_id' columns.
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
 


def assign_rsa_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSA_category column using RSA-empirical values.
    Categories: 'exposed' (RSA > 0.5), 'core' (RSA == 0.0), 'other', 'unknown'
    """
    df = df.copy()
    if "RSA-empirical" not in df.columns:
        df["RSA_category"] = "unknown"
        return df

    conditions = [df["RSA-empirical"] > 0.5, df["RSA-empirical"] == 0.0]
    choices    = ["exposed", "core"]
    df["RSA_category"] = np.select(conditions, choices, default="other")
    return df


# ── plotting ──────────────────────────────────────────────────────────────────

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

AXIS_LIM = (-4, 4)
AXIS_LIM_2 = (-4,10)

# one-letter to three-letter amino acid code mapping for BioLiP token parsing
ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def load_biolip_functional_residues(biolip_csv: Path) -> pd.DataFrame:
    """
    Parse the BioLiP CSV and return a DataFrame of functional residue join keys.

    BioLiP functional_residues format: space-separated tokens like "C95 D49 E35"
    where the first character is the one-letter AA code and the remainder is the
    residue number. Both are parsed and stored; the join uses pdb_id + chain +
    residue number only (the AA code is kept for validation/debugging).

    Returns a DataFrame with columns:
        pdb_id (uppercase), chain, residue_num (int), aa_one (str), aa_three (str)
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
                "pdb_id":    pdb_id,
                "chain":     chain,
                "residue_num": res_num,
                "aa_one":    aa_one,
                "aa_three":  aa_three,
            })

    return pd.DataFrame(records)

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

def mark_functional_residues(test_df: pd.DataFrame,
                             biolip_csv: Path) -> pd.DataFrame:
    """
    Add a boolean column 'is_functional' to test_df by joining against BioLiP.

    Join key: pdb_id (uppercase) + Chain + Residue number
    """
    func_df = load_biolip_functional_residues(biolip_csv)

    # normalise join key types in test_df
    test_df = test_df.copy()
    test_df["pdb_id_upper"]   = test_df["pdb_id"].str.upper()
    test_df["Residue number"] = test_df["Residue number"].astype(int)

    # build a set of (pdb_id, chain, residue_num) tuples for fast lookup
    func_set = set(
        zip(func_df["pdb_id"], func_df["chain"], func_df["residue_num"])
    )

    # ### error check
    test_pdb_ids   = set(test_df["pdb_id_upper"].unique())
    biolip_pdb_ids = set(func_df["pdb_id"].unique())
    overlap        = test_pdb_ids & biolip_pdb_ids
    print(f"BioLiP total functional residue tokens : {len(func_df):,}")
    print(f"BioLiP unique proteins                 : {len(biolip_pdb_ids):,}")
    print(f"Test-set unique proteins               : {len(test_pdb_ids):,}")
    print(f"Proteins in both                       : {len(overlap):,}")
    func_df_overlap = func_df[func_df["pdb_id"].isin(overlap)]
    print(f"BioLiP tokens for overlapping proteins : {len(func_df_overlap):,}")
    print(f"Unique (pdb_id, chain, residue_num) in overlap : {func_df_overlap.drop_duplicates(['pdb_id','chain','residue_num']).shape[0]:,}")

    chains_per_protein = (
        test_df.groupby("pdb_id_upper")["Chain"]
        .nunique()
        .reset_index(name="n_chains")
    )
    multi_chain = chains_per_protein[chains_per_protein["n_chains"] > 1]
    print(f"Proteins with >1 chain in test_df: {len(multi_chain)}")
    print(multi_chain.head(10))
    # ─────────────────────────────────────────────────────────────────────────

    test_df["is_functional"] = list(zip(
        test_df["pdb_id_upper"],
        test_df["Chain"],
        test_df["Residue number"],
    ))
    test_df["is_functional"] = test_df["is_functional"].isin(func_set)
    # ── diagnostic: find over-matched proteins ────────────────────────────
    func_matches = test_df[test_df["is_functional"]].copy()
    matches_per_protein = (
        func_matches.groupby(["pdb_id_upper", "Chain"])
        .size()
        .reset_index(name="n_matched")
    )
    biolip_per_protein = (
        func_df.groupby(["pdb_id", "chain"])
        .size()
        .reset_index(name="n_biolip")
    )
    biolip_per_protein = biolip_per_protein.rename(
        columns={"pdb_id": "pdb_id_upper", "chain": "Chain"}
    )
    comparison = matches_per_protein.merge(
        biolip_per_protein, on=["pdb_id_upper", "Chain"], how="left"
    )
    comparison["ratio"] = comparison["n_matched"] / comparison["n_biolip"]
    over_matched = comparison[comparison["n_matched"] > comparison["n_biolip"]]
    print(f"Proteins where matched > BioLiP tokens: {len(over_matched)}")
    print(over_matched.sort_values("ratio", ascending=False).head(20).to_string(index=False))
    # ─────────────────────────────────────────────────────────────────────
    test_df.drop(columns=["pdb_id_upper"], inplace=True)

    n_func = test_df["is_functional"].sum()
    n_total = len(test_df)
    print(f"Functional residues matched: {n_func} / {n_total} "
          f"({100 * n_func / n_total:.1f}%)")

    return test_df


def _scatter_by_rsa(ax, df, color_map, alpha=0.4, s=6, zorder=2, edgecolors="white"):
    for cat in RSA_ORDER:
        sub = df[df["RSA_category"] == cat]
        if sub.empty:
            continue
        ax.scatter(
            sub["StdCloseness"], sub["StdDegree"],
            c=color_map[cat], alpha=alpha, s=s,
            zorder=zorder, edgecolors=edgecolors, label=None,
        )


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

def _format_axes_no_title_2(ax,
                 xlabel="Standardized Closeness Centrality",
                 ylabel="Standardized Degree Centrality"):
    ax.set_xlim(AXIS_LIM_2)
    ax.set_ylim(AXIS_LIM_2)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel(xlabel, fontsize=FONT["size_label"])
    ax.set_ylabel(ylabel, fontsize=FONT["size_label"])


def plot_reference_only(ref_df: pd.DataFrame, output_path: Path):
    apply_style()    
    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])
    _scatter_by_rsa(ax, ref_df, RSA_COLORS, alpha=0.35, s=5)

    patches = [
        mpatches.Patch(color=RSA_COLORS[cat], label=RSA_LABELS[cat])
        for cat in RSA_ORDER
        if not ref_df[ref_df["RSA_category"] == cat].empty
    ]
    ax.legend(handles=patches, fontsize=FONT["size_legend"], loc="upper left")
    # _format_axes(
    #     ax,
    #     f"Reference Set: Standardized Closeness vs. Degree\n"
    #     f"(n = {len(ref_df):,} residues, {ref_df['pdb_id'].nunique()} proteins)\n"
    #     "Closeness: per-protein z-score  |  Degree: per-AA-type z-score (ref population)",
    # )
    _format_axes_no_title(ax)
    fig.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_test_only(test_df: pd.DataFrame, output_path: Path):
    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])
    _scatter_by_rsa(ax, test_df, OVERLAY_COLORS, alpha=0.70, s=18, zorder=2,
                    edgecolors="white")

    patches = [
        mpatches.Patch(color=OVERLAY_COLORS[cat], label=RSA_LABELS[cat])
        for cat in RSA_ORDER
        if not test_df[test_df["RSA_category"] == cat].empty
    ]
    ax.legend(handles=patches, fontsize=FONT["size_legend"], loc="upper left")

    ### if you want a title or not
    # _format_axes(
    #     ax,
    #     f"Small Enzyme Test Set: Standardized Closeness vs. Degree\n"
    #     f"(n = {len(test_df):,} residues, {test_df['pdb_id'].nunique()} proteins)\n"
    #     "Closeness: per-protein z-score  |  Degree: per-AA-type z-score (ref population)",
    # )
    _format_axes_no_title(ax)
    fig.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_test_only_functional(test_df: pd.DataFrame, output_path: Path):
    """
    Scatter of the test set with functional residues (from BioLiP) highlighted
    in black and all other residues in light grey.
    Requires 'is_functional' column to be present (added by mark_functional_residues).
    """
    # ── diagnostic ────────────────────────────────────────────────────────
    print(f"plot_test_only_functional received: {len(test_df):,} rows")
    print(f"  Unique chains: {test_df.groupby(['pdb_id', 'Chain']).ngroups}")
    print(f"  is_functional True count: {test_df['is_functional'].sum():,}")
    # ─────────────────────────────────────────────────────────────────────

    if "is_functional" not in test_df.columns:
        raise ValueError(
            "'is_functional' column not found. "
            "Run mark_functional_residues() before calling this function."
        )
    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])
    # non-functional residues — light grey background
    non_func = test_df[~test_df["is_functional"]]
    ax.scatter(
        non_func["StdCloseness"], non_func["StdDegree"],
        c="#CCCCCC", alpha=0.5, s=12, zorder=2,
        edgecolors="white", label=f"Non-functional (n={len(non_func):,})",
    )

    # functional residues — black on top
    func = test_df[test_df["is_functional"]]
    ax.scatter(
        func["StdCloseness"], func["StdDegree"],
        c="black", alpha=0.9, s=30, zorder=3,
        edgecolors="white", linewidths=0.5,
        label=f"Functional residues (n={len(func):,})",
    )

    ax.legend(fontsize=FONT["size_legend"], loc="upper left")
    # _format_axes(
    #     ax,
    #     f"Small Enzyme Test Set: Functional Residues Highlighted\n"
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


def plot_overlay(ref_df: pd.DataFrame, test_df: pd.DataFrame, output_path: Path):
    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])
    # reference set — muted background
    _scatter_by_rsa(ax, ref_df,  RSA_COLORS,    alpha=0.20, s=4,  zorder=2)
    # test set — prominent foreground
    _scatter_by_rsa(ax, test_df, OVERLAY_COLORS, alpha=0.70, s=18, zorder=3,
                    edgecolors="white")

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
    ax.legend(handles=ref_patches + test_patches, fontsize=FONT["size_legend"],
              loc="upper left",
              title="Reference set  /  Small enzyme test set",
              title_fontsize=FONT["size_legend"])

    # _format_axes(
    #     ax,
    #     f"Standardized Closeness vs. Degree\n"
    #     f"Reference (n = {len(ref_df):,}) + "
    #     f"Small enzyme test set (n = {len(test_df):,})\n"
    #     "Closeness: per-protein z-score  |  Degree: per-AA-type z-score (ref population)",
    # )
    _format_axes_no_title(ax)
    fig.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")

def _scatter_by_rsa_betweenness_vs_closeness(ax, df, color_map,
                                              alpha=0.4, s=6,
                                              zorder=2, edgecolors="white"):
    """
    Scatter helper: StdCloseness (x) vs. StdBetweenness (y), colored by RSA category.
    Mirrors _scatter_by_rsa() but uses StdBetweenness on the y-axis.
    """
    for cat in RSA_ORDER:
        sub = df[df["RSA_category"] == cat]
        if sub.empty:
            continue
        ax.scatter(
            sub["StdCloseness"], sub["StdBetweenness"],
            c=color_map[cat], alpha=alpha, s=s,
            zorder=zorder, edgecolors=edgecolors, label=None,
        )
 
 
def _scatter_by_rsa_betweenness_vs_degree(ax, df, color_map,
                                           alpha=0.4, s=6,
                                           zorder=2, edgecolors="white"):
    """
    Scatter helper: StdDegree (x) vs. StdBetweenness (y), colored by RSA category.
    Mirrors _scatter_by_rsa() but uses StdDegree on x and StdBetweenness on y.
    """
    for cat in RSA_ORDER:
        sub = df[df["RSA_category"] == cat]
        if sub.empty:
            continue
        ax.scatter(
            sub["StdDegree"], sub["StdBetweenness"],
            c=color_map[cat], alpha=alpha, s=s,
            zorder=zorder, edgecolors=edgecolors, label=None,
        )
 
 
# ── test-only plots ───────────────────────────────────────────────────────────
 
def plot_test_only_betweenness_vs_closeness(test_df: pd.DataFrame,
                                            output_path: Path,
                                            x_lim: tuple = AXIS_LIM,
                                            y_lim: tuple = (-4, 10)):
    """
    Scatter of the test set: StdCloseness (x) vs. StdBetweenness (y),
    colored by RSA category. Mirrors plot_test_only().

    Parameters
    ----------
    x_lim : tuple   X-axis (StdCloseness) limits. Default (-4, 4).
    y_lim : tuple   Y-axis (StdBetweenness) limits. Default (-4, 10).
    """
    if "StdBetweenness" not in test_df.columns:
        raise ValueError(
            "'StdBetweenness' column not found. "
            "Run standardize_betweenness_per_protein() before calling this function."
        )

    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])

    _scatter_by_rsa_betweenness_vs_closeness(
        ax, test_df, OVERLAY_COLORS, alpha=0.70, s=18, zorder=2,
        edgecolors="white",
    )

    patches = [
        mpatches.Patch(color=OVERLAY_COLORS[cat], label=RSA_LABELS[cat])
        for cat in RSA_ORDER
        if not test_df[test_df["RSA_category"] == cat].empty
    ]
    ax.legend(handles=patches, fontsize=FONT["size_legend"], loc="upper left")

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
 
 
def plot_test_only_betweenness_vs_degree(test_df: pd.DataFrame,
                                         output_path: Path,
                                         x_lim: tuple = AXIS_LIM,
                                         y_lim: tuple = (-4, 10)):
    """
    Scatter of the test set: StdDegree (x) vs. StdBetweenness (y),
    colored by RSA category. Mirrors plot_test_only().

    Parameters
    ----------
    x_lim : tuple   X-axis (StdDegree) limits. Default (-4, 4).
    y_lim : tuple   Y-axis (StdBetweenness) limits. Default (-4, 10).
    """
    if "StdBetweenness" not in test_df.columns:
        raise ValueError(
            "'StdBetweenness' column not found. "
            "Run standardize_betweenness_per_protein() before calling this function."
        )

    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])

    _scatter_by_rsa_betweenness_vs_degree(
        ax, test_df, OVERLAY_COLORS, alpha=0.70, s=18, zorder=2,
        edgecolors="white",
    )

    patches = [
        mpatches.Patch(color=OVERLAY_COLORS[cat], label=RSA_LABELS[cat])
        for cat in RSA_ORDER
        if not test_df[test_df["RSA_category"] == cat].empty
    ]
    ax.legend(handles=patches, fontsize=FONT["size_legend"], loc="upper left")

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
 
 
# ── overlay plots ─────────────────────────────────────────────────────────────
 
def plot_overlay_betweenness_vs_closeness(ref_df: pd.DataFrame,
                                          test_df: pd.DataFrame,
                                          output_path: Path,
                                          x_lim: tuple = AXIS_LIM,
                                          y_lim: tuple = (-4, 10)):
    """
    Overlay scatter: StdCloseness (x) vs. StdBetweenness (y).
    Reference set as muted background, test set as prominent foreground.
    Mirrors plot_overlay().

    Parameters
    ----------
    x_lim : tuple   X-axis (StdCloseness) limits. Default (-4, 4).
    y_lim : tuple   Y-axis (StdBetweenness) limits. Default (-4, 10).
    """
    if "StdBetweenness" not in ref_df.columns or "StdBetweenness" not in test_df.columns:
        raise ValueError(
            "'StdBetweenness' column not found in ref_df or test_df. "
            "Run standardize_betweenness_per_protein() on both DataFrames first."
        )

    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])

    # reference set — muted background
    _scatter_by_rsa_betweenness_vs_closeness(
        ax, ref_df, RSA_COLORS, alpha=0.20, s=4, zorder=2,
    )
    # test set — prominent foreground
    _scatter_by_rsa_betweenness_vs_closeness(
        ax, test_df, OVERLAY_COLORS, alpha=0.70, s=18, zorder=3,
        edgecolors="white",
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
        loc="upper left",
        title="Reference set  /  Small enzyme test set",
        title_fontsize=FONT["size_legend"],
    )

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
 
 
def plot_overlay_betweenness_vs_degree(ref_df: pd.DataFrame,
                                       test_df: pd.DataFrame,
                                       output_path: Path,
                                       x_lim: tuple = AXIS_LIM,
                                       y_lim: tuple = (-4, 10)):
    """
    Overlay scatter: StdDegree (x) vs. StdBetweenness (y).
    Reference set as muted background, test set as prominent foreground.
    Mirrors plot_overlay().

    Parameters
    ----------
    x_lim : tuple   X-axis (StdDegree) limits. Default (-4, 4).
    y_lim : tuple   Y-axis (StdBetweenness) limits. Default (-4, 10).
    """
    if "StdBetweenness" not in ref_df.columns or "StdBetweenness" not in test_df.columns:
        raise ValueError(
            "'StdBetweenness' column not found in ref_df or test_df. "
            "Run standardize_betweenness_per_protein() on both DataFrames first."
        )

    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])

    # reference set — muted background
    _scatter_by_rsa_betweenness_vs_degree(
        ax, ref_df, RSA_COLORS, alpha=0.20, s=4, zorder=2,
    )
    # test set — prominent foreground
    _scatter_by_rsa_betweenness_vs_degree(
        ax, test_df, OVERLAY_COLORS, alpha=0.70, s=18, zorder=3,
        edgecolors="white",
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
        loc="upper left",
        title="Reference set  /  Small enzyme test set",
        title_fontsize=FONT["size_legend"],
    )

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

# ── frequency distribution plots ──────────────────────────────────────────────

# Shared axis limits and bin edges for both distribution plots
_DIST_X_MIN   = -4.0
_DIST_X_MAX   =  4.0
_DIST_X_STEP  =  0.5
_DIST_Y_STEP  =  0.1

# RSA colour map for the distribution plots
# orange = exposed, green = core, blue = other
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


def plot_stdcloseness_frequency(df: pd.DataFrame,
                                output_path: Path,
                                x_min: float = _DIST_X_MIN,
                                x_max: float = _DIST_X_MAX,
                                x_step: float = _DIST_X_STEP,
                                y_min: float = 0.0,
                                y_max: float = None,
                                y_step: float = _DIST_Y_STEP):
    """
    Line plot of percentage-frequency vs. standardized closeness (StdCloseness),
    with one line per RSA category (exposed = orange, core = green, other = blue).

    Parameters
    ----------
    df                    : pd.DataFrame  Must contain 'StdCloseness' and 'RSA_category'.
    output_path           : Path          Destination file (PNG / PDF).
    x_min, x_max, x_step : float         X-axis range and bin width.
    y_min                 : float         Y-axis lower bound (default 0.0).
    y_max                 : float         Y-axis upper bound (default: auto from data).
    y_step                : float         Y-axis tick interval (as % frequency).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["half_page"])

    ax.set_facecolor("white")
    ax.grid(False)

    for cat in _DIST_RSA_ORDER:
        subset = df[df["RSA_category"] == cat]["StdCloseness"]
        if subset.empty:
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
                df[df["RSA_category"] == cat]["StdCloseness"], x_min, x_max, x_step
            )[1].max()
            for cat in _DIST_RSA_ORDER
            if not df[df["RSA_category"] == cat].empty
        )
        y_max = (int(y_max_data / y_step) + 2) * y_step
    y_ticks = np.arange(y_min, y_max + y_step / 2, y_step)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)

    ax.set_xlabel("Standardized Closeness Centrality", fontsize=FONT["size_label"])
    ax.set_ylabel("Frequency (%)",                     fontsize=FONT["size_label"])
    ax.legend(fontsize=FONT["size_legend"], loc="upper left")

    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_stddegree_frequency(df: pd.DataFrame,
                             output_path: Path,
                             x_min: float = _DIST_X_MIN,
                             x_max: float = _DIST_X_MAX,
                             x_step: float = _DIST_X_STEP,
                             y_min: float = 0.0,
                             y_max: float = None,
                             y_step: float = _DIST_Y_STEP):
    """
    Line plot of percentage-frequency vs. standardized degree (StdDegree),
    with one line per RSA category (exposed = orange, core = green, other = blue).

    Parameters
    ----------
    df                    : pd.DataFrame  Must contain 'StdDegree' and 'RSA_category'.
    output_path           : Path          Destination file (PNG / PDF).
    x_min, x_max, x_step : float         X-axis range and bin width.
    y_min                 : float         Y-axis lower bound (default 0.0).
    y_max                 : float         Y-axis upper bound (default: auto from data).
    y_step                : float         Y-axis tick interval (as % frequency).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["half_page"])

    ax.set_facecolor("white")
    ax.grid(False)

    for cat in _DIST_RSA_ORDER:
        subset = df[df["RSA_category"] == cat]["StdDegree"]
        if subset.empty:
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
                df[df["RSA_category"] == cat]["StdDegree"], x_min, x_max, x_step
            )[1].max()
            for cat in _DIST_RSA_ORDER
            if not df[df["RSA_category"] == cat].empty
        )
        y_max = (int(y_max_data / y_step) + 2) * y_step
    y_ticks = np.arange(y_min, y_max + y_step / 2, y_step)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)

    ax.set_xlabel("Standardized Degree Centrality", fontsize=FONT["size_label"])
    ax.set_ylabel("Frequency (%)",                  fontsize=FONT["size_label"])
    ax.legend(fontsize=FONT["size_legend"], loc="upper left")

    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_stdbetweenness_frequency(df: pd.DataFrame,
                                  output_path: Path,
                                  x_min: float = _DIST_X_MIN,
                                  x_max: float = _DIST_X_MAX,
                                  x_step: float = _DIST_X_STEP,
                                  y_min: float = 0.0,
                                  y_max: float = None,
                                  y_step: float = _DIST_Y_STEP):
    """
    Line plot of percentage-frequency vs. standardized betweenness (StdBetweenness),
    with one line per RSA category (exposed = orange, core = green, other = blue).
    Styled to match plot_stdcloseness_frequency() and plot_stddegree_frequency().
 
    Parameters
    ----------
    df                    : pd.DataFrame  Must contain 'StdBetweenness' and 'RSA_category'.
    output_path           : Path          Destination file (PNG / TIFF / PDF).
    x_min, x_max, x_step : float         X-axis range and bin width.
    y_min                 : float         Y-axis lower bound (default 0.0).
    y_max                 : float         Y-axis upper bound (default: auto from data).
    y_step                : float         Y-axis tick interval (as % frequency).
    """
    if "StdBetweenness" not in df.columns:
        raise ValueError(
            "'StdBetweenness' column not found. "
            "Run standardize_betweenness_per_protein() before calling this function."
        )
    if "RSA_category" not in df.columns:
        raise ValueError(
            "'RSA_category' column not found. "
            "Run assign_rsa_category() before calling this function."
        )
 
    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["half_page"])
 
    ax.set_facecolor("white")
    ax.grid(False)
 
    for cat in _DIST_RSA_ORDER:
        subset = df[df["RSA_category"] == cat]["StdBetweenness"]
        if subset.empty:
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
                df[df["RSA_category"] == cat]["StdBetweenness"], x_min, x_max, x_step
            )[1].max()
            for cat in _DIST_RSA_ORDER
            if not df[df["RSA_category"] == cat]["StdBetweenness"].dropna().empty
        )
        y_max = (int(y_max_data / y_step) + 2) * y_step
    y_ticks = np.arange(y_min, y_max + y_step / 2, y_step)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)
 
    ax.set_xlabel("Standardized Betweenness Centrality", fontsize=FONT["size_label"])
    ax.set_ylabel("Frequency (%)",                       fontsize=FONT["size_label"])
    ax.legend(fontsize=FONT["size_legend"], loc="upper right")
 
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")

# ── main ──────────────────────────────────────────────────────────────────────
# this run doesn't control for the size of proteins

# def main(
#     ref_centrality_dir,
#     ref_sasa_dir,
#     test_centrality_dir,
#     test_sasa_dir,
#     output_dir,
#     biolip_csv=None,
# ):
#     ref_centrality_dir  = Path(ref_centrality_dir)
#     ref_sasa_dir        = Path(ref_sasa_dir)  if ref_sasa_dir  else None
#     test_centrality_dir = Path(test_centrality_dir)
#     test_sasa_dir       = Path(test_sasa_dir) if test_sasa_dir else None
#     output_dir          = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # ── 1. load reference set ─────────────────────────────────────────────
#     print("Loading reference (old PISCES) centrality CSVs...")
#     ref_df = load_centrality_dir(ref_centrality_dir, ref_sasa_dir)
#     print(f"  Loaded {len(ref_df):,} residues from {ref_df['pdb_id'].nunique()} proteins.")

#     # ── 2. compute per-AA-type degree stats from reference set ────────────
#     aa_stats = compute_degree_stats_by_aa(ref_df)
#     aa_stats.to_csv(output_dir / "degree_stats_by_aa_type.csv", index=False)

#     # ── 3. standardize reference set ─────────────────────────────────────
#     ref_df = standardize_closeness_per_protein(ref_df)
#     ref_df = standardize_degree_by_aa(ref_df, aa_stats)
#     ref_df = assign_rsa_category(ref_df)

#     # ── 4. load small enzyme test set ────────────────────────────────────
#     print("Loading small enzyme test set centrality CSVs...")
#     test_df = load_centrality_dir(test_centrality_dir, test_sasa_dir)
#     print(f"  Loaded {len(test_df):,} residues from {test_df['pdb_id'].nunique()} proteins.")

#     # ── 5. standardize test set ───────────────────────────────────────────
#     # closeness: per-protein z-score (each test protein uses its own mean/std)
#     # degree: per-AA-type z-score using REFERENCE population stats
#     test_df = standardize_closeness_per_protein(test_df)
#     test_df = standardize_degree_by_aa(test_df, aa_stats)
#     test_df = assign_rsa_category(test_df)

#     # diagnostic prints
#     print(test_df.columns.tolist())
#     print(test_df.shape)
#     print(test_df["pdb_id"].nunique())
#     print(test_df["ClosenessCentrality"].isna().sum())


#     # ── 6. save standardized CSVs ─────────────────────────────────────────
#     ref_df.to_csv(output_dir  / "ref_standardized_ver4.csv",  index=False)
#     test_df.to_csv(output_dir / "test_standardized_ver4.csv", index=False)
#     print("Saved standardized CSVs.")

#     # ── 7. figures ────────────────────────────────────────────────────────
#     plot_reference_only(ref_df, output_dir / "fig1_reference_only_title.tiff")
#     plot_test_only(test_df,      output_dir / "fig1_test_only_title.tiff")
#     plot_overlay(ref_df, test_df, output_dir / "fig1_overlay_title.tiff")

#     #### 5/12/2026
#     # ── 8a. StdCloseness frequency distribution (reference set) ──────────
#     plot_stdcloseness_frequency(
#         ref_df,
#         output_dir / "fig_stdcloseness_frequency.tiff",
#         y_max=1.0,
#     )

#     # ── 8b. StdDegree frequency distribution (reference set) ─────────────
#     plot_stddegree_frequency(
#         ref_df,
#         output_dir / "fig_stddegree_frequency.tiff",
#         y_max=1.0,
#     )
#     ####
#     # ── 8. functional residue figure (requires BioLiP CSV) ────────────────
#     if biolip_csv is not None:
#         biolip_csv = Path(biolip_csv)
#         test_df = mark_functional_residues(test_df, biolip_csv)
#         plot_test_only_functional(test_df, output_dir / "fig1_test_functional.tiff")

#     print("\nDone.")

########################### 5/14/26 - exploring size control
# ── size-controlled helpers ────────────────────────────────────────────────────

def filter_ref_by_chain_length(df: pd.DataFrame, max_chain_length: int) -> pd.DataFrame:
    """
    Filter a reference DataFrame to retain only chains whose canonical-AA
    residue count is <= max_chain_length.

    The unit of length is the (pdb_id, Chain) pair — consistent with the
    Amitai et al. convention of treating each chain as the unit of analysis.

    Parameters
    ----------
    df               : pd.DataFrame  Output of load_centrality_dir(); must
                                     contain 'pdb_id' and 'Chain' columns.
    max_chain_length : int           Maximum number of canonical residues
                                     allowed per chain (inclusive).

    Returns
    -------
    pd.DataFrame with only the qualifying chains retained.
    """
    chain_lengths = (
        df.groupby(["pdb_id", "Chain"])
        .size()
        .reset_index(name="chain_length")
    )

    qualifying = chain_lengths[chain_lengths["chain_length"] <= max_chain_length][
        ["pdb_id", "Chain"]
    ]

    n_before = df["pdb_id"].nunique()
    filtered = df.merge(qualifying, on=["pdb_id", "Chain"], how="inner")
    n_after  = filtered["pdb_id"].nunique()

    print(
        f"  Chain-length filter (<= {max_chain_length} residues): "
        f"{n_before} -> {n_after} proteins "
        f"({len(qualifying)} chains retained out of {len(chain_lengths)} total)"
    )
    return filtered


def main_size_controlled(
    ref_centrality_dir,
    ref_sasa_dir,
    test_centrality_dir,
    test_sasa_dir,
    output_dir,
    max_chain_length=200,
    biolip_csv=None,
    exclude_test_pdb_ids=None,
    fasta_path=None
):
    """
    Identical pipeline to main(), but filters the reference set to chains
    with at most max_chain_length canonical residues before computing
    degree statistics and standardization.

    The test set is NOT filtered — only the reference population changes.
    """
    ref_centrality_dir  = Path(ref_centrality_dir)
    ref_sasa_dir        = Path(ref_sasa_dir)  if ref_sasa_dir  else None
    test_centrality_dir = Path(test_centrality_dir)
    test_sasa_dir       = Path(test_sasa_dir) if test_sasa_dir else None
    output_dir          = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. load reference set ─────────────────────────────────────────────
    print("Loading reference (old PISCES) centrality CSVs...")
    ref_df = load_centrality_dir(ref_centrality_dir, ref_sasa_dir)
    print(f"  Loaded {len(ref_df):,} residues from {ref_df['pdb_id'].nunique()} proteins.")

    # ── 2. apply chain-length filter to reference set ─────────────────────
    print(f"Applying chain-length filter (max {max_chain_length} residues)...")
    ref_df = filter_ref_by_chain_length(ref_df, max_chain_length)
    print(f"  After filter: {len(ref_df):,} residues from {ref_df['pdb_id'].nunique()} proteins.")

    # ── 3. compute per-AA-type degree stats from filtered reference set ───
    aa_stats = compute_degree_stats_by_aa(ref_df)
    aa_stats.to_csv(output_dir / "degree_stats_by_aa_type.csv", index=False)

    # ── 4. standardize reference set ─────────────────────────────────────────
    ref_df = standardize_closeness_per_protein(ref_df)
    ref_df = standardize_degree_by_aa(ref_df, aa_stats)
    ref_df = standardize_betweenness_per_protein(ref_df) 
    ref_df = assign_rsa_category(ref_df)

    # ── 5. load small enzyme test set ────────────────────────────────────
    print("Loading small enzyme test set centrality CSVs...")
    test_df = load_centrality_dir(test_centrality_dir, test_sasa_dir,
                                  exclude_pdb_ids=exclude_test_pdb_ids)
    print(f"  Loaded {len(test_df):,} residues from {test_df['pdb_id'].nunique()} proteins.")

    # ── 5b. filter test_df to FASTA chains only ───────────────────────────
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
        print("  WARNING: fasta_path is None — FASTA chain filter was NOT applied.")  # <-- add this

    # ── 6. standardize test set ───────────────────────────────────────────
    # closeness: per-protein z-score (each test protein uses its own mean/std)
    # degree: per-AA-type z-score using FILTERED reference population stats
    test_df = standardize_closeness_per_protein(test_df)
    test_df = standardize_degree_by_aa(test_df, aa_stats)
    test_df = standardize_betweenness_per_protein(test_df)
    test_df = assign_rsa_category(test_df)

    # diagnostic prints
    print(test_df.columns.tolist())
    print(test_df.shape)
    print(test_df["pdb_id"].nunique())
    print(test_df["ClosenessCentrality"].isna().sum())

    # ── 7. save standardized CSVs ─────────────────────────────────────────
    ref_df.to_csv(output_dir  / "ref_standardized.csv",  index=False)
    test_df.to_csv(output_dir / "test_standardized.csv", index=False)
    print("Saved standardized CSVs.")

    # ── 8. figures ────────────────────────────────────────────────────────
    #### change to tiff/png here 
    plot_reference_only(ref_df, output_dir / "fig1_reference_only_title.png")
    plot_test_only(test_df,      output_dir / "fig1_test_only_title.png")
    plot_overlay(ref_df, test_df, output_dir / "fig1_overlay_title.png")

    # ── 9a. StdCloseness frequency distribution (reference set) ──────────
    plot_stdcloseness_frequency(
        ref_df,
        output_dir / "fig_stdcloseness_frequency.png",
        y_max=1.0,
    )

    # ── 9b. StdDegree frequency distribution (reference set) ─────────────
    plot_stddegree_frequency(
        ref_df,
        output_dir / "fig_stddegree_frequency.png",
        y_max=1.0,
    )

    plot_stdbetweenness_frequency(
        ref_df,
        output_dir / "fig_stdbetweenness_frequency.png",
        y_max=1.0,
    )

    # ── 10. functional residue figure (requires BioLiP CSV) ───────────────
    if biolip_csv is not None:
        biolip_csv = Path(biolip_csv)
        test_df = mark_functional_residues(test_df, biolip_csv)
        plot_test_only_functional(test_df, output_dir / "fig1_test_functional.png")

    print("\nDone.")

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
    which reflects canonical-AA residue count after filtering in load_centrality_dir().

    Parameters
    ----------
    ref_csv               : str or Path  Path to ref_standardized_ver4.csv (or any
                                         standardized CSV with pdb_id + Chain columns).
    output_path           : str or Path  Destination PNG file.
    test_csv              : str or Path, optional  Path to test_standardized_ver4.csv.
                                         If provided, overlaid as a second histogram.
    max_chain_length_line : int, optional  If provided, draws a vertical dashed line
                                         at this value (e.g. 200 for the size filter).
    bin_width             : int          Width of histogram bins in residues. Default 10.
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
            label=f"Size filter ({max_chain_length_line} aa)",
        )

    ax.set_xlabel("Chain length (canonical residues)", fontsize=FONT["size_label"])
    ax.set_ylabel("Number of chains",                  fontsize=FONT["size_label"])
    ax.set_xlim(x_min, x_max)
    ax.legend(fontsize=FONT["size_legend"])

    fig.tight_layout(pad=LAYOUT["tight_pad"])
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"\nSaved: {output_path}")

## 5/20/26
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

def plot_rsa_boxplot_functional_categories(test_df: pd.DataFrame, output_path: Path):
    """
    Box plot of RSA-empirical distributions across the three RSA categories
    (exposed, core, other) for functional residues only.
    Requires 'is_functional', 'RSA_category', and 'RSA-empirical' columns.
    """
    apply_style()

    if "is_functional" not in test_df.columns:
        raise ValueError("'is_functional' column not found. Run mark_functional_residues() first.")
    if "RSA_category" not in test_df.columns:
        raise ValueError("'RSA_category' column not found. Run assign_rsa_category() first.")

    plot_df = test_df[test_df["is_functional"]][["RSA-empirical", "RSA_category"]].dropna()

    fig, ax = plt.subplots(figsize=FIG_SIZE["single_col"])

    sns.boxplot(
        data=plot_df,
        x="RSA_category",
        y="RSA-empirical",
        order=["exposed", "other", "core"],
        palette={
            "exposed": RSA_COLORS["exposed"],
            "other":   RSA_COLORS["other"],
            "core":    RSA_COLORS["core"],
        },
        width=0.5,
        linewidth=LAYOUT["spine_linewidth"],
        flierprops=dict(marker="o", markersize=2, alpha=0.3, markeredgewidth=0.5),
        ax=ax,
    )

    ax.set_xlabel("RSA category", fontsize=FONT["size_label"])
    ax.set_ylabel("RSA (empirical)", fontsize=FONT["size_label"])
    ax.set_title("RSA distribution of functional residues by category",
                 fontsize=FONT["size_title"])

    for i, cat in enumerate(["exposed", "other", "core"]):
        n = plot_df[plot_df["RSA_category"] == cat].shape[0]
        ax.text(i, ax.get_ylim()[1] * 0.97, f"n={n:,}",
                ha="center", va="top", fontsize=FONT["size_annot"])

    fig.tight_layout(pad=LAYOUT["tight_pad"])
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    # 5/20/26 - exclude proteins with unresolved EC numbers (prio_ec_number == '-1')
    # main(
    #     ref_centrality_dir  = DATA_DIR_2 / "old_PISCES" / "Centralities",
    #     ref_sasa_dir        = DATA_DIR_2 / "old_PISCES" / "SASAs",
    #     test_centrality_dir = DATA_DIR_2 / "Centralities",
    #     test_sasa_dir       = DATA_DIR_2 / "SASAs",
    #     output_dir          = DATA_DIR_2 / "figures" / "fig1_recreation_ver5",
    #     biolip_csv          = DATA_DIR_2 / "biolip_functional_residues.csv",
    # )
    # main_size_controlled(
    #     ref_centrality_dir   = DATA_DIR_2 / "old_PISCES" / "Centralities",
    #     ref_sasa_dir         = DATA_DIR_2 / "old_PISCES" / "SASAs",
    #     test_centrality_dir  = DATA_DIR_2 / "Centralities",
    #     test_sasa_dir        = DATA_DIR_2 / "SASAs",
    #     output_dir           = DATA_DIR_2 / "figures" / "fig1_recreation",
    #     max_chain_length     = 200,
    #     biolip_csv           = DATA_DIR_2 / "biolip_functional_residues.csv",
    #     exclude_test_pdb_ids = {'7ara', '4q95'}, ### these two IDs have issues
    #     fasta_path           = DATA_DIR_2 / "biolip_filtered.fasta"
    # )
    # ver_dir = DATA_DIR_2 / "figures" / "fig1_recreation"
    # plot_chain_length_distribution(
    #     ref_csv              = ver_dir / "ref_standardized.csv",
    #     test_csv             = ver_dir / "test_standardized.csv",
    #     output_path          = ver_dir / "fig_chain_length_distribution_both.png",
    #     max_chain_length_line= 200,
    #     bin_width            = 10,
    # )
    # plot_chain_length_distribution(
    #     ref_csv              = ver_dir / "ref_standardized.csv",
    #     test_csv             = None,
    #     output_path          = ver_dir / "fig_chain_length_distribution_ref.png",
    #     max_chain_length_line= 200,
    #     bin_width            = 10,
    # )

    ## 5/20/26
    # ver_dir  = DATA_DIR_2 / "figures" / "fig1_recreation"
    # ref_df   = pd.read_csv(ver_dir / "ref_standardized.csv")
    # test_df  = pd.read_csv(ver_dir / "test_standardized.csv")

    # for label, df in [("Reference set", ref_df), ("Test set", test_df)]:
    #     valid = df[["StdCloseness", "StdDegree"]].dropna()
    #     r, p  = stats.pearsonr(valid["StdCloseness"], valid["StdDegree"])
    #     print(f"\n── Centrality Correlation ({label}) ──────────────────")
    #     print(f"  n residues : {len(valid):,}")
    #     print(f"  Pearson r  : {r:.4f}")
    #     print(f"  p-value    : {p:.2e}")
    #     print("──────────────────────────────────────────────────────\n")
    
    # combined_df = pd.concat([ref_df, test_df], ignore_index=True)
    # valid = combined_df[["StdCloseness", "StdDegree"]].dropna()
    # r, p  = stats.pearsonr(valid["StdCloseness"], valid["StdDegree"])
    # print(f"  n residues : {len(valid):,}")
    # print(f"  Pearson r  : {r:.4f}")
    # print(f"  p-value    : {p:.2e}")
    # overlap = set(ref_df["pdb_id"]) & set(test_df["pdb_id"])
    # print(f"Overlapping PDB IDs: {len(overlap)}")

    # plot_reference_only(test_df, ver_dir / "fig1_test_rsa.png")

    ## 5/27/26 
    # Probably want to alter the legend to describe Buried, Partially Exposed, Exposed instead

    ## 6/1/26 - reformatted for PLOS comp. bio
    # main(
    #     ref_centrality_dir  = DATA_DIR_2 / "old_PISCES" / "Centralities",
    #     ref_sasa_dir        = DATA_DIR_2 / "old_PISCES" / "SASAs",
    #     test_centrality_dir = DATA_DIR_2 / "Centralities",
    #     test_sasa_dir       = DATA_DIR_2 / "SASAs",
    #     output_dir          = DATA_DIR_2 / "figures" / "fig1_recreation_ver0.6",
    #     biolip_csv          = DATA_DIR_2 / "biolip_functional_residues.csv",
    # )
    # main_size_controlled(
    #     ref_centrality_dir   = DATA_DIR_2 / "old_PISCES" / "Centralities",
    #     ref_sasa_dir         = DATA_DIR_2 / "old_PISCES" / "SASAs",
    #     test_centrality_dir  = DATA_DIR_2 / "Centralities",
    #     test_sasa_dir        = DATA_DIR_2 / "SASAs",
    #     output_dir           = DATA_DIR_2 / "figures" / "controlled_fig1_recreation_ver0.6",
    #     max_chain_length     = 200,
    #     biolip_csv           = DATA_DIR_2 / "biolip_functional_residues.csv",
    #     exclude_test_pdb_ids = {'7ara', '4q95'}, ### these two IDs have issues
    #     fasta_path           = DATA_DIR_2 / "biolip_filtered.fasta"
    # )
    # ver_dir = DATA_DIR_2 / "figures" / "controlled_fig1_recreation_ver0.6"
    # plot_chain_length_distribution(
    #     ref_csv              = ver_dir / "ref_standardized.csv",
    #     test_csv             = ver_dir / "test_standardized.csv",
    #     output_path          = ver_dir / "fig_chain_length_distribution_both.png",
    #     max_chain_length_line= 200,
    #     bin_width            = 10,
    # )
    # plot_chain_length_distribution(
    #     ref_csv              = ver_dir / "ref_standardized.csv",
    #     test_csv             = None,
    #     output_path          = ver_dir / "fig_chain_length_distribution_ref.png",
    #     max_chain_length_line= 200,
    #     bin_width            = 10,
    # )
    # test_df = mark_functional_residues(test_df, DATA_DIR_2 / "biolip_functional_residues.csv")
    # plot_rsa_boxplot_functional_categories(test_df, ver_dir / "fig_rsa_boxplot_functional_categories.tiff")

    ### 6/4/26 add betweenness centrality
    main_size_controlled(
        ref_centrality_dir   = DATA_DIR_2 / "old_PISCES" / "Centralities",
        ref_sasa_dir         = DATA_DIR_2 / "old_PISCES" / "SASAs",
        test_centrality_dir  = DATA_DIR_2 / "Centralities",
        test_sasa_dir        = DATA_DIR_2 / "SASAs",
        output_dir           = DATA_DIR_2 / "figures" / "controlled_fig1_recreation_ver0.7",
        max_chain_length     = 200,
        biolip_csv           = DATA_DIR_2 / "biolip_functional_residues.csv",
        exclude_test_pdb_ids = {'7ara', '4q95'}, ### these two IDs have issues
        fasta_path           = DATA_DIR_2 / "biolip_filtered.fasta"
    )

    ver_dir = DATA_DIR_2 / "figures" / "controlled_fig1_recreation_ver0.7"

    ref_df  = pd.read_csv(ver_dir / "ref_standardized.csv")
    test_df = pd.read_csv(ver_dir / "test_standardized.csv")

    ref_df  = standardize_betweenness_per_protein(ref_df)
    test_df = standardize_betweenness_per_protein(test_df)

    # distribution
    plot_stdbetweenness_frequency(
        ref_df,
        ver_dir / "fig_stdbetweenness_frequency.png",
        y_max=1.0,
    )

    # scatter
    plot_test_only_betweenness_vs_closeness(
        test_df, ver_dir / "fig_betweenness_vs_closeness_test.png", y_lim=(-4, 12)
    )
    plot_test_only_betweenness_vs_degree(
        test_df, ver_dir / "fig_betweenness_vs_degree_test.png", y_lim=(-4, 12)
    )
    plot_overlay_betweenness_vs_closeness(
        ref_df, test_df, ver_dir / "fig_betweenness_vs_closeness_overlay.png", y_lim=(-4, 12)
    )
    plot_overlay_betweenness_vs_degree(
        ref_df, test_df, ver_dir / "fig_betweenness_vs_degree_overlay.png", y_lim=(-4, 12)
    )

