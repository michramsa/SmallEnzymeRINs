# standardization_ver4.py
#
# Purpose: Alternative standardization scheme in which each centrality can be
# standardized in ONE of two ways, chosen per centrality:
#
#   "reference"   : against the reference set, matched on amino-acid type AND
#                   protein-size bin (exactly as in standardization_ver3.py).
#   "per_protein" : z-score within each protein (Amitai et al. 2004 style):
#                       Std = (raw - protein_mean) / protein_sd
#                   Reference statistics are not used at all, so ref and test
#                   proteins are treated identically and protein size is
#                   removed by construction.
#
# Default assignment (see SCHEMES in the __main__ block):
#   Degree                 -> reference   (local measure; depends on amino-acid
#                                          type, not on the rest of the protein)
#   ClosenessCentrality    -> per_protein (global measure)
#   BetweennessCentrality  -> per_protein (global measure)
#   PageRank_xN            -> per_protein (global measure)
#
# Changes from standardization_ver3.py:
#   - Per-centrality choice of scheme (SCHEMES). Setting every entry to
#     "reference" reproduces ver3.
#   - Per-protein z-scores use canonical residues only for the mean and SD
#     (population SD, ddof=0, as in ver3); non-canonical residues get NaN.
#   - A protein whose SD is 0 (or that has < MIN_PROTEIN_RESIDUES canonical
#     residues) gets NaN for that centrality (logged), not 0.
#   - Per-protein z-scores have mean 0 and SD 1 in every protein, so the ver3
#     "mean Std vs size" check is trivially flat for them. The after-check
#     therefore also reports the 90th percentile of Std per protein (the upper
#     tail, which is what thresholds act on). Mean is reported only for
#     "reference" columns.
#   - PageRank note: PageRank_xN is PageRank multiplied by one constant per
#     protein, so its per-protein z-score is identical to that of raw PageRank.
#   - Output column names are unchanged (StdDegree, StdCloseness, ...), so
#     downstream scripts work as before; a new standardization_scheme.csv
#     records which scheme produced each column.
#
# Method, for each dataset separately:
#   1. Size bins from the dataset's reference proteins (only used by
#      "reference" columns).
#   2. "reference" columns: per (amino acid, size bin) reference mean/SD;
#      cells with < MIN_CELL_RESIDUES residues fall back to the amino acid's
#      all-sizes statistics.
#   3. "per_protein" columns: mean/SD over each protein's own canonical
#      residues; applied to ref and test alike.
#
# Input : output folder of parallel_RIN_centrality_ver1.py
#         <CENTRALITY_ROOT>/<dataset>/<ref|test>/<PDBid>_centrality.csv
#         Required columns: PDBid, Residue, IsCanonical, n_nodes, plus the raw
#         columns named in COL_MAP.
#
# Outputs (under OUTPUT_ROOT):
#   <dataset>/<ref|test>/<PDBid>_std.csv       original columns + size_bin + Std*
#   <dataset>/reference_stats.csv              ("reference" columns only)
#   <dataset>/protein_stats.csv                per-protein mean/sd/n ("per_protein" columns)
#   <dataset>/size_bins.csv
#   <dataset>/standardization_scheme.csv       centrality -> scheme
#   size_check_after.csv                       size dependence of Std values (mean and/or p90)
#   figures/<dataset>/<StdColumn>_<stat>_vs_size.png   one figure per file
#
# All settings live in the `if __name__ == "__main__":` block at the bottom.

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

VALID_SCHEMES = ("reference", "per_protein")
STAT_LABELS = {"mean": "Mean", "p90": "90th percentile"}


# ── loading ───────────────────────────────────────────────────────────────────

def load_set(centrality_root, dataset, set_name) -> pd.DataFrame:
    """All residue rows for one (dataset, set) from parallel_RIN_centrality_ver1.py output."""
    folder = Path(centrality_root) / dataset / set_name
    paths = sorted(folder.glob("*_centrality.csv"))
    if not paths:
        raise FileNotFoundError(f"[{dataset}/{set_name}] no *_centrality.csv files in {folder}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["IsCanonical"] = df["IsCanonical"].astype(bool)
    print(f"  [{dataset}/{set_name}] {df['PDBid'].nunique()} proteins, {len(df):,} residues")
    return df


# ── size bins (used by "reference" columns) ──────────────────────────────────

def make_interior_edges(ref_df, n_bins, explicit_edges=None) -> np.ndarray:
    """Quantile edges of reference PROTEIN sizes; first and last bins are open-ended."""
    if explicit_edges is not None:
        return np.asarray(sorted(explicit_edges), dtype=float)
    sizes = ref_df.groupby("PDBid")["n_nodes"].first().to_numpy(float)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(sizes, qs))
    if len(edges) < n_bins - 1:
        print(f"  WARNING: only {len(edges) + 1} distinct size bins could be formed")
    return edges


def assign_bins(df, interior_edges) -> pd.DataFrame:
    df = df.copy()
    df["size_bin"] = np.digitize(df["n_nodes"].to_numpy(float), interior_edges, right=False) + 1
    return df


def bin_table(ref_df, test_df, interior_edges) -> pd.DataFrame:
    lower = np.concatenate([[-np.inf], interior_edges])
    upper = np.concatenate([interior_edges, [np.inf]])
    rows = []
    for b in range(1, len(lower) + 1):
        row = {"size_bin": b, "lower_edge_incl": lower[b - 1], "upper_edge_excl": upper[b - 1]}
        for name, df in (("ref", ref_df), ("test", test_df)):
            prot = df[df["size_bin"] == b].groupby("PDBid")["n_nodes"].first()
            row[f"n_{name}_proteins"] = len(prot)
            row[f"{name}_min_n"] = prot.min() if len(prot) else np.nan
            row[f"{name}_max_n"] = prot.max() if len(prot) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ── scheme 1: reference statistics per (amino acid, size bin) ────────────────

def reference_stats(ref_df, raw_cols, min_cell_residues):
    """
    Per (Residue, size_bin) mean / sd / count for each raw centrality, with
    small or zero-SD cells replaced by that amino acid's all-sizes statistics.

    Returns (stats_long, lookup); lookup: raw_col -> DataFrame indexed by
    (Residue, size_bin) with columns mean, sd.
    """
    ref = ref_df[ref_df["IsCanonical"]]
    all_bins = sorted(ref_df["size_bin"].unique())
    rows, lookup = [], {}

    for col in raw_cols:
        cell = (ref.groupby(["Residue", "size_bin"])[col]
                .agg(mean="mean", sd=lambda x: x.std(ddof=0), n_residues="count"))
        cell["n_proteins"] = ref.groupby(["Residue", "size_bin"])["PDBid"].nunique()
        allsize = ref.groupby("Residue")[col].agg(mean="mean", sd=lambda x: x.std(ddof=0))

        full_index = pd.MultiIndex.from_product([allsize.index, all_bins],
                                                names=["Residue", "size_bin"])
        cell = cell.reindex(full_index)
        cell["n_residues"] = cell["n_residues"].fillna(0).astype(int)
        cell["n_proteins"] = cell["n_proteins"].fillna(0).astype(int)

        bad = (cell["n_residues"] < min_cell_residues) | (cell["sd"].fillna(0) == 0)
        cell["fallback_to_all_sizes"] = bad
        aa = cell.index.get_level_values("Residue")
        cell.loc[bad, "mean"] = allsize.loc[aa[bad], "mean"].to_numpy()
        cell.loc[bad, "sd"] = allsize.loc[aa[bad], "sd"].to_numpy()
        if bad.any():
            cells = ", ".join(f"{r}/bin{b}" for r, b in cell.index[bad])
            print(f"    {col}: {bad.sum()} cell(s) used all-sizes stats ({cells})")

        lookup[col] = cell[["mean", "sd"]]
        out = cell.reset_index()
        out.insert(0, "centrality", col)
        rows.append(out)

    if not rows:
        return pd.DataFrame(), lookup
    return pd.concat(rows, ignore_index=True), lookup


def standardize_reference(df, lookup, col_map) -> pd.DataFrame:
    """Add Std columns using the (Residue, size_bin) reference statistics."""
    df = df.copy()
    key = pd.MultiIndex.from_arrays([df["Residue"], df["size_bin"]])
    for raw, std in col_map.items():
        ref = lookup[raw].reindex(key)
        mean, sd = ref["mean"].to_numpy(), ref["sd"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (df[raw].to_numpy(float) - mean) / sd
        z[~df["IsCanonical"].to_numpy() | ~np.isfinite(z)] = np.nan
        df[std] = z
    return df


# ── scheme 2: z-score within each protein ────────────────────────────────────

def standardize_per_protein(df, col_map, min_protein_residues, label=""):
    """
    Z-score each raw centrality within every protein (canonical residues only).

    Returns (df_with_std_columns, protein_stats_long).
    A protein with SD = 0 or fewer than `min_protein_residues` canonical
    residues gets NaN for that centrality.
    """
    df = df.copy()
    rows = []
    for raw, std in col_map.items():
        vals = df[raw].where(df["IsCanonical"])
        grp = vals.groupby(df["PDBid"])
        mean, sd, n = grp.mean(), grp.std(ddof=0), grp.count()

        usable = (n >= min_protein_residues) & (sd.fillna(0) > 0)
        if (~usable).any():
            print(f"    {label}{raw}: {(~usable).sum()} protein(s) not standardized "
                  f"(SD = 0 or < {min_protein_residues} canonical residues): "
                  f"{', '.join(map(str, usable.index[~usable][:10]))}"
                  f"{' ...' if (~usable).sum() > 10 else ''}")

        m = df["PDBid"].map(mean).to_numpy(float)
        s = df["PDBid"].map(sd.where(usable)).to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (df[raw].to_numpy(float) - m) / s
        z[~df["IsCanonical"].to_numpy() | ~np.isfinite(z)] = np.nan
        df[std] = z

        out = pd.DataFrame({"mean": mean, "sd": sd, "n_residues": n, "standardized": usable})
        out.insert(0, "centrality", raw)
        rows.append(out.reset_index())

    if not rows:
        return df, pd.DataFrame()
    return df, pd.concat(rows, ignore_index=True)


# ── writing ──────────────────────────────────────────────────────────────────

def write_per_protein(df, out_dir) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*_std.csv.tmp"):
        stale.unlink()
    n = 0
    for pdb_id, grp in df.groupby("PDBid", sort=False):
        tmp = out_dir / f"{pdb_id}_std.csv.tmp"
        grp.to_csv(tmp, index=False)
        tmp.replace(out_dir / f"{pdb_id}_std.csv")
        n += 1
    return n


# ── after-check: size dependence of standardized values ──────────────────────

def stats_for_scheme(scheme) -> list:
    """Per-protein z-scores have mean 0 by construction, so only p90 is informative."""
    return ["mean", "p90"] if scheme == "reference" else ["p90"]


def protein_level_table(df, std_cols) -> pd.DataFrame:
    """One row per protein: n_nodes plus <col>__mean and <col>__p90 of each Std column."""
    agg = {"n_nodes": ("n_nodes", "first")}
    for c in std_cols:
        agg[f"{c}__mean"] = (c, "mean")
        agg[f"{c}__p90"] = (c, lambda x: x.quantile(0.90))
    return df.groupby("PDBid").agg(**agg)


def size_check_after(ref_df, test_df, dataset, col_stats):
    """
    For each (Std column, statistic): per-protein value of the statistic vs
    protein size.
        spearman_rho / linear_slope : trend with size (want ~0)
        predicted_shift_sd = ref slope x (test median size - ref median size)
        observed_shift_sd  = mean over test proteins - mean over ref proteins
    Std values are already in SD units, so no further scaling is needed.
    """
    std_cols = list(col_stats)
    prot = {"ref": protein_level_table(ref_df, std_cols),
            "test": protein_level_table(test_df, std_cols)}
    n_ref_med = prot["ref"]["n_nodes"].median()
    n_test_med = prot["test"]["n_nodes"].median()

    rows = []
    for c, stat_list in col_stats.items():
        for stat in stat_list:
            col = f"{c}__{stat}"
            row = {"dataset": dataset, "centrality": c, "statistic": stat}
            for name in ("ref", "test"):
                p = prot[name].dropna(subset=[col])
                if len(p) >= 3 and p["n_nodes"].nunique() > 1:
                    row[f"{name}_spearman_rho"], row[f"{name}_spearman_p"] = \
                        stats.spearmanr(p["n_nodes"], p[col])
                    row[f"{name}_linear_slope_per_residue"] = np.polyfit(p["n_nodes"], p[col], 1)[0]
                else:
                    row[f"{name}_spearman_rho"] = row[f"{name}_spearman_p"] = np.nan
                    row[f"{name}_linear_slope_per_residue"] = np.nan
            row["predicted_shift_sd"] = row["ref_linear_slope_per_residue"] * (n_test_med - n_ref_med)
            row["observed_shift_sd"] = prot["test"][col].mean() - prot["ref"][col].mean()
            rows.append(row)
    return rows, prot


def plot_std_vs_size(prot, col, stat, label, scheme, dataset, interior_edges,
                     out_path, colors, dpi):
    key = f"{col}__{stat}"
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    names = {"ref": "Reference", "test": "Test"}
    notes = []
    for set_name in ("ref", "test"):
        p = prot[set_name].dropna(subset=[key])
        x, y = p["n_nodes"].to_numpy(float), p[key].to_numpy(float)
        ax.scatter(x, y, s=14, alpha=0.55, color=colors[set_name], edgecolor="none",
                   label=f"{names[set_name]} (n = {len(p)})", zorder=2)
        if len(p) >= 3 and np.unique(x).size > 1:
            slope, icpt = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, slope * xs + icpt, color=colors[set_name], linewidth=2, zorder=3)
            notes.append(f"{names[set_name]}: ρ = {stats.spearmanr(x, y)[0]:.2f}")

    ax.axhline(0, color="black", linewidth=0.8)
    if scheme == "reference":
        for e in interior_edges:
            ax.axvline(e, color="grey", linestyle=":", linewidth=1, zorder=1)
    ax.text(0.02, 0.98, "\n".join(notes), transform=ax.transAxes, va="top", ha="left",
            fontsize=8, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.set_xlabel("Protein size (number of nodes)")
    ax.set_ylabel(f"{STAT_LABELS[stat]} {label} per protein")
    ax.set_title(f"{dataset}: {label} ({scheme}) vs protein size", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    centrality_root,
    output_root,
    datasets: list,
    col_map: dict,
    schemes: dict,
    std_labels: dict,
    n_bins: int,
    explicit_edges,
    min_cell_residues: int,
    min_protein_residues: int,
    make_figures: bool,
    colors: dict,
    dpi: int,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    missing = [c for c in col_map if c not in schemes]
    bad = {c: s for c, s in schemes.items() if s not in VALID_SCHEMES}
    if missing or bad:
        raise ValueError(f"SCHEMES must map every raw column in COL_MAP to one of "
                         f"{VALID_SCHEMES}. Missing: {missing}. Invalid: {bad}")

    ref_map = {r: s for r, s in col_map.items() if schemes[r] == "reference"}
    prot_map = {r: s for r, s in col_map.items() if schemes[r] == "per_protein"}
    col_stats = {std: stats_for_scheme(schemes[raw]) for raw, std in col_map.items()}
    std_scheme = {std: schemes[raw] for raw, std in col_map.items()}
    print(f"Reference-standardized : {list(ref_map) or 'none'}")
    print(f"Per-protein standardized: {list(prot_map) or 'none'}")

    check_rows = []
    for dataset in datasets:
        print(f"\n══ {dataset} ══")
        ref_df = load_set(centrality_root, dataset, "ref")
        test_df = load_set(centrality_root, dataset, "test")
        ds_dir = output_root / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)

        # size bins (always assigned; only "reference" columns use them)
        edges = make_interior_edges(ref_df, n_bins, (explicit_edges or {}).get(dataset))
        ref_df, test_df = assign_bins(ref_df, edges), assign_bins(test_df, edges)
        bins = bin_table(ref_df, test_df, edges)
        bins.to_csv(ds_dir / "size_bins.csv", index=False)
        if ref_map:
            print("  Size bins (for reference-standardized columns):")
            print(bins[["size_bin", "lower_edge_incl", "upper_edge_excl",
                        "n_ref_proteins", "n_test_proteins"]].to_string(index=False))

        # scheme record
        pd.DataFrame({"raw_column": list(col_map), "std_column": list(col_map.values()),
                      "scheme": [schemes[r] for r in col_map]}
                     ).to_csv(ds_dir / "standardization_scheme.csv", index=False)

        # "reference" columns
        if ref_map:
            stats_long, lookup = reference_stats(ref_df, list(ref_map), min_cell_residues)
            stats_long.to_csv(ds_dir / "reference_stats.csv", index=False)
            print(f"  Smallest reference cell: {int(stats_long['n_residues'].min())} residues "
                  f"(median {int(stats_long['n_residues'].median())})")
            ref_std = standardize_reference(ref_df, lookup, ref_map)
            test_std = standardize_reference(test_df, lookup, ref_map)
        else:
            ref_std, test_std = ref_df, test_df

        # "per_protein" columns (ref and test alike, no reference statistics)
        if prot_map:
            ref_std, ps_ref = standardize_per_protein(ref_std, prot_map, min_protein_residues,
                                                      label="ref: ")
            test_std, ps_test = standardize_per_protein(test_std, prot_map, min_protein_residues,
                                                        label="test: ")
            ps_ref.insert(0, "set", "ref")
            ps_test.insert(0, "set", "test")
            pd.concat([ps_ref, ps_test], ignore_index=True).to_csv(
                ds_dir / "protein_stats.csv", index=False)

        for set_name, df in (("ref", ref_std), ("test", test_std)):
            n = write_per_protein(df, ds_dir / set_name)
            print(f"  Wrote {n} {set_name} files -> {ds_dir / set_name}")

        # after-check
        rows, prot = size_check_after(ref_std, test_std, dataset, col_stats)
        check_rows += rows
        if make_figures:
            fig_dir = output_root / "figures" / dataset
            fig_dir.mkdir(parents=True, exist_ok=True)
            for c, stat_list in col_stats.items():
                for stat in stat_list:
                    plot_std_vs_size(prot, c, stat, std_labels.get(c, c), std_scheme[c],
                                     dataset, edges, fig_dir / f"{c}_{stat}_vs_size.png",
                                     colors, dpi)

    check = pd.DataFrame(check_rows)
    check.to_csv(output_root / "size_check_after.csv", index=False)
    if not check.empty:
        for stat in ("mean", "p90"):
            sub = check[check["statistic"] == stat]
            if sub.empty:
                continue
            print(f"\n══ Statistic: {STAT_LABELS[stat]} of Std per protein ══")
            for title, value in (("predicted size-induced shift (SD units)", "predicted_shift_sd"),
                                 ("ref Spearman rho with size", "ref_spearman_rho"),
                                 ("observed test - ref (SD units)", "observed_shift_sd")):
                print(f"\n── {title} ──")
                print(sub.pivot(index="centrality", columns="dataset", values=value)
                      .to_string(float_format=lambda v: f"{v:+.3f}"))
    print(f"\nSaved to: {output_root}")


if __name__ == "__main__":

    # ── Edit these values ─────────────────────────────────────────────────────
    # Output folder of parallel_RIN_centrality_ver1.py (raw centralities)
    CENTRALITY_ROOT = Path("...0_4_revisions/2_experiments/4_centralities")
    # New folder for standardized output
    OUTPUT_ROOT = Path("...0_4_revisions/2_experiments/5_standardization/standardized_ver4")

    DATASETS = ["arp_iac", "arp_no_iac", "ring_iac", "ring_no_iac", "pcn5", "pcn6", "pcn7"]

    # Raw column -> standardized column (names unchanged from ver3).
    COL_MAP = {
        "Degree":                "StdDegree",
        "ClosenessCentrality":   "StdCloseness",
        "BetweennessCentrality": "StdBetweenness",
        "PageRank_xN":           "StdPageRank",
    }
    STD_LABELS = {
        "StdDegree": "StdDegree", "StdCloseness": "StdCloseness",
        "StdBetweenness": "StdBetweenness", "StdPageRank": "StdPageRank",
    }

    # Scheme per raw column: "reference" (amino acid x size bin, as in ver3) or
    # "per_protein" (z-score within each protein). All "reference" = ver3.
    SCHEMES = {
        "Degree":                "reference",     # local measure
        "ClosenessCentrality":   "per_protein",   # global measures
        "BetweennessCentrality": "per_protein",
        "PageRank_xN":           "per_protein",
    }

    # Settings for "reference" columns only
    N_BINS = 5                 # size bins from quantiles of reference protein sizes
    EXPLICIT_EDGES = None      # e.g. {"ring_iac": [90, 110, 140, 165]}; None = quantiles
    MIN_CELL_RESIDUES = 30     # smaller (amino acid, bin) cells use all-sizes stats

    # Setting for "per_protein" columns only
    MIN_PROTEIN_RESIDUES = 10  # proteins with fewer canonical residues get NaN

    MAKE_FIGURES = True
    COLORS = {"ref": "#4C72B0", "test": "#DD8452"}
    DPI = 300
    # ──────────────────────────────────────────────────────────────────────────

    main(
        centrality_root=CENTRALITY_ROOT,
        output_root=OUTPUT_ROOT,
        datasets=DATASETS,
        col_map=COL_MAP,
        schemes=SCHEMES,
        std_labels=STD_LABELS,
        n_bins=N_BINS,
        explicit_edges=EXPLICIT_EDGES,
        min_cell_residues=MIN_CELL_RESIDUES,
        min_protein_residues=MIN_PROTEIN_RESIDUES,
        make_figures=MAKE_FIGURES,
        colors=COLORS,
        dpi=DPI,
    )
