# std_distribution_diagnostics.py
#
# Purpose: Check whether the between-protein share (eta squared) reported by
# bootstrap_within_protein.py is driven by heavy tails or by a handful of
# proteins/residues, rather than by a general protein-level difference.
#
# Why: eta squared is built from squared deviations, so a few extreme values
# can make one centrality look strongly "protein-dependent". The existing
# scripts do not look at the shape of the Std distributions, and the
# frequency plots clip x to -4..4 (hiding the tails).
#
# Input : output of standardization_ver3.py
#         <STD_ROOT>/<dataset>/<ref|test>/<PDBid>_std.csv
#
# Residue filtering is identical to bootstrap_within_protein.py: proteins
# present in every dataset, canonical residues, no NaN in any Std column.
#
# What is computed, for each dataset, set (ref / test) and Std centrality:
#
#   1. Distribution shape (distribution_summary.csv)
#        mean, sd, median, skewness, excess kurtosis, min, max,
#        fraction of residues beyond +/- TAIL_THRESHOLD,
#        fraction of SS_total contributed by the top 1% most extreme residues.
#
#   2. Robustness of the between-protein share (share_robustness.csv)
#        share_full           : eta squared on all residues (same as
#                               bootstrap_within_protein.py point estimate)
#        share_trimmed        : after dropping the TRIM_FRAC of residues
#                               furthest from the mean
#        share_rank           : eta squared on pooled ranks (insensitive to
#                               outliers; equals Kruskal-Wallis H / (N - 1))
#        share_without_top<k> : after removing the k proteins that contribute
#                               most to SS_between (one column per k in
#                               LEAVE_OUT_K)
#      A share that stays about the same across these versions reflects a
#      general protein-level difference. A share that collapses when a few
#      residues or proteins are removed is an outlier effect.
#
#   3. Protein concentration of SS_between (protein_concentration.csv)
#        top1_fraction, top_fraction_of_proteins (share of SS_between from the
#        top TOP_PROTEIN_FRAC of proteins), effective number of proteins
#        (1 / sum of squared contribution fractions; P = perfectly even).
#      protein_top_contributors.csv lists the N_TOP_PROTEINS_SAVED largest
#      contributors per (dataset, set, centrality) so they can be inspected.
#
#   Point estimates only; no bootstrap is used in this script.
#
# Outputs (under OUTPUT_DIR):
#   distribution_summary.csv
#   share_robustness.csv
#   protein_concentration.csv
#   protein_top_contributors.csv
#   figures/hist/<set>/<dataset>.png                 unclipped log-density histograms
#   figures/robustness/<set>/<centrality>.png        share variants across datasets
#   figures/concentration/concentration_<set>.png    SS_between held by top proteins
#   (one figure per file)
#
# All settings live in the `if __name__ == "__main__":` block at the bottom.

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, rankdata, skew


# ── loading (same rules as bootstrap_within_protein.py) ──────────────────────

def load_set(std_root, dataset, set_name, std_cols) -> pd.DataFrame:
    folder = Path(std_root) / dataset / set_name
    paths = sorted(folder.glob("*_std.csv"))
    if not paths:
        raise FileNotFoundError(f"[{dataset}/{set_name}] no *_std.csv files in {folder}")
    df = pd.concat([pd.read_csv(p, usecols=["PDBid", "IsCanonical"] + std_cols) for p in paths],
                   ignore_index=True)
    df["PDBid"] = df["PDBid"].astype(str)
    return df


def common_proteins(frames: dict) -> list:
    sets = [set(df["PDBid"]) for df in frames.values()]
    common = set.intersection(*sets)
    for ds, s in zip(frames, sets):
        if len(s - common):
            print(f"    NOTE [{ds}]: {len(s - common)} protein(s) not in every dataset, excluded")
    return sorted(common)


def to_arrays(df, proteins, std_cols):
    pos = {p: i for i, p in enumerate(proteins)}
    df = df[df["PDBid"].isin(pos) & df["IsCanonical"].astype(bool)].dropna(subset=std_cols)
    return df[std_cols].to_numpy(float), df["PDBid"].map(pos).to_numpy(int)


# ── statistics ────────────────────────────────────────────────────────────────

def eta_squared(x, prot_idx, n_prot):
    """SS_between / SS_total for one variable; proteins with no residues are ignored."""
    N = len(x)
    if N < 2:
        return np.nan
    n = np.bincount(prot_idx, minlength=n_prot).astype(float)
    s1 = np.bincount(prot_idx, weights=x, minlength=n_prot)
    corr = x.sum() ** 2 / N
    sst = (x ** 2).sum() - corr
    ok = n > 0
    ssb = (s1[ok] ** 2 / n[ok]).sum() - corr
    return ssb / sst if sst > 0 else np.nan


def protein_contributions(x, prot_idx, n_prot):
    """
    Each protein's contribution to SS_between: n_i * (protein mean - grand mean)^2.
    Returns (protein indices, residue counts, protein means, contributions).
    """
    n = np.bincount(prot_idx, minlength=n_prot).astype(float)
    s1 = np.bincount(prot_idx, weights=x, minlength=n_prot)
    ok = np.flatnonzero(n > 0)
    mean_p = s1[ok] / n[ok]
    contrib = n[ok] * (mean_p - x.mean()) ** 2
    return ok, n[ok], mean_p, contrib


def trimmed_share(x, prot_idx, n_prot, trim_frac):
    """Eta squared after dropping the trim_frac of residues furthest from the mean."""
    dev = np.abs(x - x.mean())
    keep = dev <= np.quantile(dev, 1.0 - trim_frac)
    return eta_squared(x[keep], prot_idx[keep], n_prot)


def share_without_top_proteins(x, prot_idx, n_prot, k, order_ids):
    """Eta squared after removing the k proteins with the largest SS_between contribution."""
    keep = ~np.isin(prot_idx, order_ids[:k])
    return eta_squared(x[keep], prot_idx[keep], n_prot)


def distribution_row(x, tail_threshold):
    dev2 = (x - x.mean()) ** 2
    k = max(1, int(np.ceil(0.01 * len(x))))
    top = np.sort(dev2)[::-1][:k].sum()
    return {
        "n_residues": len(x),
        "mean": x.mean(),
        "sd": x.std(ddof=0),
        "median": np.median(x),
        "skewness": skew(x),
        "excess_kurtosis": kurtosis(x),
        "min": x.min(),
        "max": x.max(),
        f"frac_beyond_abs_{tail_threshold:g}": float((np.abs(x) > tail_threshold).mean()),
        "frac_sst_from_top1pct_residues": top / dev2.sum(),
    }


# ── plotting (one figure per file) ────────────────────────────────────────────

def side_legend(ax):
    """Legend outside the axes, to the right (never on top of the data)."""
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)


def plot_hist(X, std_cols, ds, set_name, tail_threshold, bin_width, out_path, colors, dpi):
    lo = np.floor(X.min() / bin_width) * bin_width
    hi = np.ceil(X.max() / bin_width) * bin_width + bin_width
    edges = np.arange(lo, hi + bin_width / 2, bin_width)
    centres = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for c, col in enumerate(std_cols):
        dens, _ = np.histogram(X[:, c], bins=edges, density=True)
        shown = dens > 0
        ax.plot(centres[shown], dens[shown], color=colors[col], linewidth=1.3, label=col)
    for v in (-tail_threshold, tail_threshold):
        ax.axvline(v, color="black", linestyle=":", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Std value (unclipped); dotted lines = tail threshold")
    ax.set_ylabel("Density (log scale)")
    ax.set_title(f"{ds} ({'reference' if set_name == 'ref' else 'test'}, n = {len(X):,} residues)",
                 fontsize=10)
    side_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_robustness(rob_df, set_name, col, variants, variant_labels, datasets, out_path,
                    variant_colors, variant_markers, dpi):
    s = rob_df[(rob_df["set_name"] == set_name) & (rob_df["centrality"] == col)
               ].set_index("dataset").reindex(datasets)
    x = np.arange(len(datasets))
    offs = np.linspace(-0.3, 0.3, len(variants))
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(datasets)), 4))
    # one shaded band per construction (alternating) plus a separator between constructions
    for i in range(0, len(datasets), 2):
        ax.axvspan(i - 0.5, i + 0.5, color="#F0F0F0", zorder=0)
    for i in range(1, len(datasets)):
        ax.axvline(i - 0.5, color="#BBBBBB", linewidth=0.8, zorder=1)
    # thin connector through the versions of one construction, so each group reads as a unit
    for i, ds in enumerate(datasets):
        ax.plot(x[i] + offs, s.loc[ds, variants].to_numpy(float),
                color="#888888", linewidth=0.8, zorder=2)
    for off, v, color, marker in zip(offs, variants, variant_colors, variant_markers):
        ax.plot(x + off, s[v], marker, color=color, linestyle="none", markersize=6.5,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3,
                label=variant_labels[v])
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Share of variation between proteins")
    ax.set_title(f"{col} ({'reference' if set_name == 'ref' else 'test'} set)", fontsize=10)
    side_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_concentration(conc_df, set_name, datasets, std_cols, top_protein_frac, out_path,
                       colors, dpi):
    sub = conc_df[conc_df["set_name"] == set_name]
    x = np.arange(len(datasets))
    width = 0.8 / len(std_cols)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, c in enumerate(std_cols):
        s = sub[sub["centrality"] == c].set_index("dataset").reindex(datasets)
        ax.bar(x - 0.4 + width * (i + 0.5), s["top_fraction_of_proteins"], width,
               color=colors[c], label=c)
    ax.axhline(top_protein_frac, color="black", linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"Share of SS_between from top {top_protein_frac:.0%} of proteins")
    ax.set_title(f"{'Reference' if set_name == 'ref' else 'Test'} set "
                 f"(dotted line = perfectly even contribution)", fontsize=10)
    side_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    std_root,
    output_dir,
    datasets: list,
    std_cols: list,
    tail_threshold: float,
    trim_frac: float,
    leave_out_k: list,
    top_protein_frac: float,
    n_top_proteins_saved: int,
    hist_bin_width: float,
    hist_datasets,
    make_figures: bool,
    colors: dict,
    variant_colors: list,
    variant_markers: list,
    dpi: int,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hist_datasets = datasets if hist_datasets is None else hist_datasets

    dist_rows, rob_rows, conc_rows, top_rows = [], [], [], []
    hist_data = {}

    for set_name in ("ref", "test"):
        print(f"\n══ {set_name} ══")
        frames = {ds: load_set(std_root, ds, set_name, std_cols) for ds in datasets}
        proteins = common_proteins(frames)
        P = len(proteins)
        print(f"  {P} proteins")

        for ds in datasets:
            t0 = time.time()
            X, prot_idx = to_arrays(frames[ds], proteins, std_cols)
            if ds in hist_datasets:
                hist_data[(set_name, ds)] = X

            for c, col in enumerate(std_cols):
                x = X[:, c]
                base = {"dataset": ds, "set_name": set_name, "centrality": col}

                dist_rows.append({**base, **distribution_row(x, tail_threshold)})

                ids, n_i, mean_i, contrib = protein_contributions(x, prot_idx, P)
                order = np.argsort(contrib)[::-1]
                frac = contrib / contrib.sum()
                k_top = max(1, int(np.ceil(top_protein_frac * len(frac))))
                conc_rows.append({
                    **base, "n_proteins": len(ids),
                    "top1_fraction": frac[order[0]],
                    "top_fraction_of_proteins": frac[order[:k_top]].sum(),
                    "effective_n_proteins": 1.0 / (frac ** 2).sum(),
                })
                for rank, j in enumerate(order[:n_top_proteins_saved], start=1):
                    top_rows.append({**base, "rank": rank, "PDBid": proteins[ids[j]],
                                     "n_residues": int(n_i[j]), "protein_mean": mean_i[j],
                                     "ss_between_fraction": frac[j]})

                row = {**base,
                       "share_full": eta_squared(x, prot_idx, P),
                       "share_trimmed": trimmed_share(x, prot_idx, P, trim_frac),
                       "share_rank": eta_squared(rankdata(x), prot_idx, P)}
                top_ids = ids[order]
                for k in leave_out_k:
                    row[f"share_without_top{k}"] = share_without_top_proteins(
                        x, prot_idx, P, k, top_ids)
                rob_rows.append(row)
            print(f"  [{ds}] {len(X):,} residues, {time.time() - t0:.1f} s")

    dist_df = pd.DataFrame(dist_rows)
    rob_df = pd.DataFrame(rob_rows)
    conc_df = pd.DataFrame(conc_rows)
    top_df = pd.DataFrame(top_rows)
    dist_df.to_csv(output_dir / "distribution_summary.csv", index=False)
    rob_df.to_csv(output_dir / "share_robustness.csv", index=False)
    conc_df.to_csv(output_dir / "protein_concentration.csv", index=False)
    top_df.to_csv(output_dir / "protein_top_contributors.csv", index=False)

    variants = ["share_full", "share_trimmed", "share_rank"] + [f"share_without_top{k}" for k in leave_out_k]
    variant_labels = {"share_full": "All residues",
                      "share_trimmed": f"Trimmed ({trim_frac:.0%} most extreme residues)",
                      "share_rank": "Ranks",
                      **{f"share_without_top{k}": f"Without top {k} proteins" for k in leave_out_k}}

    if make_figures:
        fig_root = output_dir / "figures"
        for set_name in ("ref", "test"):
            for ds in hist_datasets:
                d = fig_root / "hist" / set_name
                d.mkdir(parents=True, exist_ok=True)
                plot_hist(hist_data[(set_name, ds)], std_cols, ds, set_name, tail_threshold,
                          hist_bin_width, d / f"{ds}.png", colors, dpi)
            d = fig_root / "robustness" / set_name
            d.mkdir(parents=True, exist_ok=True)
            for col in std_cols:
                plot_robustness(rob_df, set_name, col, variants, variant_labels, datasets,
                                d / f"{col}.png", variant_colors, variant_markers, dpi)
            d = fig_root / "concentration"
            d.mkdir(parents=True, exist_ok=True)
            plot_concentration(conc_df, set_name, datasets, std_cols, top_protein_frac,
                               d / f"concentration_{set_name}.png", colors, dpi)

    for set_name in ("ref", "test"):
        print(f"\n── Between-protein share: full vs. robust versions ({set_name}) ──")
        t = rob_df[rob_df["set_name"] == set_name]
        print(t.drop(columns="set_name").set_index(["dataset", "centrality"])
              .to_string(float_format=lambda v: f"{v:.3f}"))
        print(f"\n── Excess kurtosis ({set_name}) ──")
        t = dist_df[dist_df["set_name"] == set_name]
        print(t.pivot(index="centrality", columns="dataset", values="excess_kurtosis")[datasets]
              .to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":

    # ── Edit these values ─────────────────────────────────────────────────────
    STD_ROOT = Path("...0_4_revisions/2_experiments/5_standardization/...")   # output of standardization_ver3.py
    OUTPUT_DIR = Path("...0_4_revisions/2_experiments/6_bootstrap/...")

    DATASETS = ["arp_iac", "arp_no_iac", "ring_iac", "ring_no_iac", "pcn5", "pcn6", "pcn7"]
    STD_COLS = ["StdDegree", "StdCloseness", "StdBetweenness", "StdPageRank"]

    TAIL_THRESHOLD = 4.0          # |Std| beyond this counts as a tail residue
    TRIM_FRAC = 0.01              # fraction of residues furthest from the mean dropped for share_trimmed
    LEAVE_OUT_K = [5, 10]         # number of top-contributing proteins removed (one share column each)
    TOP_PROTEIN_FRAC = 0.10       # "top X% of proteins" for the concentration figure
    N_TOP_PROTEINS_SAVED = 10     # largest SS_between contributors listed per dataset/set/centrality

    HIST_BIN_WIDTH = 0.25
    HIST_DATASETS = None          # None = histogram for every dataset; or a list, e.g. ["ring_iac", "pcn6"]
    MAKE_FIGURES = True
    COLORS = {"StdDegree": "#4C72B0", "StdCloseness": "#DD8452",
              "StdBetweenness": "#55A868", "StdPageRank": "#C44E52"}
    VARIANT_COLORS = ["#4C72B0", "#C44E52", "#8172B2", "#55A868", "#CCB974"]   # one per share version
    VARIANT_MARKERS = ["o", "s", "^", "D", "v"]                                 # one per share version
    DPI = 300
    # ──────────────────────────────────────────────────────────────────────────

    main(
        std_root=STD_ROOT,
        output_dir=OUTPUT_DIR,
        datasets=DATASETS,
        std_cols=STD_COLS,
        tail_threshold=TAIL_THRESHOLD,
        trim_frac=TRIM_FRAC,
        leave_out_k=LEAVE_OUT_K,
        top_protein_frac=TOP_PROTEIN_FRAC,
        n_top_proteins_saved=N_TOP_PROTEINS_SAVED,
        hist_bin_width=HIST_BIN_WIDTH,
        hist_datasets=HIST_DATASETS,
        make_figures=MAKE_FIGURES,
        colors=COLORS,
        variant_colors=VARIANT_COLORS,
        variant_markers=VARIANT_MARKERS,
        dpi=DPI,
    )
