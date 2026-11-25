# bootstrap_within_protein.py
#
# Purpose: Check whether the POOLED correlations from bootstrap_correlations_ver1.py
# are driven by differences BETWEEN proteins rather than by how residues relate
# WITHIN proteins (Simpson's paradox check).
#
# Why: after reference-based standardization, each protein's residues do not
# average zero. A compact protein can sit high on several Std centralities at
# once and an elongated one low on all of them. Pooling residues then picks up
# that protein-level pattern as if it were a residue-level relationship.
#
# Input : output of standardization_ver3.py
#         <STD_ROOT>/<dataset>/<ref|test>/<PDBid>_std.csv
#
# What is computed, for each dataset and set (ref / test):
#
#   1. Between-protein share of each Std centrality
#        share = variation of protein averages / total variation
#              = SS_between / SS_total  (eta squared; residues weighted equally)
#      0 = protein averages are all the same (pooled = within);
#      1 = all variation is between proteins.
#
#   2. Three correlations for every pair, each with Pearson and Spearman:
#        pooled          : all residues pooled (same as bootstrap_correlations_ver1.py)
#        within_centered : each protein's own mean is subtracted from its residues
#                          before pooling, so only within-protein variation remains.
#                          Spearman version: residues are ranked WITHIN each
#                          protein (ranks scaled to 0-1 so every protein spans the
#                          same range), centred, then pooled.
#        mean_per_protein: correlation computed inside each protein separately,
#                          then averaged across proteins (each protein counts once;
#                          proteins with fewer than MIN_RESIDUES_PER_PROTEIN
#                          residues or no variation are skipped).
#
#   Bootstrap: identical design to bootstrap_correlations_ver1.py (proteins
#   resampled with replacement, N_BOOT draws, same draw shared by every dataset
#   in a set, ref and test drawn independently). With the same SEED, N_BOOT and
#   proteins, the draws are the same as in that script.
#
# Outputs (under OUTPUT_DIR):
#   between_protein_share.csv     share + CI per (dataset, set, centrality)
#   within_vs_pooled.csv          the three correlations + CIs, and
#                                 pooled - within_centered with CI
#   within_test_minus_ref.csv     test - ref for within_centered and
#                                 mean_per_protein, with CI
#   figures/between_share_<set>.png
#   figures/<method>/<set>/pooled_vs_within_<pair>.png
#   figures/<method>/within_test_minus_ref_<pair>.png
#   (one figure per file)
#
# All settings live in the `if __name__ == "__main__":` block at the bottom.

import itertools
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ── loading (same rules as bootstrap_correlations_ver1.py) ───────────────────

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


# ── per-protein building blocks ───────────────────────────────────────────────

def within_protein_ranks(X, prot_idx):
    """Ranks computed inside each protein (ties averaged), scaled to (0, 1)."""
    R = np.empty_like(X)
    order = np.argsort(prot_idx, kind="stable")
    bounds = np.flatnonzero(np.r_[True, np.diff(prot_idx[order]) != 0, True])
    for s, e in zip(bounds[:-1], bounds[1:]):
        rows = order[s:e]
        R[rows] = (rankdata(X[rows], axis=0) - 0.5) / len(rows)
    return R


class ProteinStats:
    """
    Per-protein sums for one variable matrix (raw values or within-protein ranks):
        n  (P,)        residue count
        s1 (P, k)      sums
        s2 (P, k, k)   sums of cross-products
        C  (P, k, k)   within-protein (centred) cross-products = s2 - s1 s1' / n
    """

    def __init__(self, X, prot_idx, n_prot):
        k = X.shape[1]
        self.k = k
        self.n = np.bincount(prot_idx, minlength=n_prot).astype(float)
        self.s1 = np.zeros((n_prot, k))
        np.add.at(self.s1, prot_idx, X)
        self.s2 = np.zeros((n_prot, k * k))
        np.add.at(self.s2, prot_idx, (X[:, :, None] * X[:, None, :]).reshape(len(X), k * k))
        self.s2 = self.s2.reshape(n_prot, k, k)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_outer = np.where(self.n[:, None, None] > 0,
                                  self.s1[:, :, None] * self.s1[:, None, :] / self.n[:, None, None], 0.0)
        self.C = self.s2 - mean_outer

    def per_protein_r(self, pairs, min_residues):
        """(P, n_pairs) correlation inside each protein; NaN where undefined."""
        out = np.full((len(self.n), len(pairs)), np.nan)
        for j, (a, b) in enumerate(pairs):
            den = np.sqrt(self.C[:, a, a] * self.C[:, b, b])
            ok = (self.n >= min_residues) & (den > 1e-12)
            out[ok, j] = self.C[ok, a, b] / den[ok]
        return out


def r_from_moments(N, S1, S2, pairs):
    """Pearson r from pooled moments. N (B,), S1 (B,k), S2 (B,k,k)."""
    out = np.empty((len(N), len(pairs)))
    for j, (a, b) in enumerate(pairs):
        cov = N * S2[:, a, b] - S1[:, a] * S1[:, b]
        out[:, j] = cov / np.sqrt((N * S2[:, a, a] - S1[:, a] ** 2) * (N * S2[:, b, b] - S1[:, b] ** 2))
    return out


def r_from_centred(Csum, pairs):
    """Pearson r from summed within-protein cross-products (mean is 0 by construction)."""
    out = np.empty((len(Csum), len(pairs)))
    for j, (a, b) in enumerate(pairs):
        out[:, j] = Csum[:, a, b] / np.sqrt(Csum[:, a, a] * Csum[:, b, b])
    return out


# ── pooled Spearman (global ranks; same algorithm as bootstrap_correlations_ver1.py)

class PooledSpearman:
    def __init__(self, X, prot_idx, pairs):
        self.prot_idx, self.pairs, self.cols = prot_idx, pairs, []
        for c in range(X.shape[1]):
            order = np.argsort(X[:, c], kind="stable")
            v = X[order, c]
            starts = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])
            group_id = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(v)]))
            self.cols.append((order, starts, group_id))

    def r(self, W):
        w = np.atleast_2d(W).astype(float)[:, self.prot_idx]
        ranks = []
        for order, starts, group_id in self.cols:
            gsum = np.add.reduceat(w[:, order], starts, axis=1)
            mid = np.cumsum(gsum, axis=1) - gsum + (gsum + 1.0) / 2.0
            R = np.empty_like(w)
            R[:, order] = mid[:, group_id]
            ranks.append(R)
        N = w.sum(axis=1)
        S1 = [(w * R).sum(axis=1) for R in ranks]
        S2 = [(w * R * R).sum(axis=1) for R in ranks]
        out = np.empty((len(w), len(self.pairs)))
        for j, (a, b) in enumerate(self.pairs):
            cov = N * (w * ranks[a] * ranks[b]).sum(axis=1) - S1[a] * S1[b]
            out[:, j] = cov / np.sqrt((N * S2[a] - S1[a] ** 2) * (N * S2[b] - S1[b] ** 2))
        return out


# ── one (dataset, set): all statistics for a block of draws ──────────────────

class DatasetEngine:
    def __init__(self, X, prot_idx, n_prot, pairs, methods, min_residues):
        self.pairs, self.methods = pairs, methods
        self.raw = ProteinStats(X, prot_idx, n_prot)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.s1sq_over_n = np.where(self.raw.n[:, None] > 0,
                                        self.raw.s1 ** 2 / self.raw.n[:, None], 0.0)
        self.s2diag = np.einsum("pkk->pk", self.raw.s2)
        self.per_r = {"pearson": self.raw.per_protein_r(pairs, min_residues)}
        if "spearman" in methods:
            self.rank = ProteinStats(within_protein_ranks(X, prot_idx), prot_idx, n_prot)
            self.per_r["spearman"] = self.rank.per_protein_r(pairs, min_residues)
            self.pooled_spearman = PooledSpearman(X, prot_idx, pairs)
        self.n_valid_proteins = {m: int(np.isfinite(v).all(axis=1).sum()) for m, v in self.per_r.items()}

    def compute(self, W):
        W = np.atleast_2d(W).astype(float)
        B, k = len(W), self.raw.k
        out = {}

        # between-protein share per centrality
        N = W @ self.raw.n
        S = W @ self.raw.s1
        sst = W @ self.s2diag - S ** 2 / N[:, None]
        ssb = W @ self.s1sq_over_n - S ** 2 / N[:, None]
        out["between_share"] = ssb / sst                                   # (B, k)

        for m in self.methods:
            if m == "pearson":
                S2 = (W @ self.raw.s2.reshape(len(self.raw.n), k * k)).reshape(B, k, k)
                pooled = r_from_moments(N, S, S2, self.pairs)
                Csum = (W @ self.raw.C.reshape(len(self.raw.n), k * k)).reshape(B, k, k)
            else:
                pooled = self.pooled_spearman.r(W)
                Csum = (W @ self.rank.C.reshape(len(self.rank.n), k * k)).reshape(B, k, k)
            pr = self.per_r[m]
            valid = np.isfinite(pr)
            mean_pp = (W @ np.where(valid, pr, 0.0)) / (W @ valid.astype(float))
            out[m] = {"pooled": pooled,
                      "within_centered": r_from_centred(Csum, self.pairs),
                      "mean_per_protein": mean_pp}
        return out


def run_in_batches(engine, W, batch_size):
    parts = [engine.compute(W[i:i + batch_size]) for i in range(0, len(W), batch_size)]
    merged = {"between_share": np.vstack([p["between_share"] for p in parts])}
    for m in engine.methods:
        merged[m] = {kind: np.vstack([p[m][kind] for p in parts]) for kind in parts[0][m]}
    return merged


# ── summaries ─────────────────────────────────────────────────────────────────

def ci_row(point, reps, ci_level):
    lo, hi = np.percentile(reps, [50 * (1 - ci_level), 50 * (1 + ci_level)])
    return {"estimate": point, "ci_low": lo, "ci_high": hi, "boot_se": reps.std(ddof=1),
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


KINDS = ["pooled", "within_centered", "mean_per_protein"]
KIND_LABELS = {"pooled": "Pooled", "within_centered": "Within (centred)",
               "mean_per_protein": "Mean per protein"}


# ── plotting (one figure per file) ────────────────────────────────────────────

def plot_between_share(share_df, set_name, datasets, std_cols, out_path, dpi):
    sub = share_df[share_df["set_name"] == set_name]
    x = np.arange(len(datasets))
    width = 0.8 / len(std_cols)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, c in enumerate(std_cols):
        s = sub[sub["centrality"] == c].set_index("dataset").reindex(datasets)
        xs = x - 0.4 + width * (i + 0.5)
        ax.bar(xs, s["estimate"], width, label=c)
        ax.errorbar(xs, s["estimate"], yerr=[s["estimate"] - s["ci_low"], s["ci_high"] - s["estimate"]],
                    fmt="none", ecolor="black", capsize=2, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Share of variation between proteins (95% CI)")
    ax.set_title(f"{'Reference' if set_name == 'ref' else 'Test'} set", fontsize=10)
    # legend sits outside the axes (right of the bars) so it never covers data
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pooled_vs_within(cmp_df, method, set_name, pair, datasets, out_path, colors, dpi):
    sub = cmp_df[(cmp_df["method"] == method) & (cmp_df["set_name"] == set_name) & (cmp_df["pair"] == pair)]
    s = sub.set_index("dataset").reindex(datasets)
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for off, kind in zip((-0.2, 0.0, 0.2), KINDS):
        est = s[f"{kind}_r"]
        ax.errorbar(x + off, est, yerr=[est - s[f"{kind}_ci_low"], s[f"{kind}_ci_high"] - est],
                    fmt="o", color=colors[kind], capsize=3, markersize=5, label=KIND_LABELS[kind])
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel(f"{method.capitalize()} r (95% CI)")
    ax.set_title(f"{pair.replace('__', ' vs ')} ({'reference' if set_name == 'ref' else 'test'})",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_within_test_minus_ref(diff_df, method, pair, datasets, out_path, colors, dpi):
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.axhline(0, color="black", linewidth=0.8)
    for off, kind in zip((-0.2, 0.0, 0.2), KINDS):
        s = diff_df[(diff_df["method"] == method) & (diff_df["pair"] == pair)
                    & (diff_df["kind"] == kind)].set_index("dataset").reindex(datasets)
        ax.errorbar(x + off, s["estimate"],
                    yerr=[s["estimate"] - s["ci_low"], s["ci_high"] - s["estimate"]],
                    fmt="o", color=colors[kind], capsize=3, markersize=5, label=KIND_LABELS[kind])
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel(f"{method.capitalize()} r: test − reference (95% CI)")
    ax.set_title(pair.replace("__", " vs "), fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_all_figures(output_dir, share_df, cmp_df, diff_df, datasets, std_cols,
                     methods, pair_labels, colors, dpi):
    fig_root = Path(output_dir) / "figures"
    fig_root.mkdir(parents=True, exist_ok=True)
    for set_name in ("ref", "test"):
        plot_between_share(share_df, set_name, datasets, std_cols,
                           fig_root / f"between_share_{set_name}.tif", dpi)
    for m in methods:
        for set_name in ("ref", "test"):
            d = fig_root / m / set_name
            d.mkdir(parents=True, exist_ok=True)
            for lab in pair_labels:
                plot_pooled_vs_within(cmp_df, m, set_name, lab, datasets,
                                      d / f"pooled_vs_within_{lab}.tif", colors, dpi)
        for lab in pair_labels:
            plot_within_test_minus_ref(diff_df, m, lab, datasets,
                                       fig_root / m / f"within_test_minus_ref_{lab}.tif", colors, dpi)


def replot_from_csvs(output_dir, datasets, std_cols, methods, pair_labels, colors, dpi):
    """Redraw every figure from the CSVs saved by an earlier run (no recomputation)."""
    output_dir = Path(output_dir)
    files = {"share": "between_protein_share.csv", "cmp": "within_vs_pooled.csv",
             "diff": "within_test_minus_ref.csv"}
    for f in files.values():
        if not (output_dir / f).exists():
            raise FileNotFoundError(f"REPLOT_ONLY needs {output_dir / f} from an earlier full run")
    share_df = pd.read_csv(output_dir / files["share"])
    cmp_df = pd.read_csv(output_dir / files["cmp"])
    diff_df = pd.read_csv(output_dir / files["diff"])
    print(f"Re-plotting from saved CSVs in {output_dir} (no bootstrap recomputation)")
    make_all_figures(output_dir, share_df, cmp_df, diff_df, datasets, std_cols,
                     methods, pair_labels, colors, dpi)
    print(f"Figures saved to: {output_dir / 'figures'}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    std_root,
    output_dir,
    datasets: list,
    std_cols: list,
    methods: list,
    n_boot: int,
    ci_level: float,
    seed: int,
    batch_size: int,
    min_residues_per_protein: int,
    make_figures: bool,
    colors: dict,
    dpi: int,
    replot_only: bool = False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pairs = list(itertools.combinations(range(len(std_cols)), 2))
    pair_labels = [f"{std_cols[a]}__{std_cols[b]}" for a, b in pairs]

    if replot_only:
        replot_from_csvs(output_dir, datasets, std_cols, methods, pair_labels, colors, dpi)
        return

    point, reps = {}, {}
    share_rows, cmp_rows = [], []

    for set_name in ("ref", "test"):                       # same order as ver1 -> same draws
        print(f"\n══ {set_name} ══")
        frames = {ds: load_set(std_root, ds, set_name, std_cols) for ds in datasets}
        proteins = common_proteins(frames)
        P = len(proteins)
        W = rng.multinomial(P, np.full(P, 1.0 / P), size=n_boot)
        print(f"  {P} proteins, {n_boot} bootstrap draws")
        point[set_name], reps[set_name] = {}, {}

        for ds in datasets:
            t0 = time.time()
            X, prot_idx = to_arrays(frames[ds], proteins, std_cols)
            eng = DatasetEngine(X, prot_idx, P, pairs, methods, min_residues_per_protein)
            pt = eng.compute(np.ones((1, P)))
            rp = run_in_batches(eng, W, batch_size)
            point[set_name][ds], reps[set_name][ds] = pt, rp

            for c, col in enumerate(std_cols):
                share_rows.append({"dataset": ds, "set_name": set_name, "centrality": col,
                                   **ci_row(pt["between_share"][0, c], rp["between_share"][:, c], ci_level)})
            for m in methods:
                for j, lab in enumerate(pair_labels):
                    row = {"dataset": ds, "set_name": set_name, "method": m, "pair": lab,
                           "n_proteins": P, "n_residues": len(X),
                           "n_proteins_with_own_r": eng.n_valid_proteins[m]}
                    for kind in KINDS:
                        c = ci_row(pt[m][kind][0, j], rp[m][kind][:, j], ci_level)
                        row[f"{kind}_r"], row[f"{kind}_ci_low"], row[f"{kind}_ci_high"] = (
                            c["estimate"], c["ci_low"], c["ci_high"])
                    d = ci_row(pt[m]["pooled"][0, j] - pt[m]["within_centered"][0, j],
                               rp[m]["pooled"][:, j] - rp[m]["within_centered"][:, j], ci_level)
                    row.update({"pooled_minus_within": d["estimate"],
                                "pooled_minus_within_ci_low": d["ci_low"],
                                "pooled_minus_within_ci_high": d["ci_high"],
                                "pooled_minus_within_excludes_zero": d["ci_excludes_zero"]})
                    cmp_rows.append(row)
            print(f"  [{ds}] {len(X):,} residues, {time.time() - t0:.1f} s")

    share_df = pd.DataFrame(share_rows)
    share_df.to_csv(output_dir / "between_protein_share.csv", index=False)
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(output_dir / "within_vs_pooled.csv", index=False)

    diff_rows = []
    for ds in datasets:
        for m in methods:
            for kind in KINDS:
                for j, lab in enumerate(pair_labels):
                    diff_rows.append({
                        "dataset": ds, "method": m, "kind": kind, "pair": lab,
                        **ci_row(point["test"][ds][m][kind][0, j] - point["ref"][ds][m][kind][0, j],
                                 reps["test"][ds][m][kind][:, j] - reps["ref"][ds][m][kind][:, j],
                                 ci_level),
                    })
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(output_dir / "within_test_minus_ref.csv", index=False)

    if make_figures:
        make_all_figures(output_dir, share_df, cmp_df, diff_df, datasets, std_cols,
                         methods, pair_labels, colors, dpi)

    print("\n── Between-protein share of variation (test set) ────────────")
    print(share_df[share_df["set_name"] == "test"]
          .pivot(index="centrality", columns="dataset", values="estimate")[datasets]
          .to_string(float_format=lambda v: f"{v:.3f}"))
    for m in methods:
        print(f"\n── {m}: pooled - within_centered (test set) ─────────────")
        t = cmp_df[(cmp_df["method"] == m) & (cmp_df["set_name"] == "test")]
        print(t.pivot(index="pair", columns="dataset", values="pooled_minus_within")[datasets]
              .to_string(float_format=lambda v: f"{v:+.3f}"))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":

    # ── Edit these values ─────────────────────────────────────────────────────
    STD_ROOT = Path("...0_4_revisions/2_experiments/5_standardization/...")        # output of standardization_ver3.py
    OUTPUT_DIR = Path("...0_4_revisions/2_experiments/6_bootstrap/std_ver4/within_protein")

    DATASETS = ["arp_iac", "arp_no_iac", "ring_iac", "ring_no_iac", "pcn5", "pcn6", "pcn7"]
    STD_COLS = ["StdDegree", "StdCloseness", "StdBetweenness", "StdPageRank"]
    METHODS = ["spearman", "pearson"]                    # either or both

    # Keep these equal to bootstrap_correlations_ver1.py so the draws match
    N_BOOT = 10_000
    CI_LEVEL = 0.95
    SEED = 20261115

    BATCH_SIZE = 100                 # replicates computed at once; lower if memory is tight
    MIN_RESIDUES_PER_PROTEIN = 10    # proteins smaller than this get no per-protein r
    MAKE_FIGURES = True
    COLORS = {"pooled": "#4C72B0", "within_centered": "#C44E52", "mean_per_protein": "#8172B2"}
    DPI = 300

    # True  = skip all computation and redraw figures from the CSVs already in OUTPUT_DIR
    #         (needs a previous full run; STD_ROOT, N_BOOT, SEED etc. are then ignored)
    # False = full run (recompute everything, overwrite CSVs and figures)
    REPLOT_ONLY = True
    # ──────────────────────────────────────────────────────────────────────────

    main(
        std_root=STD_ROOT,
        output_dir=OUTPUT_DIR,
        datasets=DATASETS,
        std_cols=STD_COLS,
        methods=METHODS,
        n_boot=N_BOOT,
        ci_level=CI_LEVEL,
        seed=SEED,
        batch_size=BATCH_SIZE,
        min_residues_per_protein=MIN_RESIDUES_PER_PROTEIN,
        make_figures=MAKE_FIGURES,
        colors=COLORS,
        dpi=DPI,
        replot_only=REPLOT_ONLY,
    )
