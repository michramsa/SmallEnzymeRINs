# bootstrap_correlations_ver1.py
#
# Purpose: Correlations between the standardized centralities (all pairs of
# StdDegree, StdCloseness, StdBetweenness, StdPageRank), with 95% confidence
# intervals from a protein-level bootstrap.
#
# Input : output of standardization_ver3.py
#         <STD_ROOT>/<dataset>/<ref|test>/<PDBid>_std.csv
#
# Method
#   Point estimate : pool every residue of every protein in a (dataset, set) and
#                    compute r (Pearson and Spearman) for each pair.
#   Bootstrap      : draw N proteins WITH replacement from the N proteins of the
#                    set (N = 232 test, 534 ref; nothing is removed on purpose;
#                    on average ~37% of proteins are left out of each draw by
#                    chance and others appear more than once). Pool the residues
#                    of the drawn proteins (a protein drawn twice counts twice)
#                    and recompute r. Repeat N_BOOT times.
#   95% CI         : 2.5th and 97.5th percentiles of the bootstrap r values.
#   Proteins, not residues, are resampled because residues within a protein
#   are not independent.
#
#   Paired across datasets: within each set, every replicate uses the SAME
#   protein draw in all datasets, so differences between graph constructions
#   (e.g. r[ring_iac] - r[pcn6]) get their own CIs. Only proteins present in
#   every dataset are used.
#   Test - reference: test and reference proteins are drawn independently;
#   the CI of the difference comes from subtracting replicate by replicate.
#
#   Residues used: canonical residues with all four Std values present
#   (non-canonical residues are NaN after standardization). The same residues
#   are used for every pair.
#
#   Computation: replicates are done with resampling COUNTS (how many times each
#   protein was drawn) rather than by copying residues, which gives exactly the
#   same r values much faster. Pearson uses per-protein sums; Spearman sorts each
#   centrality once and recomputes tie-averaged ranks with the counts.
#
# Outputs (under OUTPUT_DIR):
#   correlations.csv               r + CI per (dataset, set, method, pair)
#   diff_between_datasets.csv      r[dataset A] - r[dataset B] + CI, per set/method/pair
#   diff_test_minus_ref.csv        r[test] - r[ref] + CI, per dataset/method/pair
#   bootstrap_replicates.npz       all replicate r values (optional)
#   figures/<method>/r_<pair>.png                 r +/- CI across datasets, ref and test
#   figures/<method>/test_minus_ref_<pair>.png    test - ref difference +/- CI
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


# ── loading ───────────────────────────────────────────────────────────────────

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
    """Proteins present in every dataset (needed for paired draws across datasets)."""
    sets = [set(df["PDBid"]) for df in frames.values()]
    common = set.intersection(*sets)
    for ds, s in zip(frames, sets):
        if len(s - common):
            print(f"    NOTE [{ds}]: {len(s - common)} protein(s) not in every dataset, excluded")
    return sorted(common)


def to_arrays(df, proteins, std_cols):
    """Residue matrix X (n x k) and the protein index of each residue."""
    pos = {p: i for i, p in enumerate(proteins)}
    df = df[df["PDBid"].isin(pos) & df["IsCanonical"].astype(bool)].dropna(subset=std_cols)
    X = df[std_cols].to_numpy(float)
    prot_idx = df["PDBid"].map(pos).to_numpy(int)
    return X, prot_idx


# ── Pearson via per-protein sufficient statistics ─────────────────────────────

class PearsonEngine:
    def __init__(self, X, prot_idx, n_prot, pairs):
        k = X.shape[1]
        self.pairs = pairs
        self.n = np.bincount(prot_idx, minlength=n_prot).astype(float)               # (P,)
        self.s1 = np.zeros((n_prot, k))
        np.add.at(self.s1, prot_idx, X)                                               # (P, k)
        outer = (X[:, :, None] * X[:, None, :]).reshape(len(X), k * k)
        self.s2 = np.zeros((n_prot, k * k))
        np.add.at(self.s2, prot_idx, outer)                                           # (P, k*k)
        self.k = k

    def r(self, W):
        """W: (B, P) resampling counts -> (B, n_pairs) Pearson r."""
        W = np.atleast_2d(W).astype(float)
        N = W @ self.n
        S1 = W @ self.s1
        S2 = (W @ self.s2).reshape(len(W), self.k, self.k)
        out = np.empty((len(W), len(self.pairs)))
        for j, (a, b) in enumerate(self.pairs):
            cov = N * S2[:, a, b] - S1[:, a] * S1[:, b]
            va = N * S2[:, a, a] - S1[:, a] ** 2
            vb = N * S2[:, b, b] - S1[:, b] ** 2
            out[:, j] = cov / np.sqrt(va * vb)
        return out


# ── Spearman via weighted tie-averaged ranks ─────────────────────────────────

class SpearmanEngine:
    """
    For a replicate with residue weights w (= draw count of the residue's protein),
    the Spearman r of the resampled data equals the w-weighted Pearson r of
    tie-averaged ranks, where the rank of a value is computed from the weighted
    counts. Each column is sorted once; replicates only need cumulative sums.
    """

    def __init__(self, X, prot_idx, pairs):
        self.prot_idx = prot_idx
        self.pairs = pairs
        self.cols = []
        for c in range(X.shape[1]):
            order = np.argsort(X[:, c], kind="stable")
            v = X[order, c]
            starts = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])
            group_id = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(v)]))
            self.cols.append((order, starts, group_id))

    def r(self, W):
        W = np.atleast_2d(W).astype(float)
        w = W[:, self.prot_idx]                                   # (B, n) residue weights
        ranks = []
        for order, starts, group_id in self.cols:
            ws = w[:, order]
            gsum = np.add.reduceat(ws, starts, axis=1)            # weight of each tie group
            before = np.cumsum(gsum, axis=1) - gsum
            mid = before + (gsum + 1.0) / 2.0                     # average rank of the group
            R = np.empty_like(w)
            R[:, order] = mid[:, group_id]
            ranks.append(R)

        N = w.sum(axis=1)
        S1 = [(w * R).sum(axis=1) for R in ranks]
        S2 = [(w * R * R).sum(axis=1) for R in ranks]
        out = np.empty((len(W), len(self.pairs)))
        for j, (a, b) in enumerate(self.pairs):
            sab = (w * ranks[a] * ranks[b]).sum(axis=1)
            cov = N * sab - S1[a] * S1[b]
            out[:, j] = cov / np.sqrt((N * S2[a] - S1[a] ** 2) * (N * S2[b] - S1[b] ** 2))
        return out


def run_engine(engine, W_all, batch_size):
    return np.vstack([engine.r(W_all[i:i + batch_size]) for i in range(0, len(W_all), batch_size)])


# ── summaries ─────────────────────────────────────────────────────────────────

def ci_row(point, reps, ci_level):
    lo, hi = np.percentile(reps, [50 * (1 - ci_level), 50 * (1 + ci_level)])
    return {"estimate": point, "ci_low": lo, "ci_high": hi,
            "boot_mean": reps.mean(), "boot_se": reps.std(ddof=1),
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


# ── plotting (one figure per file) ────────────────────────────────────────────

def plot_r_across_datasets(corr, method, pair_label, datasets, out_path, colors, dpi):
    sub = corr[(corr["method"] == method) & (corr["pair"] == pair_label)]
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for off, set_name, name in ((-0.12, "ref", "Reference"), (0.12, "test", "Test")):
        s = sub[sub["set_name"] == set_name].set_index("dataset").reindex(datasets)
        ax.errorbar(x + off, s["estimate"],
                    yerr=[s["estimate"] - s["ci_low"], s["ci_high"] - s["estimate"]],
                    fmt="o", color=colors[set_name], capsize=3, markersize=5, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel(f"{method.capitalize()} r (95% CI)")
    ax.set_title(pair_label.replace("__", " vs "), fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_test_minus_ref(diff, method, pair_label, datasets, out_path, color, dpi):
    s = diff[(diff["method"] == method) & (diff["pair"] == pair_label)].set_index("dataset").reindex(datasets)
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.errorbar(x, s["estimate"], yerr=[s["estimate"] - s["ci_low"], s["ci_high"] - s["estimate"]],
                fmt="o", color=color, capsize=3, markersize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel(f"{method.capitalize()} r: test − reference (95% CI)")
    ax.set_title(pair_label.replace("__", " vs "), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


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
    save_replicates: bool,
    make_figures: bool,
    colors: dict,
    dpi: int,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pair_idx = list(itertools.combinations(range(len(std_cols)), 2))
    pair_labels = [f"{std_cols[a]}__{std_cols[b]}" for a, b in pair_idx]

    # point[set][method]: (D, n_pairs);  reps[set][method]: (D, B, n_pairs)
    point, reps, corr_rows = {}, {}, []

    for set_name in ("ref", "test"):
        print(f"\n══ {set_name} ══")
        frames = {ds: load_set(std_root, ds, set_name, std_cols) for ds in datasets}
        proteins = common_proteins(frames)
        P = len(proteins)
        # one draw per replicate, shared by every dataset and method in this set
        W = rng.multinomial(P, np.full(P, 1.0 / P), size=n_boot)
        print(f"  {P} proteins, {n_boot} bootstrap draws "
              f"(mean proteins left out per draw: {(W == 0).sum(axis=1).mean():.1f})")

        point[set_name] = {m: np.empty((len(datasets), len(pair_idx))) for m in methods}
        reps[set_name] = {m: np.empty((len(datasets), n_boot, len(pair_idx))) for m in methods}

        for d, ds in enumerate(datasets):
            X, prot_idx = to_arrays(frames[ds], proteins, std_cols)
            ones = np.ones((1, P))
            for m in methods:
                t0 = time.time()
                eng = (PearsonEngine(X, prot_idx, P, pair_idx) if m == "pearson"
                       else SpearmanEngine(X, prot_idx, pair_idx))
                point[set_name][m][d] = eng.r(ones)[0]
                reps[set_name][m][d] = run_engine(eng, W, batch_size)
                for j, lab in enumerate(pair_labels):
                    corr_rows.append({
                        "dataset": ds, "set_name": set_name, "method": m, "pair": lab,
                        "var1": std_cols[pair_idx[j][0]], "var2": std_cols[pair_idx[j][1]],
                        "n_proteins": P, "n_residues": len(X),
                        **ci_row(point[set_name][m][d, j], reps[set_name][m][d, :, j], ci_level),
                    })
                print(f"  [{ds}] {m:8s} {len(X):,} residues, {time.time() - t0:.1f} s")

    corr = pd.DataFrame(corr_rows)
    corr.to_csv(output_dir / "correlations.csv", index=False)

    # between-dataset differences (paired draws within each set)
    rows = []
    for set_name in ("ref", "test"):
        for m in methods:
            for a, b in itertools.combinations(range(len(datasets)), 2):
                for j, lab in enumerate(pair_labels):
                    rows.append({
                        "set_name": set_name, "method": m, "pair": lab,
                        "dataset_A": datasets[a], "dataset_B": datasets[b],
                        **ci_row(point[set_name][m][a, j] - point[set_name][m][b, j],
                                 reps[set_name][m][a, :, j] - reps[set_name][m][b, :, j], ci_level),
                    })
    pd.DataFrame(rows).to_csv(output_dir / "diff_between_datasets.csv", index=False)

    # test - reference differences (independent draws)
    rows = []
    for m in methods:
        for d, ds in enumerate(datasets):
            for j, lab in enumerate(pair_labels):
                rows.append({
                    "dataset": ds, "method": m, "pair": lab,
                    **ci_row(point["test"][m][d, j] - point["ref"][m][d, j],
                             reps["test"][m][d, :, j] - reps["ref"][m][d, :, j], ci_level),
                })
    diff_tr = pd.DataFrame(rows)
    diff_tr.to_csv(output_dir / "diff_test_minus_ref.csv", index=False)

    if save_replicates:
        np.savez_compressed(
            output_dir / "bootstrap_replicates.npz",
            datasets=np.array(datasets), pairs=np.array(pair_labels),
            **{f"{s}_{m}": reps[s][m] for s in reps for m in methods},
        )

    if make_figures:
        for m in methods:
            fig_dir = output_dir / "figures" / m
            fig_dir.mkdir(parents=True, exist_ok=True)
            for lab in pair_labels:
                plot_r_across_datasets(corr, m, lab, datasets, fig_dir / f"r_{lab}.png", colors, dpi)
                plot_test_minus_ref(diff_tr, m, lab, datasets,
                                    fig_dir / f"test_minus_ref_{lab}.png", colors["diff"], dpi)

    for m in methods:
        print(f"\n── {m}: r [95% CI], test set ──")
        t = corr[(corr["method"] == m) & (corr["set_name"] == "test")].copy()
        t["value"] = t.apply(lambda r: f"{r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1)
        print(t.pivot(index="pair", columns="dataset", values="value")[datasets].to_string())
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":

    # ── Edit these values ─────────────────────────────────────────────────────
    STD_ROOT = Path("...0_4_revisions/2_experiments/5_standardization/...")        # output of standardization_ver3.py
    OUTPUT_DIR = Path("...0_4_revisions/2_experiments/6_bootstrap/...")

    DATASETS = ["arp_iac", "arp_no_iac", "ring_iac", "ring_no_iac", "pcn5", "pcn6", "pcn7"]
    STD_COLS = ["StdDegree", "StdCloseness", "StdBetweenness", "StdPageRank"]
    METHODS = ["spearman", "pearson"]                    # either or both

    N_BOOT = 10000
    CI_LEVEL = 0.95
    SEED = 20261115                                      # fixed for reproducibility
    BATCH_SIZE = 100       # replicates computed at once; lower it if memory is tight
    SAVE_REPLICATES = True
    MAKE_FIGURES = True
    COLORS = {"ref": "#4C72B0", "test": "#DD8452", "diff": "#55A868"}
    DPI = 300
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
        save_replicates=SAVE_REPLICATES,
        make_figures=MAKE_FIGURES,
        colors=COLORS,
        dpi=DPI,
    )
