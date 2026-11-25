"""
amitai_jackknife_stage3_ver1.py
===============================
Stage 3 of the Amitai (2004)-style jackknife: figures. Reads the Stage 1 and
Stage 2 outputs; computes nothing new except the full-data MCC surface for the
heatmaps. Every figure is its own file.

Figures (under <out_root>/stage3/<grid_name>/, one file each)
-------------------------------------------------------------
    roc/roc_<model>                 held-out ROC (windowed score) with 95% band,
                                    unwindowed score (dashed), the model's
                                    operating point, the RSA-only operating
                                    point, and the random-classifier diagonal
    pr/pr_<model>                   same for precision-recall; random baseline =
                                    fraction of residues that are functional
    roc_compare/roc_<feature>       all datasets overlaid for one centrality
    pr_compare/pr_<feature>         same for precision-recall
    compare/<metric>_<feature>      datasets side by side (estimate + 95% CI),
                                    RSA-only as a reference line and band
    heatmap/mcc_<model>             median MCC over ALL proteins (descriptive)
                                    across RSA lower x RSA upper, at the
                                    centrality bound chosen most often in the
                                    leave-one-out; that combination is outlined
    partner/partner_<model>         pooled sensitivity per partner type with
                                    95% CI, random expectation (a) and (b)
    null/null_<model>               random-classifier null distribution of the
                                    median MCC (a, b) and the observed value
    stability/stability_<model>     Bootstrap A: how often the most frequently
                                    selected combinations were chosen

Style: uses figure_style_ver2.py (apply_style / save_fig / COLORS) if it can
be imported from STYLE_DIR; otherwise a minimal built-in style.

All settings live in the `if __name__ == "__main__":` block at the bottom.
"""

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from amitai_jackknife_stage1_ver1 import mcc_from_counts  # noqa: E402

# Built-in fallback style (same palette as figure_style_ver2.py)
_FALLBACK_COLORS = ["#4878CF", "#D65F00", "#59A14F", "#B07AA1",
                    "#76B7B2", "#E15759", "#EDC948", "#9C755F"]
_GRAY = "#8D8D8D"
_DARK = "#2C2C2C"


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

class Style:
    def __init__(self, style_dir, style_module, fig_format, dpi):
        self.fmt = fig_format
        self.dpi = dpi
        self.mod = None
        if style_dir is not None:
            sys.path.insert(0, str(style_dir))
        try:
            self.mod = importlib.import_module(style_module)
            self.mod.apply_style()
            self.colors = list(self.mod.COLOR_CYCLE)
            self.fig_size = self.mod.FIG_SIZE
            print(f"Using figure style from {style_module}")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not import {style_module} ({exc}); using built-in style")
            plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                                 "font.size": 9, "axes.titlesize": 10,
                                 "axes.titleweight": "bold", "legend.fontsize": 8,
                                 "xtick.labelsize": 8, "ytick.labelsize": 8})
            self.colors = _FALLBACK_COLORS
            self.fig_size = {"single_sq": (3.27, 3.27), "half_page": (4.96, 3.2),
                             "full_width": (6.83, 3.5)}

        plt.rcParams["axes.axisbelow"] = True   # grid lines behind bars

    def size(self, key):
        return self.fig_size.get(key, (4.96, 3.2))

    def new_fig(self, key, legend_in):
        """
        Figure whose plotting area keeps the size of FIG_SIZE[key], with an
        extra `legend_in` inches on the right for the legend.
        """
        w, h = self.size(key)
        left, bottom, right, top = 0.6, 0.5, 0.15, 0.35      # margins (inches)
        fig = plt.figure(figsize=(w + legend_in, h))
        ax = fig.add_axes([left / (w + legend_in), bottom / h,
                           (w - left - right) / (w + legend_in), (h - bottom - top) / h])
        fig._fixed_layout = True
        return fig, ax

    def save(self, fig, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path = path.with_suffix(f".{self.fmt}")
        if not getattr(fig, "_fixed_layout", False):
            fig.tight_layout()
        # bbox_inches="tight" keeps tick labels, titles and the side legend in the file
        if self.mod is not None and hasattr(self.mod, "save_fig"):
            self.mod.save_fig(fig, path, dpi=self.dpi, fmt=self.fmt)
        else:
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight", format=self.fmt)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def label_of(model, dataset_labels, feature_labels):
    if model == "rsa_only":
        return "RSA only"
    ds, f = model.split("__", 1)
    return f"{feature_labels.get(f, f)} ({dataset_labels.get(ds, ds)})"


def side_legend(ax, title=None):
    """Legend outside the plot, to the right of the axes (never on the data)."""
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0,
              frameon=False, fontsize=7, title=title, title_fontsize=7)


def fmt_ci(est, lo, hi):
    return f"{est:.2f} [{lo:.2f}, {hi:.2f}]"


# ---------------------------------------------------------------------------
# ROC / PR per model
# ---------------------------------------------------------------------------

def plot_roc(model, roc, ops, summ, st, title, rsa_point):
    fig, ax = st.new_fig("single_sq", legend_in=2.4)
    c = st.colors[0]
    w = roc[(roc["model"] == model) & (roc["score"] == "windowed")]
    u = roc[(roc["model"] == model) & (roc["score"] == "unwindowed")]
    s = summ.loc[model]
    ax.fill_between(w["fpr"], w["tpr_lo"], w["tpr_hi"], color=c, alpha=0.2, lw=0)
    ax.plot(w["fpr"], w["tpr"], color=c,
            label=f"In RSA window, AUC {fmt_ci(s['auc_windowed'], s['auc_windowed_lo'], s['auc_windowed_hi'])}")
    if len(u):
        ax.plot(u["fpr"], u["tpr"], color=c, ls="--", lw=1,
                label=f"No RSA window, AUC {s['auc_unwindowed']:.2f}")
    ax.plot([0, 1], [0, 1], color=_GRAY, ls=":", lw=1, label="Random (AUC 0.50)")
    op = ops.loc[model]
    ax.plot(op["fpr"], op["tpr"], "o", color=_DARK, ms=6, label="Jackknife rule")
    if rsa_point is not None:
        ax.plot(rsa_point["fpr"], rsa_point["tpr"], "s", color=st.colors[1], ms=5,
                label="RSA-only rule")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(title, fontsize=9)
    side_legend(ax)
    return fig


def plot_pr(model, pr, ops, summ, st, title, rsa_point):
    fig, ax = st.new_fig("single_sq", legend_in=2.4)
    c = st.colors[0]
    w = pr[(pr["model"] == model) & (pr["score"] == "windowed")]
    u = pr[(pr["model"] == model) & (pr["score"] == "unwindowed")]
    s = summ.loc[model]
    op = ops.loc[model]
    ax.fill_between(w["recall"], w["precision_lo"], w["precision_hi"],
                    color=c, alpha=0.2, lw=0, step="post")
    ax.step(w["recall"], w["precision"], where="post", color=c,
            label=f"In RSA window, AP {fmt_ci(s['ap_windowed'], s['ap_windowed_lo'], s['ap_windowed_hi'])}")
    if len(u):
        ax.step(u["recall"], u["precision"], where="post", color=c, ls="--", lw=1,
                label=f"No RSA window, AP {s['ap_unwindowed']:.2f}")
    ax.axhline(op["prevalence"], color=_GRAY, ls=":", lw=1,
               label=f"Random ({op['prevalence']:.3f})")
    ax.plot(op["tpr"], op["precision"], "o", color=_DARK, ms=6, label="Jackknife rule")
    if rsa_point is not None:
        ax.plot(rsa_point["tpr"], rsa_point["precision"], "s", color=st.colors[1], ms=5,
                label="RSA-only rule")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontsize=9)
    side_legend(ax)
    return fig


def plot_compare_curves(feature, datasets, curves, summ, st, kind, prevalence,
                        dataset_labels, feature_labels):
    fig, ax = st.new_fig("single_sq", legend_in=1.4)
    xcol, ycol = ("fpr", "tpr") if kind == "roc" else ("recall", "precision")
    metric = "auc_windowed" if kind == "roc" else "ap_windowed"
    for i, ds in enumerate(datasets):
        m = f"{ds}__{feature}"
        d = curves[(curves["model"] == m) & (curves["score"] == "windowed")]
        if d.empty:
            continue
        lab = f"{dataset_labels.get(ds, ds)} ({summ.loc[m, metric]:.2f})"
        if kind == "roc":
            ax.plot(d[xcol], d[ycol], color=st.colors[i % len(st.colors)], lw=1.2, label=lab)
        else:
            ax.step(d[xcol], d[ycol], where="post",
                    color=st.colors[i % len(st.colors)], lw=1.2, label=lab)
    if kind == "roc":
        ax.plot([0, 1], [0, 1], color=_GRAY, ls=":", lw=1, label="Random")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        side_legend(ax, title="AUC")
    else:
        ax.axhline(prevalence, color=_GRAY, ls=":", lw=1, label="Random")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        side_legend(ax, title="AP")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None if kind == "pr" else 1.01)
    ax.set_title(f"{feature_labels.get(feature, feature)}: held-out "
                 f"{'ROC' if kind == 'roc' else 'precision-recall'}", fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# Dataset comparison (estimate + CI)
# ---------------------------------------------------------------------------

def plot_compare_metric(feature, datasets, summ, metric, metric_label, st,
                        dataset_labels, feature_labels):
    fig, ax = st.new_fig("half_page", legend_in=1.1)
    x = np.arange(len(datasets))
    est, lo, hi = [], [], []
    for ds in datasets:
        s = summ.loc[f"{ds}__{feature}"]
        est.append(s[metric])
        lo.append(s[f"{metric}_lo"])
        hi.append(s[f"{metric}_hi"])
    est, lo, hi = map(np.array, (est, lo, hi))
    if metric in summ.columns and not pd.isna(summ.loc["rsa_only", metric]):
        r = summ.loc["rsa_only"]
        ax.axhspan(r[f"{metric}_lo"], r[f"{metric}_hi"], color=st.colors[1], alpha=0.15, lw=0)
        ax.axhline(r[metric], color=st.colors[1], ls="--", lw=1, label="RSA only")
    ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt="o", color=st.colors[0],
                capsize=3, label=feature_labels.get(feature, feature))
    if metric.startswith("auc"):
        ax.axhline(0.5, color=_GRAY, ls=":", lw=1, label="Random")
    ax.set_xticks(x)
    ax.set_xticklabels([dataset_labels.get(d, d) for d in datasets], rotation=30, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{feature_labels.get(feature, feature)}: {metric_label} by network "
                 f"(95% CI)", fontsize=9)
    side_legend(ax)
    return fig


# ---------------------------------------------------------------------------
# Heatmap of the MCC surface
# ---------------------------------------------------------------------------

def plot_heatmap(model, grid_dir, combos_file, sel, st, title, cmap):
    z = np.load(grid_dir / f"confusion_{model}.npz")
    combos = pd.read_csv(grid_dir / combos_file)
    med = np.median(mcc_from_counts(z["tp"], z["fp"], z["fn"], z["n_res"]), axis=0)
    combos = combos.assign(median_mcc=med)
    top = sel[sel["model"] == model].sort_values("n_folds", ascending=False).iloc[0]
    if model != "rsa_only":
        combos = combos[np.isclose(combos["centrality_lower"], top["centrality_lower"])]
    piv = combos.pivot(index="rsa_lower", columns="rsa_upper", values="median_mcc")

    fig, ax = plt.subplots(figsize=st.size("single_sq"))
    im = ax.imshow(piv.to_numpy(), origin="lower", aspect="auto", cmap=cmap)
    ax.grid(False)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in piv.columns], rotation=90)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{v:.3f}" for v in piv.index])
    ax.set_xlabel("RSA upper bound")
    ax.set_ylabel("RSA lower bound")
    yi = int(np.argmin(np.abs(piv.index.to_numpy() - top["rsa_lower"])))
    xi = int(np.argmin(np.abs(piv.columns.to_numpy() - top["rsa_upper"])))
    ax.add_patch(Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False, ec="red", lw=1.5))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Median MCC (all proteins)")
    sub = "" if model == "rsa_only" else f"\ncentrality ≥ {top['centrality_lower']:g}"
    ax.set_title(f"{title}{sub}", fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# Partner types, random null, stability
# ---------------------------------------------------------------------------

def plot_partner(model, partner, role_set, min_residues, st, title, partner_labels):
    d = partner[(partner["model"] == model) & (partner["role_set"] == role_set) &
                (partner["n_residues"] >= min_residues)]
    d = d.sort_values("pooled_sensitivity", ascending=False)
    fig, ax = st.new_fig("half_page", legend_in=1.7)
    if d.empty:
        ax.text(0.5, 0.5, "No partner type with enough residues", ha="center",
                transform=ax.transAxes)
        return fig
    x = np.arange(len(d))
    s = d["pooled_sensitivity"].to_numpy()
    ax.bar(x, s, color=st.colors[0], alpha=0.85, label="Observed")
    ax.errorbar(x, s, yerr=[s - d["pooled_sensitivity_lo"], d["pooled_sensitivity_hi"] - s],
                fmt="none", ecolor=_DARK, capsize=3, lw=1)
    ax.plot(x, d["random_a_mean"], "_", color=_DARK, ms=14, mew=2, zorder=4,
            label="Random (whole protein)")
    if "random_b_mean" in d and d["random_b_mean"].notna().any():
        ax.plot(x, d["random_b_mean"], "_", color=st.colors[1], ms=14, mew=2, zorder=4,
                label="Random (inside RSA window)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{partner_labels.get(t, t)}\n(n={n})"
                        for t, n in zip(d["partner_type"], d["n_residues"])],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Sensitivity (pooled)")
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}: sensitivity by partner type ({role_set})", fontsize=9)
    side_legend(ax)
    return fig


def plot_null(model, null, summ, st, title):
    fig, ax = st.new_fig("half_page", legend_in=1.7)
    for v, col, lab in (("a", _GRAY, "Random (whole protein)"),
                        ("b", st.colors[1], "Random (inside RSA window)")):
        d = null[(null["model"] == model) & (null["variant"] == v)]["median_mcc"]
        if len(d):
            ax.hist(d, bins=30, color=col, alpha=0.6, label=lab)
    obs = summ.loc[model, "median_mcc"]
    ax.axvline(obs, color=_DARK, lw=1.5, label=f"Observed ({obs:.2f})")
    ax.set_xlabel("Median MCC over held-out proteins")
    ax.set_ylabel("Random replicates")
    ax.set_title(f"{title}: observed vs random classifier", fontsize=9)
    side_legend(ax)
    return fig


def plot_stability(model, boot_sel, top_n, st, title):
    d = boot_sel[boot_sel["model"] == model].sort_values("fraction", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=st.size("half_page"))
    labels = []
    for _, r in d.iterrows():
        c = "" if pd.isna(r["centrality_lower"]) else f"c≥{r['centrality_lower']:g}, "
        labels.append(f"{c}RSA {r['rsa_lower']:g}-{r['rsa_upper']:g}")
    y = np.arange(len(d))[::-1]
    ax.barh(y, d["fraction"], color=st.colors[0])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Fraction of bootstrap resamples selecting it")
    ax.set_xlim(0, 1)
    ax.set_title(f"{title}: threshold stability (Bootstrap A)", fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(out_root, grid_name, datasets, features, dataset_labels, feature_labels,
         partner_labels, partner_role_set, partner_min_residues, compare_metrics,
         stability_top_n, make, style_dir, style_module, fig_format, dpi):
    out_root = Path(out_root)
    grid_dir = out_root / "stage1" / "grids" / grid_name
    s2 = out_root / "stage2" / grid_name
    fig_dir = out_root / "stage3" / grid_name
    st = Style(style_dir, style_module, fig_format, dpi)
    cmap = getattr(st.mod, "CMAPS", {}).get("blues", "viridis") if st.mod else "viridis"

    summ = pd.read_csv(s2 / "summary_models.csv").set_index("model")
    ops = pd.read_csv(s2 / "operating_points.csv").set_index("model")
    sel = pd.read_csv(s2 / "loo_selection_counts.csv")
    roc = pd.read_csv(s2 / "curves_roc.csv") if (s2 / "curves_roc.csv").exists() else None
    pr = pd.read_csv(s2 / "curves_pr.csv") if (s2 / "curves_pr.csv").exists() else None
    partner = pd.read_csv(s2 / "partner_sensitivity.csv")
    null = pd.read_csv(s2 / "random_null.csv")
    boot_sel = pd.read_csv(s2 / "bootstrapA_selection.csv")
    rsa_point = ops.loc["rsa_only"] if "rsa_only" in ops.index else None
    prevalence = float(ops["prevalence"].iloc[0])

    cent_models = [f"{ds}__{f}" for f in features for ds in datasets]
    all_models = cent_models + ["rsa_only"]
    n = 0

    def lab(m):
        return label_of(m, dataset_labels, feature_labels)

    for m in cent_models:
        if make.get("roc") and roc is not None:
            st.save(plot_roc(m, roc, ops, summ, st, lab(m), rsa_point), fig_dir / "roc" / f"roc_{m}")
            n += 1
        if make.get("pr") and pr is not None:
            st.save(plot_pr(m, pr, ops, summ, st, lab(m), rsa_point), fig_dir / "pr" / f"pr_{m}")
            n += 1

    for f in features:
        if make.get("roc_compare") and roc is not None:
            st.save(plot_compare_curves(f, datasets, roc, summ, st, "roc", prevalence,
                                        dataset_labels, feature_labels),
                    fig_dir / "roc_compare" / f"roc_{f}")
            n += 1
        if make.get("pr_compare") and pr is not None:
            st.save(plot_compare_curves(f, datasets, pr, summ, st, "pr", prevalence,
                                        dataset_labels, feature_labels),
                    fig_dir / "pr_compare" / f"pr_{f}")
            n += 1
        if make.get("compare"):
            for metric, mlabel in compare_metrics.items():
                if metric not in summ.columns:
                    continue
                st.save(plot_compare_metric(f, datasets, summ, metric, mlabel, st,
                                            dataset_labels, feature_labels),
                        fig_dir / "compare" / f"{metric}_{f}")
                n += 1

    for m in all_models:
        combos_file = ("combos_rsa_only.csv" if m == "rsa_only"
                       else f"combos_centrality_{m.split('__', 1)[1]}.csv")
        if make.get("heatmap"):
            st.save(plot_heatmap(m, grid_dir, combos_file, sel, st, lab(m), cmap),
                    fig_dir / "heatmap" / f"mcc_{m}")
            n += 1
        if make.get("partner"):
            st.save(plot_partner(m, partner, partner_role_set, partner_min_residues,
                                 st, lab(m), partner_labels),
                    fig_dir / "partner" / f"partner_{m}")
            n += 1
        if make.get("null"):
            st.save(plot_null(m, null, summ, st, lab(m)), fig_dir / "null" / f"null_{m}")
            n += 1
        if make.get("stability"):
            st.save(plot_stability(m, boot_sel, stability_top_n, st, lab(m)),
                    fig_dir / "stability" / f"stability_{m}")
            n += 1

    print(f"Stage 3 complete: {n} figures in {fig_dir}")


if __name__ == "__main__":

    # ── Edit these values ─────────────────────────────────────────────────────
    # Same OUT_ROOT as Stages 1 and 2
    OUT_ROOT = Path("...0_4_revisions/2_experiments/8_jackknife")
    GRID_NAME = "percentile" ######### EDIT AS NEEDED
    DATASETS = ["arp_iac", "arp_no_iac", "ring_iac", "ring_no_iac", "pcn5", "pcn6", "pcn7"]
    FEATURES = ["StdDegree", "StdCloseness", "StdBetweenness", "StdPageRank"]

    # Labels used in titles, legends and tick labels
    DATASET_LABELS = {
        "arp_iac": "Arpeggio (IAC)", "arp_no_iac": "Arpeggio (no IAC)",
        "ring_iac": "RING (IAC)", "ring_no_iac": "RING (no IAC)",
        "pcn5": "PCN 5 Å", "pcn6": "PCN 6 Å", "pcn7": "PCN 7 Å",
    }
    FEATURE_LABELS = {
        "StdDegree": "Degree", "StdCloseness": "Closeness",
        "StdBetweenness": "Betweenness", "StdPageRank": "PageRank",
    }
    PARTNER_LABELS = {
        "metal_ion": "Metal ion", "nonmetal_ion": "Non-metal ion",
        "small_molecule": "Small molecule", "cofactor": "Cofactor",
        "saccharide": "Saccharide", "peptide": "Peptide", "DNA": "DNA",
        "RNA": "RNA", "unknown": "Unknown",
    }
    PARTNER_ROLE_SET = "binding+dual"   # or "binding"
    PARTNER_MIN_RESIDUES = 10           # hide partner types with fewer residues

    # Dataset-comparison figures: column in summary_models.csv -> axis label
    COMPARE_METRICS = {
        "median_mcc": "Median MCC",
        "auc_windowed": "ROC AUC",
        "ap_windowed": "Average precision",
        "pooled_sensitivity": "Sensitivity (pooled)",
    }
    STABILITY_TOP_N = 10

    # Which figure families to make
    MAKE = {"roc": True, "pr": True, "roc_compare": True, "pr_compare": True,
            "compare": True, "heatmap": True, "partner": True, "null": True,
            "stability": True}

    # Style: folder containing figure_style_ver2.py (None = this folder / sys.path)
    STYLE_DIR = Path(__file__).parent.parent
    STYLE_MODULE = "figure_style_ver2"
    FIG_FORMAT = "png"                 # PLOS accepts tiff or eps; "png" for quick looks
    DPI = 300
    # ──────────────────────────────────────────────────────────────────────────

    main(
        out_root=OUT_ROOT,
        grid_name=GRID_NAME,
        datasets=DATASETS,
        features=FEATURES,
        dataset_labels=DATASET_LABELS,
        feature_labels=FEATURE_LABELS,
        partner_labels=PARTNER_LABELS,
        partner_role_set=PARTNER_ROLE_SET,
        partner_min_residues=PARTNER_MIN_RESIDUES,
        compare_metrics=COMPARE_METRICS,
        stability_top_n=STABILITY_TOP_N,
        make=MAKE,
        style_dir=STYLE_DIR,
        style_module=STYLE_MODULE,
        fig_format=FIG_FORMAT,
        dpi=DPI,
    )
