"""
amitai_jackknife_mcc_iqr_ver1.py
================================
Spread of held-out MCC across leave-one-out folds (one MCC per held-out
protein), for the Amitai (2004)-style jackknife. Reads Stage 2 output only;
fits nothing.

Input
-----
    <out_root>/stage2/<grid_name>/loo_per_protein.csv   (columns: model, mcc)
    <out_root>/stage2/<grid_name>/summary_models.csv    (consistency check only)

Outputs (under <out_root>/stage3/<grid_name>/iqr/, one figure per file)
-----------------------------------------------------------------------
    box_mcc_<feature>       box plot of per-protein MCC, one box per network,
                            RSA-only as a separate reference box.
                            Box = interquartile range (Q1-Q3), line = median,
                            whiskers = most extreme value within 1.5 x IQR,
                            dots = individual proteins (optional).
                            Optional reference median / IQR band from a
                            published study (e.g. Amitai et al. 2004).
    mcc_iqr_summary.csv     per model: n, median, Q1, Q3, IQR, min, max

Quartiles use numpy's default (linear) method, the same one matplotlib's
boxplot uses, so the table and the boxes agree exactly.

Needs amitai_jackknife_stage3_ver1.py (and therefore
amitai_jackknife_stage1_ver1.py) in the same folder, for the shared style.

All settings live in the `if __name__ == "__main__":` block at the bottom.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from amitai_jackknife_stage3_ver1 import Style, _GRAY, _DARK  # noqa: E402


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def mcc_values(loo: pd.DataFrame, model: str) -> np.ndarray:
    return loo.loc[loo["model"] == model, "mcc"].dropna().to_numpy(float)


def iqr_table(loo: pd.DataFrame, models: list) -> pd.DataFrame:
    rows = []
    for m in models:
        v = mcc_values(loo, m)
        if not len(v):
            continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        rows.append({"model": m, "n_proteins": len(v), "median_mcc": med,
                     "q1_mcc": q1, "q3_mcc": q3, "iqr_mcc": q3 - q1,
                     "min_mcc": v.min(), "max_mcc": v.max(),
                     "frac_mcc_le_0": float(np.mean(v <= 0))})
    return pd.DataFrame(rows)


def check_against_summary(tab: pd.DataFrame, summ_path: Path):
    """Medians here must equal Stage 2's median_mcc (same proteins, same values)."""
    if not summ_path.exists():
        return
    summ = pd.read_csv(summ_path).set_index("model")
    for _, r in tab.iterrows():
        if r["model"] in summ.index:
            s = summ.loc[r["model"], "median_mcc"]
            if not np.isclose(s, r["median_mcc"]):
                print(f"WARNING: {r['model']}: median {r['median_mcc']:.4f} here vs "
                      f"{s:.4f} in summary_models.csv")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_box(feature, datasets, loo, st, dataset_labels, feature_labels,
             show_rsa_only, show_points, point_alpha, reference, rng):
    fig, ax = st.new_fig("half_page", legend_in=1.7)
    c_main, c_rsa = st.colors[0], st.colors[1]

    data, labels, colors = [], [], []
    for ds in datasets:
        v = mcc_values(loo, f"{ds}__{feature}")
        if not len(v):
            print(f"  no rows for {ds}__{feature}; skipped")
            continue
        data.append(v)
        labels.append(dataset_labels.get(ds, ds))
        colors.append(c_main)
    pos = list(np.arange(1, len(data) + 1, dtype=float))

    if show_rsa_only:
        v = mcc_values(loo, "rsa_only")
        if len(v):
            data.append(v)
            labels.append("RSA only")
            colors.append(c_rsa)
            pos.append(pos[-1] + 1.5 if pos else 1.0)       # small gap before reference

    if not data:
        ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
        return fig

    # reference study (drawn first so it sits behind the boxes)
    if reference is not None:
        if reference.get("q1") is not None and reference.get("q3") is not None:
            ax.axhspan(reference["q1"], reference["q3"], color=_GRAY, alpha=0.15, lw=0)
        ax.axhline(reference["median"], color=_DARK, ls="--", lw=1)

    ax.axhline(0, color=_GRAY, ls=":", lw=1)

    if show_points:
        for x, v in zip(pos, data):
            jit = rng.uniform(-0.18, 0.18, size=len(v))
            ax.scatter(x + jit, v, s=3, color=_DARK, alpha=point_alpha, lw=0, zorder=3)

    bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True,
                    showfliers=not show_points,
                    medianprops={"color": _DARK, "lw": 1.5},
                    whiskerprops={"color": _DARK, "lw": 1},
                    capprops={"color": _DARK, "lw": 1},
                    flierprops={"marker": "o", "ms": 2, "mfc": _GRAY, "mec": "none"},
                    zorder=2)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.45)
        patch.set_edgecolor(_DARK)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_xlim(min(pos) - 0.7, max(pos) + 0.7)
    ax.set_ylabel("Held-out MCC per protein")
    ns = sorted({len(v) for v in data})
    n_txt = f"n = {ns[0]} proteins" if len(ns) == 1 else f"n = {ns[0]}-{ns[-1]} proteins"
    ax.set_title(f"{feature_labels.get(feature, feature)}: per-protein held-out MCC "
                 f"by network ({n_txt})", fontsize=9)

    handles = [Patch(facecolor=c_main, alpha=0.45, edgecolor=_DARK,
                     label=f"{feature_labels.get(feature, feature)} + RSA window")]
    if show_rsa_only and colors[-1] == c_rsa:
        handles.append(Patch(facecolor=c_rsa, alpha=0.45, edgecolor=_DARK, label="RSA only"))
    if reference is not None:
        lab = reference.get("label", "Reference")
        has_band = reference.get("q1") is not None and reference.get("q3") is not None
        handles.append(Line2D([], [], color=_DARK, ls="--", lw=1, label=f"{lab} (median)"))
        if has_band:
            handles.append(Patch(facecolor=_GRAY, alpha=0.15, label=f"{lab} (IQR)"))
    handles.append(Line2D([], [], color=_GRAY, ls=":", lw=1, label="MCC = 0 (random)"))
    # same placement as side_legend() in Stage 3, but with explicit handles
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0, frameon=False, fontsize=7)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(out_root, grid_name, datasets, features, dataset_labels, feature_labels,
         show_rsa_only, show_points, point_alpha, reference, seed,
         style_dir, style_module, fig_format, dpi):
    out_root = Path(out_root)
    s2 = out_root / "stage2" / grid_name
    out_dir = out_root / "stage3" / grid_name / "iqr"
    out_dir.mkdir(parents=True, exist_ok=True)
    st = Style(style_dir, style_module, fig_format, dpi)
    rng = np.random.default_rng(seed)

    loo = pd.read_csv(s2 / "loo_per_protein.csv", usecols=["model", "mcc"])

    models = [f"{ds}__{f}" for f in features for ds in datasets] + ["rsa_only"]
    tab = iqr_table(loo, models)
    check_against_summary(tab, s2 / "summary_models.csv")
    tab.to_csv(out_dir / "mcc_iqr_summary.csv", index=False, float_format="%.4f")

    n = 0
    for f in features:
        fig = plot_box(f, datasets, loo, st, dataset_labels, feature_labels,
                       show_rsa_only, show_points, point_alpha, reference, rng)
        st.save(fig, out_dir / f"box_mcc_{f}")
        n += 1

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"MCC IQR complete: {n} figures + mcc_iqr_summary.csv in {out_dir}")


if __name__ == "__main__":

    # ── Edit these values ─────────────────────────────────────────────────────
    # Same OUT_ROOT / GRID_NAME as Stages 1-3
    OUT_ROOT = Path("...0_4_revisions/2_experiments/8_jackknife")
    GRID_NAME = "amitai"
    DATASETS = ["arp_iac", "arp_no_iac", "ring_iac", "ring_no_iac", "pcn5", "pcn6", "pcn7"]
    FEATURES = ["StdDegree", "StdCloseness", "StdBetweenness", "StdPageRank"]

    DATASET_LABELS = {
        "arp_iac": "Arpeggio (IAC)", "arp_no_iac": "Arpeggio (no IAC)",
        "ring_iac": "RING (IAC)", "ring_no_iac": "RING (no IAC)",
        "pcn5": "PCN 5 Å", "pcn6": "PCN 6 Å", "pcn7": "PCN 7 Å",
    }
    FEATURE_LABELS = {
        "StdDegree": "Degree", "StdCloseness": "Closeness",
        "StdBetweenness": "Betweenness", "StdPageRank": "PageRank",
    }

    SHOW_RSA_ONLY = True        # add the RSA-only model as a reference box
    SHOW_POINTS = True          # one dot per protein (outlier markers hidden when on)
    POINT_ALPHA = 0.25          # dot transparency (lower if many proteins)
    SEED = 1                    # jitter of the dots only; does not affect numbers

    # Published reference drawn as a dashed median line + shaded IQR band.
    # None = not drawn. q1 / q3 may be None if only a median was reported.
    # Example: {"median": 0.30, "q1": 0.20, "q3": 0.40, "label": "Amitai et al. 2004"}
    REFERENCE = None

    STYLE_DIR = Path(__file__).parent.parent   # folder with figure_style_ver2.py
    STYLE_MODULE = "figure_style_ver2"
    FIG_FORMAT = "png"                         # "tiff" / "eps" for PLOS submission
    DPI = 300
    # ──────────────────────────────────────────────────────────────────────────

    main(
        out_root=OUT_ROOT,
        grid_name=GRID_NAME,
        datasets=DATASETS,
        features=FEATURES,
        dataset_labels=DATASET_LABELS,
        feature_labels=FEATURE_LABELS,
        show_rsa_only=SHOW_RSA_ONLY,
        show_points=SHOW_POINTS,
        point_alpha=POINT_ALPHA,
        reference=REFERENCE,
        seed=SEED,
        style_dir=STYLE_DIR,
        style_module=STYLE_MODULE,
        fig_format=FIG_FORMAT,
        dpi=DPI,
    )
