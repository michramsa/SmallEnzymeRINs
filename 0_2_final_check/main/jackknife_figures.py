"""
0_5_loo_figures.py
------------------
Post-processing and figure generation for jackknife_experiment.py outputs.

Reads the per-protein LOO results CSVs and produces one publication-ready
figure (PLOS Computational Biology format):

    Figure 1 — Prediction class breakdown (full / partial / none)
               Stacked vertical bar chart, one bar per metric.
               Mirrors Amitai et al. (2004) Figure 3A for direct comparison.

Classification rule (applied to per-protein tp / fn columns):
    full    : fn == 0              (sensitivity == 1.0; all functional residues predicted)
    partial : tp > 0 and fn > 0   (0 < sensitivity < 1.0)
    none    : tp == 0             (sensitivity == 0.0; no functional residues predicted)

Outputs (written to output_dir):
    fig1_prediction_class_breakdown.png
    prediction_class_summary.csv   <- combined table with classification column
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from imports import DATA_DIR_3
from figure_style_ver2 import (
    apply_style, save_fig,
    FIG_SIZE, FONT, LAYOUT, COLORS,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FUNCTIONAL_TYPES = ["catalytic", "binding", "dual"]

# Human-readable labels for metrics
METRIC_LABELS = {
    "closeness":   "Std. CC",
    "betweenness": "Std. BC",
    "rca":         "RCA z-score",
    "pagerank":    "Std. PR",
}

# Prediction class display order and colours
CLASS_ORDER  = ["full", "partial", "none"]
CLASS_LABELS = {"full": "Fully Predicted", "partial": "Partial Prediction", "none": "None"}
CLASS_COLORS = {
    "full":    "#8E8278",
    "partial": "#E8E4DF",
    "none":    "#35302B",
}


# ---------------------------------------------------------------------------
# Data loading and classification
# ---------------------------------------------------------------------------

def load_loo_results(
    input_dir: Path,
    metrics: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Load LOO results CSVs for each metric from input_dir.

    Expected filename pattern (written by jackknife_experiment.py):
        jackknife_results_{metric}.csv

    Returns a dict {metric_name: DataFrame} for each file found.
    Metrics whose CSV is missing are skipped with a warning.
    """
    results = {}
    for metric in metrics:
        path = input_dir / f"jackknife_results_{metric}.csv"
        if not path.exists():
            print(f"  [WARNING] Results file not found, skipping: {path}")
            continue
        df = pd.read_csv(path)
        for col in ["tp", "fn", "fp", "tn", "sensitivity", "specificity", "mcc"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for ftype in FUNCTIONAL_TYPES:
            for stat in ["tp", "tn", "fp", "fn", "mcc"]:
                col = f"{stat}_{ftype}"
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        results[metric] = df
        print(f"  Loaded [{metric}]: {len(df)} proteins from {path.name}")
    return results


def classify_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'prediction_class' column to a LOO results DataFrame.

    Classification uses the per-protein tp and fn counts:
        full    : fn == 0              (all functional residues predicted)
        partial : tp > 0 and fn > 0   (some predicted, some missed)
        none    : tp == 0             (no functional residues predicted)
    """
    df = df.copy()
    conditions = [
        (df["fn"] == 0),
        (df["tp"] > 0) & (df["fn"] > 0),
        (df["tp"] == 0),
    ]
    choices = ["full", "partial", "none"]
    df["prediction_class"] = np.select(conditions, choices, default="unknown")
    return df


def build_summary(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all metric DataFrames into a single summary CSV with
    a 'metric' column and a 'prediction_class' column.
    """
    frames = []
    for metric, df in results.items():
        df_classified = classify_predictions(df)
        df_classified["metric"] = metric
        frames.append(df_classified)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Figure 1 — Prediction class breakdown
# ---------------------------------------------------------------------------

def fig1_prediction_class_breakdown(
    results: dict[str, pd.DataFrame],
    output_dir: Path,
    filename: str,
):
    """
    Stacked vertical bar chart showing the number of proteins classified
    as full / partial / none for each metric.

    Bars are annotated with percentage labels inside each segment
    (omitted if segment is too narrow to fit text).
    The total n is reported once in the legend title.

    Saves to output_dir / filename.
    """
    apply_style()

    metrics = list(results.keys())
    if not metrics:
        print("  [fig1] No results to plot.")
        return

    # Build counts per metric
    counts = {}
    for metric, df in results.items():
        df_c  = classify_predictions(df)
        total = len(df_c)
        counts[metric] = {
            cls: df_c[df_c["prediction_class"] == cls].shape[0]
            for cls in CLASS_ORDER
        }
        counts[metric]["total"] = total

    fig, ax = plt.subplots(figsize=FIG_SIZE["half_page"])

    x_positions = np.arange(len(metrics))
    bar_width   = 0.55

    # Per-segment text colour: white on dark "none" segment, dark on light segments
    SEGMENT_TEXT_COLOR = {
        "full":    COLORS["dark_gray"],
        "partial": COLORS["dark_gray"],
        "none":    COLORS["white"],
    }

    for i, metric in enumerate(metrics):
        bottom = 0
        total  = counts[metric]["total"]
        for cls in CLASS_ORDER:
            n   = counts[metric][cls]
            pct = (n / total * 100) if total > 0 else 0.0
            ax.bar(
                x_positions[i],
                n,
                bottom    = bottom,
                width     = bar_width,
                color     = CLASS_COLORS[cls],
                edgecolor = COLORS["white"],
                linewidth = 0.5,
            )
            # Annotate with percentage if segment is tall enough
            if pct >= 8:
                ax.text(
                    x_positions[i],
                    bottom + n / 2,
                    f"{pct:.0f}%",
                    ha         = "center",
                    va         = "center",
                    fontsize   = FONT["size_annot"],
                    color      = SEGMENT_TEXT_COLOR[cls],
                    fontweight = "bold",
                )
            bottom += n

    # Axes formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [METRIC_LABELS.get(m, m) for m in metrics],
        fontsize = FONT["size_tick"],
    )
    ax.set_ylabel("Number of proteins", fontsize=FONT["size_label"])
    ax.set_ylim(0, max(counts[m]["total"] for m in metrics) * 1.05)
    ax.grid(False)

    # n reported once in legend title; legend placed below the axes
    n_total      = counts[metrics[0]]["total"]
    legend_title = f"n = {n_total} test set enzymes"

    handles = [
        mpatches.Patch(
            facecolor = CLASS_COLORS[cls],
            edgecolor = COLORS["mid_gray"],
            linewidth = 0.4,
            label     = CLASS_LABELS[cls],
        )
        for cls in CLASS_ORDER
    ]
    ax.legend(
        handles        = handles,
        title          = legend_title,
        fontsize       = FONT["size_legend"],
        title_fontsize = FONT["size_legend"],
        framealpha     = LAYOUT["legend_framealpha"],
        loc            = "upper center",
        bbox_to_anchor = (0.5, -0.18),
        ncol           = len(CLASS_ORDER),
    )

    fig.tight_layout(pad=LAYOUT["tight_pad"])

    out_path = output_dir / filename
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"  [fig1] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def save_summary_csv(
    results: dict[str, pd.DataFrame],
    output_dir: Path,
    filename: str,
):
    """
    Write a combined CSV containing all per-protein LOO results
    across all metrics, with a 'metric' column and a 'prediction_class'
    column added.

    Saves to output_dir / filename.
    """
    summary = build_summary(results)
    if summary.empty:
        print("  [csv] No data to write.")
        return
    out_path = output_dir / filename
    summary.to_csv(out_path, index=False)
    print(f"  [csv] Saved: {out_path}")

    # Print a compact breakdown table
    print("\n  Prediction class counts:")
    print(f"  {'Metric':<20} {'Full':>6} {'Partial':>8} {'None':>6} {'Total':>7}")
    print("  " + "-" * 50)
    for metric, df in results.items():
        df_c  = classify_predictions(df)
        full  = (df_c["prediction_class"] == "full").sum()
        part  = (df_c["prediction_class"] == "partial").sum()
        none  = (df_c["prediction_class"] == "none").sum()
        total = len(df_c)
        print(
            f"  {METRIC_LABELS.get(metric, metric):<20} "
            f"{full:>6} ({full/total*100:4.1f}%)  "
            f"{part:>6} ({part/total*100:4.1f}%)  "
            f"{none:>6} ({none/total*100:4.1f}%)  "
            f"{total:>6}"
        )




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    input_dir,
    output_dir,
    metrics,
    fig1_filename,
    summary_filename,
):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("0_5_loo_figures.py")
    print("=" * 60)
    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print(f"Metrics: {metrics}")

    # -- Load results -------------------------------------------------------
    print("\nLoading LOO results...")
    results = load_loo_results(input_dir, metrics)

    if not results:
        print("No results loaded — check input_dir and metrics list. Exiting.")
        return

    # -- Summary CSV --------------------------------------------------------
    print("\nWriting summary CSV...")
    save_summary_csv(results, output_dir, summary_filename)

    # -- Figure 1 -----------------------------------------------------------
    print("\nGenerating figure...")
    fig1_prediction_class_breakdown(results, output_dir, fig1_filename)

    print("\nDone.")
    print("=" * 60)


# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -- Input / output paths -----------------------------------------------
    # input_dir must match the output_dir jackknife_experiment.py was actually
    # run with -- confirmed as DATA_DIR_3 / "test" / "jackknife_experiment".
    input_dir  = DATA_DIR_3 / "test" / "jackknife_experiment"
    output_dir = DATA_DIR_3 / "test" / "jackknife_experiment" / "figures"

    # -- Metrics to process -------------------------------------------------
    # Include only metrics whose jackknife_results_{metric}.csv exists in input_dir
    metrics = ["closeness", "betweenness", "rca", "pagerank"]

    # -- Output filenames ---------------------------------------------------
    fig1_filename    = "fig1_prediction_class_breakdown.png"
    summary_filename = "prediction_class_summary.csv"

    # -----------------------------------------------------------------------

    main(
        input_dir        = input_dir,
        output_dir       = output_dir,
        metrics          = metrics,
        fig1_filename    = fig1_filename,
        summary_filename = summary_filename,
    )
