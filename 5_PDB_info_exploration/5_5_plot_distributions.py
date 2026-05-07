# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

import matplotlib.ticker as ticker

PALETTE = {
    "bg":          "#0F1117",
    "panel":       "#1A1D27",
    "grid":        "#2A2D3A",
    "text":        "#E8EAF0",
    "subtext":     "#8B90A8",
    "accent1":     "#4FC3F7",   # light blue  – method bars
    "accent2":     "#81C784",   # soft green  – resolution bars
    "accent_edge": "#FFFFFF",
}

FONT_TITLE  = {"fontsize": 13, "fontweight": "bold",  "color": PALETTE["text"],    "fontfamily": "monospace"}
FONT_LABEL  = {"fontsize": 10, "fontweight": "normal","color": PALETTE["subtext"], "fontfamily": "monospace"}
FONT_TICK   = {"labelsize": 9,  "colors": PALETTE["subtext"]}
FONT_COUNT  = {"fontsize": 8.5,"fontweight": "bold",  "color": PALETTE["text"],    "fontfamily": "monospace"}


def style_axes(ax, ylabel="Count"):
    """Apply shared dark-theme styling to an axes."""
    ax.set_facecolor(PALETTE["panel"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(PALETTE["grid"])
    ax.tick_params(axis="both", **FONT_TICK)
    ax.set_ylabel(ylabel, **FONT_LABEL)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)


def add_count_labels(ax, bars, counts):
    """Place count labels centred above each bar."""
    for bar, count in zip(bars, counts):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.text(x, y + 0.3, str(count), ha="center", va="bottom", **FONT_COUNT)

def plot_methods(ax, df):
    method_counts = (
        df["experimental_method"]
        .fillna("UNKNOWN")
        .str.strip()
        .value_counts()
        .sort_values(ascending=False)
    )

    x = np.arange(len(method_counts))
    bars = ax.bar(
        x,
        method_counts.values,
        color=PALETTE["accent1"],
        edgecolor=PALETTE["bg"],
        linewidth=0.8,
        width=0.6,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        method_counts.index,
        rotation=30,
        ha="right",
        fontsize=8.5,
        color=PALETTE["subtext"],
        fontfamily="monospace",
    )
    ax.set_title("Experimental Method Distribution", **FONT_TITLE, pad=12)
    ax.set_xlabel("Method", **FONT_LABEL)
    style_axes(ax)
    add_count_labels(ax, bars, method_counts.values)

    # Annotate total entries
    ax.text(
        0.98, 0.97,
        f"n = {len(df)}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8, color=PALETTE["subtext"], fontfamily="monospace",
    )

def plot_resolution(ax, df):
    res = pd.to_numeric(df["resolution"], errors="coerce").dropna()

    if res.empty:
        ax.text(0.5, 0.5, "No resolution data available",
                ha="center", va="center", transform=ax.transAxes,
                color=PALETTE["subtext"], fontfamily="monospace")
        ax.set_title("Resolution Distribution", **FONT_TITLE, pad=12)
        style_axes(ax)
        return

    # Build bins: <1.0, then 0.2-Å steps up to max, then an open ">last" bin
    step   = 0.2
    lo     = 1.0
    hi_raw = np.ceil(res.max() / step) * step  # round up to nearest 0.2
    edges  = np.arange(lo, hi_raw + step, step)   # [1.0, 1.2, 1.4, …]

    # Count sub-1.0 separately
    below_1 = (res < lo).sum()

    # Bin the rest
    labels = []
    counts = []

    if below_1 > 0:
        labels.append("< 1.0")
        counts.append(int(below_1))

    for i in range(len(edges) - 1):
        lo_b, hi_b = edges[i], edges[i + 1]
        mask = (res >= lo_b) & (res < hi_b)
        cnt  = mask.sum()
        labels.append(f"{lo_b:.1f}–{hi_b:.1f}")
        counts.append(int(cnt))

    # Drop trailing empty bins
    while counts and counts[-1] == 0:
        counts.pop()
        labels.pop()

    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        counts,
        color=PALETTE["accent2"],
        edgecolor=PALETTE["bg"],
        linewidth=0.8,
        width=0.7,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=40,
        ha="right",
        fontsize=8.5,
        color=PALETTE["subtext"],
        fontfamily="monospace",
    )
    ax.set_title("Resolution Distribution (Å)", **FONT_TITLE, pad=12)
    ax.set_xlabel("Resolution bin (Å)", **FONT_LABEL)
    style_axes(ax)
    add_count_labels(ax, bars, counts)

    # Annotate stats
    stats = (
        f"n = {len(res)}\n"
        f"mean = {res.mean():.2f} Å\n"
        f"median = {res.median():.2f} Å"
    )
    ax.text(
        0.98, 0.97, stats,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=7.5, color=PALETTE["subtext"], fontfamily="monospace",
        linespacing=1.5,
    )


def main(csv_path, output_dir):
    csv_path   = Path(csv_path)
    output_dir = Path(output_dir)

    if not csv_path.exists():
        sys.exit(f"Error: CSV file not found: {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")
    print(f"Columns: {list(df.columns)}\n")

    # Normalise column names: strip whitespace, lower-case
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {"experimental_method", "resolution"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(f"Error: CSV is missing required columns: {missing}\n"
                 f"Found columns: {list(df.columns)}")

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6), facecolor=PALETTE["bg"])
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.22, wspace=0.38)

    gs  = GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.set_facecolor(PALETTE["panel"])
    ax2.set_facecolor(PALETTE["panel"])

    plot_methods(ax1, df)
    plot_resolution(ax2, df)

    fig.suptitle(
        "PDB Dataset — Structural Characterisation Summary",
        fontsize=15, fontweight="bold",
        color=PALETTE["text"], fontfamily="monospace",
        y=0.97,
    )

    out_path = output_dir / "filtered_pdb_distributions.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"Figure saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":

    # pdb_distributions.png is the non-filtered for resolution
    # filtered_pdb_distributions.png is filtered for at most 3.0 angstroms for the resolutions, with NMR/solution nmr etc. included
    CSV_PATH   =  ENZYMES_DIR / "info_CSVs" / "filtered_consolidated.csv"  # Path to input CSV file
    OUTPUT_DIR = ENZYMES_DIR / "info_CSVs" / "figures"  # Directory where pdb_distributions.png will be saved

    main(csv_path=CSV_PATH, output_dir=OUTPUT_DIR)
