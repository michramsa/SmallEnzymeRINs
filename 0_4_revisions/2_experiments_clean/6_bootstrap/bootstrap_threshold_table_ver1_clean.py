"""
bootstrap_threshold_table_ver1.py
=================================
Summarise Bootstrap A (threshold stability) from amitai_jackknife_stage2_ver1.py
as one table instead of one bar chart per model.

Inputs (from <STAGE2_DIR>, written by Stage 2):
    bootstrapA_thresholds.csv   n_distinct_selected, top_combo_fraction
    bootstrapA_selection.csv    how often each threshold combination was selected

One row per model (<dataset>__<feature>, or rsa_only), with columns:
    Dataset | Centrality | Most-selected combination |
    Selection frequency | Distinct combinations

Outputs (under OUT_DIR):
    bootstrapA_threshold_table.csv    numeric values (for checking / other tools)
    bootstrapA_threshold_table.tex    LaTeX tabular, ready to paste (no extra packages)

All settings live in the `if __name__ == "__main__":` block at the bottom.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_model(model: str):
    """'arp_iac__StdDegree' -> ('arp_iac', 'StdDegree'); 'rsa_only' -> ('--', 'RSA only')."""
    if model == "rsa_only":
        return "--", "RSA only"
    dataset, _, feature = model.partition("__")
    return dataset, feature


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def combo_text(row, decimals, latex):
    """Label for one combination: 'c>=X, RSA a-b' (centrality part dropped if NaN)."""
    parts = []
    if not pd.isna(row["centrality_lower"]):
        c = f"{row['centrality_lower']:.{decimals}f}"
        parts.append(f"$c \\geq {c}$" if latex else f"c>={c}")
    a, b = f"{row['rsa_lower']:.{decimals}f}", f"{row['rsa_upper']:.{decimals}f}"
    parts.append(f"RSA {a}--{b}" if latex else f"RSA {a}-{b}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def build_table(thresholds: pd.DataFrame, selection: pd.DataFrame,
                datasets, features, decimals: int, latex: bool) -> pd.DataFrame:
    """One formatted row per model, ordered by feature then dataset (RSA_only last)."""
    # Most-selected combination per model (ties -> first row after sorting by fraction).
    top = (selection.sort_values(["model", "fraction"], ascending=[True, False])
           .groupby("model", sort=False).head(1).set_index("model"))

    rows = []
    for _, r in thresholds.iterrows():
        dataset, feature = split_model(r["model"])
        rows.append({
            "model": r["model"],
            "Dataset": dataset,
            "Centrality": feature,
            "Most-selected combination": combo_text(top.loc[r["model"]], decimals, latex),
            "Selection frequency": f"{r['top_combo_fraction']:.2f}",
            "Distinct combinations": int(r["n_distinct_selected"]),
        })
    df = pd.DataFrame(rows)

    # Order: features in the requested order, datasets in the requested order, RSA_only last.
    feat_rank = {f: i for i, f in enumerate(features)}
    data_rank = {d: i for i, d in enumerate(datasets)}
    df["_f"] = df["Centrality"].map(lambda f: feat_rank.get(f, len(feat_rank)))
    df["_d"] = df["Dataset"].map(lambda d: data_rank.get(d, len(data_rank)))
    df = df.sort_values(["_f", "_d"]).drop(columns=["_f", "_d", "model"])
    return df.reset_index(drop=True)


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Plain tabular with a header rule and a thin rule between centrality groups."""
    cols = list(df.columns)
    align = "lll" + "rr"
    lines = [r"\begin{table}[ht]", r"\centering", r"\small",
             f"\\begin{{tabular}}{{{align}}}", r"\hline",
             " & ".join(f"\\textbf{{{c}}}" for c in cols) + r" \\", r"\hline"]
    prev = None
    for _, r in df.iterrows():
        if prev is not None and r["Centrality"] != prev:
            lines.append(r"\hline")
        prev = r["Centrality"]
        cells = [tex_escape(str(r[c])) if c != "Most-selected combination" else str(r[c])
                 for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}",
              f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(stage2_dir, out_dir, datasets, features, decimals, caption, label):
    stage2_dir, out_dir = Path(stage2_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = pd.read_csv(stage2_dir / "bootstrapA_thresholds.csv")
    selection = pd.read_csv(stage2_dir / "bootstrapA_selection.csv")

    plain = build_table(thresholds, selection, datasets, features, decimals, latex=False)
    plain.to_csv(out_dir / "bootstrapA_threshold_table.csv", index=False)

    tex_df = build_table(thresholds, selection, datasets, features, decimals, latex=True)
    (out_dir / "bootstrapA_threshold_table.tex").write_text(
        to_latex(tex_df, caption, label))

    print(f"Wrote {len(plain)} rows to {out_dir}")
    return plain


if __name__ == "__main__":

    # ── Edit these values ─────────────────────────────────────────────────────
    # Folder holding bootstrapA_thresholds.csv / bootstrapA_selection.csv
    # (= <OUT_ROOT>/stage2/<GRID_NAME> from Stage 2)
    STAGE2_DIR = Path("...0_4_revisions/2_experiments/8_jackknife/stage2/...")
    OUT_DIR = STAGE2_DIR / "tables"

    # Row order in the table (models not listed are placed last)
    DATASETS = ["arp_iac", "arp_no_iac", "ring_iac", "ring_no_iac", "pcn5", "pcn6", "pcn7"]
    FEATURES = ["StdDegree", "StdCloseness", "StdBetweenness", "StdPageRank"]

    DECIMALS = 2                    # decimals shown for thresholds
    CAPTION = ("Stability of the selected thresholds across 1,000 protein-level "
               "bootstrap resamples. Selection frequency is the fraction of resamples "
               "that selected the most-selected combination; distinct combinations is "
               "the number of different combinations selected across resamples.")
    LABEL = "tab:threshold_stability"
    # ──────────────────────────────────────────────────────────────────────────

    main(
        stage2_dir=STAGE2_DIR,
        out_dir=OUT_DIR,
        datasets=DATASETS,
        features=FEATURES,
        decimals=DECIMALS,
        caption=CAPTION,
        label=LABEL,
    )
