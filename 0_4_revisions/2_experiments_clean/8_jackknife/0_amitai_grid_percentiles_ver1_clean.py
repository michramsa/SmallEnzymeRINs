"""
amitai_grid_percentiles_ver1.py
===============================
Computes a label-free threshold grid for the Amitai-style jackknife and writes
it out ready to paste into the __main__ block of amitai_jackknife_stage1_ver1.py.

Why label-free
--------------
Every candidate threshold is a percentile of the feature values of ALL
canonical residues (functional labels are never read). The grid therefore
carries no information about which residues are functional, so one fixed grid
can be used for every jackknife fold.

What it computes
----------------
For each centrality (StdDegree, StdCloseness, ...):
    N_STEPS lower-bound thresholds, evenly spaced in percentile between
    CENT_PCT[0] and CENT_PCT[1], of the centrality values POOLED over all
    datasets (Stage 1 uses one grid per centrality, shared by every dataset).
For RSA-empirical:
    N_STEPS lower bounds between RSA_LOWER_PCT[0] and RSA_LOWER_PCT[1]
    N_STEPS upper bounds between RSA_UPPER_PCT[0] and RSA_UPPER_PCT[1]

Checks reported
---------------
    - duplicate values after rounding (ties, e.g. StdDegree is discrete);
      duplicates are removed and the reduced count is reported
    - for every dataset, the percentile that each pooled threshold falls at in
      that dataset alone (if the datasets differ a lot, a shared grid covers
      them unevenly)
    - every RSA lower bound must be below every RSA upper bound

Input
-----
<STAGE1_DIR>/residue_table.csv from amitai_jackknife_stage1_ver1.py
(any grid run is fine; the residue table does not depend on the grid).

Outputs (in OUT_DIR)
--------------------
    grid_for_stage1.txt          paste-ready GRID_* block for Stage 1
    grid_values.csv              every threshold: axis, feature, step,
                                 target percentile, value
    grid_dataset_percentiles.csv where each pooled centrality threshold falls
                                 in each dataset's own distribution
    grid_feature_summary.csv     per feature (and dataset): n, min, percentiles, max

All settings live in the `if __name__ == "__main__":` block at the bottom.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def thresholds(values: np.ndarray, pct_range: tuple, n_steps: int, decimals: int):
    """Percentile targets and the rounded, de-duplicated threshold values."""
    targets = np.linspace(pct_range[0], pct_range[1], n_steps)
    vals = np.round(np.nanpercentile(values, targets), decimals)
    keep = np.r_[True, np.diff(vals) > 0]          # drop ties after rounding
    return targets[keep], vals[keep], int((~keep).sum())


def percentile_of(values: np.ndarray, x: float) -> float:
    """Percent of values strictly below x (where the threshold cuts)."""
    v = values[~np.isnan(values)]
    return 100.0 * np.mean(v < x)


def fmt_list(vals, decimals):
    return "[" + ", ".join(f"{v:.{decimals}f}" for v in vals) + "]"


def main(stage1_dir, out_dir, datasets, features, cent_pct, rsa_lower_pct,
         rsa_upper_pct, n_steps, decimals):
    stage1_dir, out_dir = Path(stage1_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(stage1_dir / "residue_table.csv")
    print(f"Residue table: {res['pdb_id'].nunique()} proteins, {len(res):,} residues")

    rows, ds_rows, summ_rows, notes = [], [], [], []
    cent_lists = {}

    def summarise(name, dataset, v):
        v = v[~np.isnan(v)]
        r = {"feature": name, "dataset": dataset, "n": len(v), "min": v.min()}
        for p in (1, 5, 10, 25, 50, 75, 90, 95, 98, 99):
            r[f"p{p}"] = np.percentile(v, p)
        r["max"] = v.max()
        summ_rows.append(r)

    # ---- centralities (pooled over datasets) -------------------------------
    for f in features:
        cols = [f"{ds}__{f}" for ds in datasets]
        missing = [c for c in cols if c not in res.columns]
        if missing:
            raise KeyError(f"Columns not in residue table: {missing}")
        pooled = res[cols].to_numpy(float).ravel()
        t, v, n_dup = thresholds(pooled, cent_pct, n_steps, decimals)
        cent_lists[f] = v
        if n_dup:
            notes.append(f"{f}: {n_dup} duplicate value(s) after rounding removed "
                         f"-> {len(v)} thresholds")
        for k, (tt, vv) in enumerate(zip(t, v)):
            rows.append({"axis": "centrality_lower", "feature": f, "step": k,
                         "target_percentile": tt, "value": vv})
        summarise(f, "pooled", pooled)
        for ds in datasets:
            col = res[f"{ds}__{f}"].to_numpy(float)
            summarise(f, ds, col)
            for k, vv in enumerate(v):
                ds_rows.append({"feature": f, "dataset": ds, "step": k, "value": vv,
                                "target_percentile_pooled": t[k],
                                "percentile_in_dataset": percentile_of(col, vv)})

    # ---- RSA ---------------------------------------------------------------
    rsa = res["rsa_empirical"].to_numpy(float)
    summarise("rsa_empirical", "all", rsa)
    t_lo, v_lo, d_lo = thresholds(rsa, rsa_lower_pct, n_steps, decimals)
    t_hi, v_hi, d_hi = thresholds(rsa, rsa_upper_pct, n_steps, decimals)
    for name, t, v, d in (("rsa_lower", t_lo, v_lo, d_lo), ("rsa_upper", t_hi, v_hi, d_hi)):
        if d:
            notes.append(f"{name}: {d} duplicate value(s) after rounding removed")
        for k, (tt, vv) in enumerate(zip(t, v)):
            rows.append({"axis": name, "feature": "rsa_empirical", "step": k,
                         "target_percentile": tt, "value": vv})
    if v_lo.max() >= v_hi.min():
        notes.append(f"WARNING: highest RSA lower bound ({v_lo.max():g}) is not below "
                     f"the lowest RSA upper bound ({v_hi.min():g}); some combinations "
                     f"will be invalid. Narrow RSA_LOWER_PCT or raise RSA_UPPER_PCT.")

    # ---- dataset spread check ----------------------------------------------
    dsp = pd.DataFrame(ds_rows)
    for f in features:
        g = dsp[dsp["feature"] == f]
        spread = g.groupby("step")["percentile_in_dataset"].agg(lambda s: s.max() - s.min())
        if spread.max() > 10:
            notes.append(f"{f}: the same threshold sits up to {spread.max():.1f} "
                         f"percentile points apart across datasets "
                         f"(see grid_dataset_percentiles.csv)")

    # ---- write -------------------------------------------------------------
    pd.DataFrame(rows).to_csv(out_dir / "grid_values.csv", index=False)
    dsp.to_csv(out_dir / "grid_dataset_percentiles.csv", index=False)
    pd.DataFrame(summ_rows).to_csv(out_dir / "grid_feature_summary.csv", index=False)

    n_combo = {f: len(cent_lists[f]) * len(v_lo) * len(v_hi) for f in features}
    lines = [
        "    # Label-free percentile grid (amitai_grid_percentiles_ver1.py)",
        f"    #   centrality lower: p{cent_pct[0]:g}-p{cent_pct[1]:g} of all residues "
        f"(pooled over {len(datasets)} datasets), {n_steps} steps",
        f"    #   RSA lower: p{rsa_lower_pct[0]:g}-p{rsa_lower_pct[1]:g}, "
        f"RSA upper: p{rsa_upper_pct[0]:g}-p{rsa_upper_pct[1]:g} of all residues",
        "    #   combinations: " + ", ".join(f"{f} {n:,}" for f, n in n_combo.items()),
        '    GRID_NAME = "percentile"',
        "    GRID_CENTRALITY_LOWER = {",
        '        "default": (0.9, 2.1, 0.1),   # unused when every feature is listed below',
    ]
    for f in features:
        lines.append(f'        "{f}": {fmt_list(cent_lists[f], decimals)},')
    lines += ["    }",
              f"    GRID_RSA_LOWER = {fmt_list(v_lo, decimals)}",
              f"    GRID_RSA_UPPER = {fmt_list(v_hi, decimals)}"]
    block = "\n".join(lines)
    (out_dir / "grid_for_stage1.txt").write_text(block + "\n")

    print("\n── Paste into the __main__ block of amitai_jackknife_stage1_ver1.py ──")
    print(block)
    if notes:
        print("\n── Notes ──")
        for n in notes:
            print("  " + n)
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":

    # ── Edit these values ─────────────────────────────────────────────────────
    # Stage 1 output folder (holds residue_table.csv)
    STAGE1_DIR = Path("....0_4_revisions/2_experiments/8_jackknife/stage1")
    OUT_DIR = STAGE1_DIR / "grid_percentiles"

    DATASETS = ["arp_iac", "arp_no_iac", "ring_iac", "ring_no_iac", "pcn5", "pcn6", "pcn7"]
    FEATURES = ["StdDegree", "StdCloseness", "StdBetweenness", "StdPageRank"]

    # Percentile ranges (of ALL canonical residues; labels are not used)
    CENT_PCT = (50, 98)          # centrality lower bound
    RSA_LOWER_PCT = (1, 30)      # RSA lower bound
    RSA_UPPER_PCT = (40, 95)     # RSA upper bound
    N_STEPS = 13                 # values per threshold (Amitai used 13)
    DECIMALS = 3                 # rounding of the printed thresholds
    # ──────────────────────────────────────────────────────────────────────────

    main(
        stage1_dir=STAGE1_DIR,
        out_dir=OUT_DIR,
        datasets=DATASETS,
        features=FEATURES,
        cent_pct=CENT_PCT,
        rsa_lower_pct=RSA_LOWER_PCT,
        rsa_upper_pct=RSA_UPPER_PCT,
        n_steps=N_STEPS,
        decimals=DECIMALS,
    )
