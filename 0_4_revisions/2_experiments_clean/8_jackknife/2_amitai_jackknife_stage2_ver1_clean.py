"""
amitai_jackknife_stage2_ver1.py
===============================
Stage 2 of the Amitai (2004)-style jackknife. Reads the Stage 1 tables
(amitai_jackknife_stage1_ver1.py must sit in the same folder; its MCC and
selection functions are reused so both stages apply the identical rule) and
writes every number the paper needs. Figures are made in Stage 3.

Models: every <dataset>__<centrality> (centrality >= c_lower and
rsa_lower <= RSA <= rsa_upper) plus RSA_only (rsa_lower <= RSA <= rsa_upper).

1. Leave-one-out (Amitai's jackknife)
   For each protein: drop it, compute every combination's MCC on each of the
   remaining proteins, take the median, pick the best combination (ties ->
   fewer predicted residues -> narrower RSA window -> lowest grid index),
   apply it to the held-out protein.

2. Random classifier
   For each held-out protein, k = number of residues the rule predicted.
   (a) k residues drawn at random from the whole protein
   (b) k residues drawn at random from residues inside that fold's RSA window
       (tests centrality beyond RSA; not defined for RSA_only)
   Picking k of N residues of which P are functional gives
   TP ~ Hypergeometric(P, N - P, k), so draws are taken directly from that
   distribution. N_RANDOM replicates -> null distribution of the median MCC,
   pooled sensitivity, and pooled sensitivity per partner type.
   Also Amitai's analytic Z-score (method of Aloy et al. 2001):
       z = (PO - PR) / sqrt(PR (1 - PR) / n)
   PO = pooled precision of the predictions, PR = fraction of all residues
   that are functional, n = all residues (as in Amitai et al. 2004).

3. Bootstrap A -- threshold stability
   N_BOOT_A resamples of the proteins (with replacement); the best combination
   is selected again on each. Selection frequencies and 95% intervals of each
   threshold.

4. Bootstrap B -- uncertainty of the leave-one-out results
   N_BOOT_B resamples of the proteins' held-out results (nothing refit). The
   SAME resamples are used for every model, so differences between models
   (dataset vs dataset, centrality vs centrality, centrality vs RSA_only) get
   paired confidence intervals.

5. ROC / PR curves from held-out proteins (bands from Bootstrap B)
   score "windowed"   : centrality z-score for residues inside the fold's RSA
                        window, -inf outside (ranked last). Main curve.
   score "unwindowed" : centrality z-score of every residue (no RSA filter).
   RSA_only has no continuous score; it appears as an operating point only.
   Missing centrality -> -inf.

Partner types: sensitivity only, for role set "binding" and "binding+dual".
A residue contacting several partner types counts in each.

Outputs (under <out_root>/stage2/<grid_name>/)
----------------------------------------------
    summary_models.csv            one row per model: everything headline
    loo_per_protein.csv           held-out result of every protein x model
    loo_selection_counts.csv      how often each combination was chosen across folds
    partner_sensitivity.csv       pooled + median sensitivity per partner type,
                                  CIs, random expectation and p-values
    partner_per_protein.csv       partner counts / hits per protein x model
    random_null.csv               null distribution of median MCC (a and b)
    bootstrapA_selection.csv      selection frequency of each combination
    bootstrapA_thresholds.csv     95% intervals of each threshold
    bootstrapB_replicates.csv     per replicate x model: median MCC, pooled
                                  sensitivity/precision, AUC, average precision
    bootstrapB_contrasts.csv      paired differences between models with CIs
    curves_roc.csv / curves_pr.csv  curve on a fixed grid + 95% band
    operating_points.csv          pooled FPR/TPR/precision of each model's rule
    bootstrap_indices.npz         the resampled protein indices (reproducible)
    stage2.log

All settings live in the `if __name__ == "__main__":` block at the bottom.
"""

import logging
import multiprocessing as mp
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from amitai_jackknife_stage1_ver1 import (  # noqa: E402
    mcc_from_counts, select_best, edge_flags,
)

ROLE_SETS = {"binding": ["binding"], "binding+dual": ["binding", "dual"]}
CURVE_GRID = np.linspace(0, 1, 101)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("amitai_stage2")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(out_dir / "stage2.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def build_model_list(datasets, features) -> list:
    models = [{"model": f"{ds}__{f}", "dataset": ds, "feature": f,
               "value_col": f"{ds}__{f}", "combos_file": f"combos_centrality_{f}.csv"}
              for f in features for ds in datasets]
    models.append({"model": "rsa_only", "dataset": "all", "feature": "RSA_only",
                   "value_col": None, "combos_file": "combos_rsa_only.csv"})
    return models


def load_model(grid_dir: Path, m: dict) -> dict:
    z = np.load(grid_dir / f"confusion_{m['model']}.npz")
    combos = pd.read_csv(grid_dir / m["combos_file"])
    d = {k: z[k] for k in ("tp", "fp", "fn", "n_res", "n_pos")}
    d["pdb_ids"] = z["pdb_ids"].astype(str)
    d["combos"] = combos
    d["mcc"] = mcc_from_counts(d["tp"], d["fp"], d["fn"], d["n_res"])
    d["n_pred"] = d["tp"] + d["fp"]
    return d


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def choose(mcc_rows, n_pred_rows, combos) -> int:
    """Best combination on the given protein rows (Amitai rule + tie-break)."""
    med = np.median(mcc_rows, axis=0)
    return select_best(med, n_pred_rows.sum(axis=0),
                       combos["rsa_width"].to_numpy(), combos["valid"].to_numpy())


def run_loo(d: dict) -> np.ndarray:
    """Chosen combination for each held-out protein."""
    n_p = d["mcc"].shape[0]
    chosen = np.empty(n_p, dtype=int)
    keep = np.ones(n_p, dtype=bool)
    for i in range(n_p):
        keep[i] = False
        chosen[i] = choose(d["mcc"][keep], d["n_pred"][keep], d["combos"])
        keep[i] = True
    return chosen


def bootstrap_a_worker(args) -> np.ndarray:
    """Re-select the best combination on each bootstrap resample."""
    mcc, n_pred, combos, idx = args
    return np.array([choose(mcc[rows], n_pred[rows], combos) for rows in idx])


# ---------------------------------------------------------------------------
# Held-out residues: predictions, scores, partners
# ---------------------------------------------------------------------------

def heldout_residues(res: pd.DataFrame, m: dict, d: dict, chosen: np.ndarray,
                     ptypes: list) -> pd.DataFrame:
    """
    Per residue of every protein, using that protein's own fold thresholds:
    pred, in_window, score_windowed, score_unwindowed.
    """
    combos = d["combos"]
    prot_index = {p: i for i, p in enumerate(d["pdb_ids"])}
    r = res[res["pdb_id"].isin(prot_index)].copy()
    pi = r["pdb_id"].map(prot_index).to_numpy()
    cb = combos.iloc[chosen[pi]]
    rsa = r["rsa_empirical"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        in_win = (rsa >= cb["rsa_lower"].to_numpy()) & (rsa <= cb["rsa_upper"].to_numpy())
        if m["value_col"] is None:
            pred = in_win
            s_win = np.full(len(r), np.nan)
            s_all = np.full(len(r), np.nan)
        else:
            v = r[m["value_col"]].to_numpy(float)
            pred = in_win & (v >= cb["centrality_lower"].to_numpy())
            s_all = np.where(np.isnan(v), -np.inf, v)
            s_win = np.where(in_win, s_all, -np.inf)

    out = pd.DataFrame({
        "prot": pi, "pdb_id": r["pdb_id"].to_numpy(),
        "y": r["is_functional"].to_numpy(int),
        "functional_type": r["functional_type"].to_numpy(),
        "pred": pred, "in_window": in_win,
        "score_windowed": s_win, "score_unwindowed": s_all,
    })
    for t in ptypes:
        out[f"partner_{t}"] = r[f"partner_{t}"].astype(bool).to_numpy()
    return out


def per_protein_table(hr: pd.DataFrame, d: dict, chosen: np.ndarray,
                      m: dict, logger) -> pd.DataFrame:
    is_pos = hr["y"].to_numpy() == 1
    flags = pd.DataFrame({
        "prot": hr["prot"].to_numpy(),
        "tp": hr["pred"].to_numpy() & is_pos,
        "fp": hr["pred"].to_numpy() & ~is_pos,
        "n_win": hr["in_window"].to_numpy(),
        "p_win": hr["in_window"].to_numpy() & is_pos,
    })
    n_p = len(d["pdb_ids"])
    sums = flags.groupby("prot").sum().reindex(range(n_p), fill_value=0).astype(int)
    tp, fp = sums["tp"].to_numpy(), sums["fp"].to_numpy()
    n_win, p_win = sums["n_win"], sums["p_win"]

    # consistency check against the Stage 1 confusion tables
    tp1 = d["tp"][np.arange(n_p), chosen]
    fp1 = d["fp"][np.arange(n_p), chosen]
    if not (np.array_equal(tp, tp1) and np.array_equal(fp, fp1)):
        bad = np.where((tp != tp1) | (fp != fp1))[0]
        raise RuntimeError(f"[{m['model']}] held-out predictions disagree with the "
                           f"Stage 1 confusion table for {len(bad)} proteins "
                           f"(e.g. {d['pdb_ids'][bad[0]]}). Was the residue table "
                           f"rebuilt after the confusion tables?")

    n, pos = d["n_res"].astype(int), d["n_pos"].astype(int)
    fn = pos - tp
    tn = n - tp - fp - fn
    cb = d["combos"].iloc[chosen].reset_index(drop=True)
    t = pd.DataFrame({
        "model": m["model"], "dataset": m["dataset"], "feature": m["feature"],
        "pdb_id": d["pdb_ids"], "combo_id": chosen,
        "centrality_lower": cb["centrality_lower"], "rsa_lower": cb["rsa_lower"],
        "rsa_upper": cb["rsa_upper"],
        "n_residues": n, "n_functional": pos, "n_predicted": tp + fp,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_in_window": n_win.to_numpy(),
        "n_functional_in_window": p_win.to_numpy(),
    })
    t["mcc"] = mcc_from_counts(tp, fp, fn, n)
    t["sensitivity"] = tp / pos
    with np.errstate(invalid="ignore", divide="ignore"):
        t["precision"] = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
    t["specificity_tnr"] = tn / (tn + fp)
    return t


def partner_per_protein(hr: pd.DataFrame, pp: pd.DataFrame, ptypes: list,
                        m: dict) -> pd.DataFrame:
    rows = []
    for rs_name, roles in ROLE_SETS.items():
        role_ok = hr["functional_type"].isin(roles)
        for t in ptypes:
            mask = role_ok & hr[f"partner_{t}"]
            if not mask.any():
                continue
            sub = hr[mask]
            g = sub.groupby("prot")
            agg = pd.DataFrame({
                "n_partner": g.size(),
                "n_hit": g["pred"].sum(),
                "n_partner_in_window": g["in_window"].sum(),
            }).reset_index()
            agg["model"] = m["model"]
            agg["role_set"] = rs_name
            agg["partner_type"] = t
            rows.append(agg)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["pdb_id"] = pp["pdb_id"].to_numpy()[out["prot"]]
    return out


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------

def prepare_curve(scores: np.ndarray, y: np.ndarray, prot: np.ndarray) -> dict:
    """Sort residues once (descending score) and find tie groups."""
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    ends = np.r_[np.nonzero(s[1:] != s[:-1])[0], len(s) - 1]
    return {"y": y[order].astype(float), "prot": prot[order], "ends": ends}


def curve_from_weights(c: dict, w_prot: np.ndarray) -> dict:
    """ROC / PR points, AUC, average precision with protein weights."""
    w = w_prot[c["prot"]]
    ctp = np.cumsum(w * c["y"])[c["ends"]]
    cfp = np.cumsum(w * (1 - c["y"]))[c["ends"]]
    P, N = ctp[-1], cfp[-1]
    tpr = np.r_[0.0, ctp / P]
    fpr = np.r_[0.0, cfp / N]
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = ctp / (ctp + cfp)
    rec = ctp / P
    auc = float(np.trapezoid(tpr, fpr))
    ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    return {"fpr": fpr, "tpr": tpr, "recall": rec, "precision": prec,
            "auc": auc, "ap": ap}


def on_grid(cv: dict) -> tuple:
    """TPR at fixed FPR; precision at fixed recall (first point reaching it)."""
    fpr, tpr = cv["fpr"], cv["tpr"]
    # upper envelope of each vertical segment
    last = np.r_[np.nonzero(np.diff(fpr))[0], len(fpr) - 1]   # fpr is sorted
    tpr_g = np.interp(CURVE_GRID, fpr[last], tpr[last])
    pos = np.searchsorted(cv["recall"], CURVE_GRID, side="left")
    pos = np.clip(pos, 0, len(cv["precision"]) - 1)
    prec_g = cv["precision"][pos]
    return tpr_g, prec_g


def curves_worker(args) -> dict:
    """Full-data curve + Bootstrap B replicates for one model and score type."""
    scores, y, prot, n_p, weights = args
    c = prepare_curve(scores, y, prot)
    full = curve_from_weights(c, np.ones(n_p))
    tg, pg = on_grid(full)
    auc_b, ap_b, tg_b, pg_b = [], [], [], []
    for w in weights:
        cv = curve_from_weights(c, w)
        auc_b.append(cv["auc"])
        ap_b.append(cv["ap"])
        a, b = on_grid(cv)
        tg_b.append(a)
        pg_b.append(b)
    return {"auc": full["auc"], "ap": full["ap"], "tpr_grid": tg, "prec_grid": pg,
            "auc_b": np.array(auc_b), "ap_b": np.array(ap_b),
            "tpr_grid_b": np.array(tg_b), "prec_grid_b": np.array(pg_b)}


# ---------------------------------------------------------------------------
# Random classifier
# ---------------------------------------------------------------------------

def random_null(pp: pd.DataFrame, n_rep: int, rng, variant: str) -> dict:
    """Null distribution of median MCC and pooled sensitivity."""
    n = pp["n_residues"].to_numpy()
    pos = pp["n_functional"].to_numpy()
    k = pp["n_predicted"].to_numpy()
    if variant == "b":
        n, pos = pp["n_in_window"].to_numpy(), pp["n_functional_in_window"].to_numpy()
    tp = rng.hypergeometric(pos, n - pos, k, size=(n_rep, len(k)))
    N_all = pp["n_residues"].to_numpy()
    P_all = pp["n_functional"].to_numpy()
    fp = k - tp
    fn = P_all - tp
    mcc = mcc_from_counts(tp, fp, fn, np.broadcast_to(N_all, tp.shape))
    return {"median_mcc": np.median(mcc, axis=1),
            "pooled_sens": tp.sum(axis=1) / P_all.sum()}


def partner_null(ppp: pd.DataFrame, pp: pd.DataFrame, n_rep: int, rng,
                 variant: str) -> np.ndarray:
    """Null pooled sensitivity for one (model, role set, partner type)."""
    k = pp["n_predicted"].to_numpy()[ppp["prot"]]
    if variant == "a":
        n = pp["n_residues"].to_numpy()[ppp["prot"]]
        good = ppp["n_partner"].to_numpy()
    else:
        n = pp["n_in_window"].to_numpy()[ppp["prot"]]
        good = ppp["n_partner_in_window"].to_numpy()
    hits = rng.hypergeometric(good, n - good, k, size=(n_rep, len(k)))
    return hits.sum(axis=1) / ppp["n_partner"].sum()


def p_greater(null: np.ndarray, obs: float) -> float:
    return float((1 + np.sum(null >= obs - 1e-12)) / (len(null) + 1))


def ci(x, level=0.95):
    a = (1 - level) / 2
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return np.nan, np.nan
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(out_root, grid_name, datasets, features, n_random, n_boot_a,
         n_boot_b, seed, n_processes, ci_level):
    stage1_dir = Path(out_root) / "stage1"
    grid_dir = stage1_dir / "grids" / grid_name
    out_dir = Path(out_root) / "stage2" / grid_name
    logger = setup_logger(out_dir)
    rng = np.random.default_rng(seed)

    logger.info("=" * 70)
    logger.info("AMITAI JACKKNIFE -- STAGE 2: LOO, random classifier, bootstraps")
    logger.info(f"  stage1 grid : {grid_dir}")
    logger.info(f"  random reps : {n_random}   bootstrap A : {n_boot_a}   "
                f"bootstrap B : {n_boot_b}   seed : {seed}   processes : {n_processes}")
    logger.info("=" * 70)

    res = pd.read_csv(stage1_dir / "residue_table.csv",
                      dtype={"chain": str, "icode": str}, keep_default_na=False,
                      na_values=[""])
    res["icode"] = res["icode"].fillna("")
    ptypes = sorted(c[len("partner_"):] for c in res.columns if c.startswith("partner_"))
    for t in ptypes:
        res[f"partner_{t}"] = res[f"partner_{t}"].astype(str).str.lower() == "true"

    models = build_model_list(datasets, features)
    data = {m["model"]: load_model(grid_dir, m) for m in models}
    pdb_ids = data[models[0]["model"]]["pdb_ids"]
    for m in models:
        if not np.array_equal(data[m["model"]]["pdb_ids"], pdb_ids):
            raise RuntimeError(f"Protein order differs in {m['model']}")
    n_p = len(pdb_ids)
    if (data[models[0]["model"]]["n_pos"] == 0).any():
        raise RuntimeError("Some proteins have no functional residues; rerun Stage 1")
    logger.info(f"{n_p} proteins, {len(models)} models, partner types: {ptypes}")

    # shared resamples
    idx_a = rng.integers(0, n_p, size=(n_boot_a, n_p))
    idx_b = rng.integers(0, n_p, size=(n_boot_b, n_p))
    w_b = np.stack([np.bincount(r, minlength=n_p) for r in idx_b]).astype(float)
    np.savez_compressed(out_dir / "bootstrap_indices.npz", pdb_ids=pdb_ids,
                        bootstrap_a=idx_a, bootstrap_b=idx_b)

    # ---- 1. leave-one-out ---------------------------------------------------
    loo_tables, partner_tables, hres, chosen_all = [], [], {}, {}
    sel_rows = []
    for m in models:
        d = data[m["model"]]
        chosen = run_loo(d)
        chosen_all[m["model"]] = chosen
        hr = heldout_residues(res, m, d, chosen, ptypes)
        pp = per_protein_table(hr, d, chosen, m, logger)
        hres[m["model"]] = hr
        loo_tables.append(pp)
        partner_tables.append(partner_per_protein(hr, pp, ptypes, m))
        vc = pd.Series(chosen).value_counts()
        for cid, cnt in vc.items():
            row = d["combos"].iloc[cid]
            sel_rows.append({"model": m["model"], "combo_id": int(cid),
                             "centrality_lower": row["centrality_lower"],
                             "rsa_lower": row["rsa_lower"], "rsa_upper": row["rsa_upper"],
                             "n_folds": int(cnt), "fraction_folds": cnt / n_p,
                             "on_grid_edge": edge_flags(d["combos"], int(cid))})
        logger.info(f"LOO {m['model']:<28} median MCC {pp['mcc'].median():.3f}  "
                    f"distinct choices {len(vc)}")
    loo = pd.concat(loo_tables, ignore_index=True)
    ppart = pd.concat([p for p in partner_tables if len(p)], ignore_index=True)
    loo.to_csv(out_dir / "loo_per_protein.csv", index=False)
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(out_dir / "loo_selection_counts.csv", index=False)
    ppart.drop(columns="prot").to_csv(out_dir / "partner_per_protein.csv", index=False)
    edge = sel[sel["on_grid_edge"] != ""]
    if len(edge):
        logger.warning(f"{edge['model'].nunique()} models chose a combination on the "
                       f"grid edge in at least one fold (see loo_selection_counts.csv)")

    # ---- 2. bootstrap A + curves (parallel) --------------------------------
    logger.info("Bootstrap A (threshold stability) and ROC/PR bootstrap ...")
    jobs_a = [(data[m["model"]]["mcc"], data[m["model"]]["n_pred"],
               data[m["model"]]["combos"], idx_a) for m in models]
    curve_keys, jobs_c = [], []
    for m in models:
        if m["value_col"] is None:
            continue
        hr = hres[m["model"]]
        for st in ("windowed", "unwindowed"):
            curve_keys.append((m["model"], st))
            jobs_c.append((hr[f"score_{st}"].to_numpy(float), hr["y"].to_numpy(),
                           hr["prot"].to_numpy(), n_p, w_b))
    if n_processes and n_processes > 1:
        with mp.Pool(n_processes) as pool:
            sel_a = pool.map(bootstrap_a_worker, jobs_a)
            curves = pool.map(curves_worker, jobs_c)
    else:
        sel_a = [bootstrap_a_worker(j) for j in jobs_a]
        curves = [curves_worker(j) for j in jobs_c]
    curves = dict(zip(curve_keys, curves))

    # Bootstrap A outputs
    a_rows, a_ci = [], []
    for m, s in zip(models, sel_a):
        combos = data[m["model"]]["combos"]
        vc = pd.Series(s).value_counts()
        for cid, cnt in vc.items():
            row = combos.iloc[cid]
            a_rows.append({"model": m["model"], "combo_id": int(cid),
                           "centrality_lower": row["centrality_lower"],
                           "rsa_lower": row["rsa_lower"], "rsa_upper": row["rsa_upper"],
                           "n_selected": int(cnt), "fraction": cnt / n_boot_a})
        ch = combos.iloc[s]
        r = {"model": m["model"], "n_distinct_selected": len(vc),
             "top_combo_fraction": vc.iloc[0] / n_boot_a}
        for col in ("centrality_lower", "rsa_lower", "rsa_upper"):
            r[f"{col}_median"] = ch[col].median()
            r[f"{col}_lo"], r[f"{col}_hi"] = ci(ch[col], ci_level)
        a_ci.append(r)
    pd.DataFrame(a_rows).to_csv(out_dir / "bootstrapA_selection.csv", index=False)
    a_ci = pd.DataFrame(a_ci)
    a_ci.to_csv(out_dir / "bootstrapA_thresholds.csv", index=False)

    # ---- 3. random classifier ---------------------------------------------
    logger.info("Random classifier ...")
    null_rows, rand = [], {}
    for m in models:
        pp = loo[loo["model"] == m["model"]].reset_index(drop=True)
        for v in ("a", "b"):
            if v == "b" and m["value_col"] is None:
                continue
            nd = random_null(pp, n_random, rng, v)
            rand[(m["model"], v)] = nd
            null_rows.append(pd.DataFrame({"model": m["model"], "variant": v,
                                           "replicate": np.arange(n_random),
                                           "median_mcc": nd["median_mcc"],
                                           "pooled_sensitivity": nd["pooled_sens"]}))
    pd.concat(null_rows, ignore_index=True).to_csv(out_dir / "random_null.csv", index=False)

    # ---- 4. bootstrap B: scalar metrics -----------------------------------
    logger.info("Bootstrap B ...")
    rep_rows, rep = [], {}
    for m in models:
        pp = loo[loo["model"] == m["model"]]
        mcc = pp["mcc"].to_numpy()
        tp, fp, pos = (pp[c].to_numpy(float) for c in ("tp", "fp", "n_functional"))
        r = {"median_mcc": np.median(mcc[idx_b], axis=1),
             "mean_mcc": (w_b @ mcc) / n_p,
             "pooled_sensitivity": (w_b @ tp) / (w_b @ pos),
             "pooled_precision": (w_b @ tp) / np.maximum(w_b @ (tp + fp), 1)}
        if m["value_col"] is not None:
            for st in ("windowed", "unwindowed"):
                r[f"auc_{st}"] = curves[(m["model"], st)]["auc_b"]
                r[f"ap_{st}"] = curves[(m["model"], st)]["ap_b"]
        rep[m["model"]] = r
        rep_rows.append(pd.DataFrame({"model": m["model"],
                                      "replicate": np.arange(n_boot_b), **r}))
    pd.concat(rep_rows, ignore_index=True).to_csv(out_dir / "bootstrapB_replicates.csv",
                                                  index=False)

    # ---- 5. summary per model ---------------------------------------------
    n_total = int(data[models[0]["model"]]["n_res"].sum())
    prevalence = data[models[0]["model"]]["n_pos"].sum() / n_total
    summ = []
    op_rows = []
    for m in models:
        pp = loo[loo["model"] == m["model"]]
        tp, fp, fn, tn = (pp[c].sum() for c in ("tp", "fp", "fn", "tn"))
        prec = tp / (tp + fp) if tp + fp else np.nan
        r = {"model": m["model"], "dataset": m["dataset"], "feature": m["feature"],
             "n_proteins": n_p,
             "median_mcc": pp["mcc"].median(), "mean_mcc": pp["mcc"].mean(),
             "pooled_sensitivity": tp / (tp + fn), "pooled_precision": prec,
             "mean_sensitivity": pp["sensitivity"].mean(),
             "mean_precision": pp["precision"].mean(),
             "frac_proteins_full_hit": (pp["sensitivity"] == 1).mean(),
             "frac_proteins_any_hit": (pp["tp"] > 0).mean(),
             "n_predicted_total": int(tp + fp)}
        for k in ("median_mcc", "mean_mcc", "pooled_sensitivity", "pooled_precision"):
            r[f"{k}_lo"], r[f"{k}_hi"] = ci(rep[m["model"]][k], ci_level)
        if m["value_col"] is not None:
            for st in ("windowed", "unwindowed"):
                cv = curves[(m["model"], st)]
                r[f"auc_{st}"], r[f"ap_{st}"] = cv["auc"], cv["ap"]
                r[f"auc_{st}_lo"], r[f"auc_{st}_hi"] = ci(cv["auc_b"], ci_level)
                r[f"ap_{st}_lo"], r[f"ap_{st}_hi"] = ci(cv["ap_b"], ci_level)
        for v in ("a", "b"):
            nd = rand.get((m["model"], v))
            if nd is None:
                continue
            r[f"random_{v}_median_mcc_mean"] = nd["median_mcc"].mean()
            r[f"random_{v}_median_mcc_hi"] = ci(nd["median_mcc"], ci_level)[1]
            r[f"random_{v}_p_median_mcc"] = p_greater(nd["median_mcc"], r["median_mcc"])
            r[f"random_{v}_pooled_sens_mean"] = nd["pooled_sens"].mean()
            r[f"random_{v}_p_pooled_sens"] = p_greater(nd["pooled_sens"],
                                                       r["pooled_sensitivity"])
        r["amitai_z"] = (prec - prevalence) / np.sqrt(prevalence * (1 - prevalence) / n_total)
        sc = sel[sel["model"] == m["model"]].sort_values("n_folds", ascending=False).iloc[0]
        r["loo_top_centrality_lower"] = sc["centrality_lower"]
        r["loo_top_rsa_lower"] = sc["rsa_lower"]
        r["loo_top_rsa_upper"] = sc["rsa_upper"]
        r["loo_top_fraction_folds"] = sc["fraction_folds"]
        r["loo_n_distinct_choices"] = int((sel["model"] == m["model"]).sum())
        r["loo_any_edge_choice"] = bool((sel[sel["model"] == m["model"]]["on_grid_edge"] != "").any())
        summ.append(r)
        op_rows.append({"model": m["model"], "fpr": fp / (fp + tn),
                        "tpr": tp / (tp + fn), "precision": prec,
                        "prevalence": prevalence})
    summary = pd.DataFrame(summ).merge(a_ci.add_prefix("bootA_").rename(
        columns={"bootA_model": "model"}), on="model", how="left")
    summary.to_csv(out_dir / "summary_models.csv", index=False)
    pd.DataFrame(op_rows).to_csv(out_dir / "operating_points.csv", index=False)

    # ---- 6. curves on grid ------------------------------------------------
    roc_rows, pr_rows = [], []
    for (model, st), cv in curves.items():
        lo_t, hi_t = np.nanquantile(cv["tpr_grid_b"], [(1 - ci_level) / 2,
                                                        1 - (1 - ci_level) / 2], axis=0)
        lo_p, hi_p = np.nanquantile(cv["prec_grid_b"], [(1 - ci_level) / 2,
                                                         1 - (1 - ci_level) / 2], axis=0)
        roc_rows.append(pd.DataFrame({"model": model, "score": st, "fpr": CURVE_GRID,
                                      "tpr": cv["tpr_grid"], "tpr_lo": lo_t, "tpr_hi": hi_t}))
        pr_rows.append(pd.DataFrame({"model": model, "score": st, "recall": CURVE_GRID,
                                     "precision": cv["prec_grid"],
                                     "precision_lo": lo_p, "precision_hi": hi_p}))
    if roc_rows:
        pd.concat(roc_rows).to_csv(out_dir / "curves_roc.csv", index=False)
        pd.concat(pr_rows).to_csv(out_dir / "curves_pr.csv", index=False)

    # ---- 7. partner sensitivity -------------------------------------------
    logger.info("Partner-type sensitivity ...")
    p_rows = []
    for (model, rs, t), g in ppart.groupby(["model", "role_set", "partner_type"]):
        pp = loo[loo["model"] == model].reset_index(drop=True)
        n_part = g["n_partner"].to_numpy(float)
        hit = g["n_hit"].to_numpy(float)
        prot = g["prot"].to_numpy()
        w = w_b[:, prot]
        boot = (w @ hit) / (w @ n_part)
        r = {"model": model, "role_set": rs, "partner_type": t,
             "n_residues": int(n_part.sum()), "n_proteins": len(g),
             "pooled_sensitivity": hit.sum() / n_part.sum(),
             "median_protein_sensitivity": np.median(hit / n_part)}
        r["pooled_sensitivity_lo"], r["pooled_sensitivity_hi"] = ci(boot, ci_level)
        for v in ("a", "b"):
            if v == "b" and model == "rsa_only":
                continue
            nd = partner_null(g.reset_index(drop=True), pp, n_random, rng, v)
            r[f"random_{v}_mean"] = nd.mean()
            r[f"random_{v}_p"] = p_greater(nd, r["pooled_sensitivity"])
        p_rows.append(r)
    partner = pd.DataFrame(p_rows)
    partner.to_csv(out_dir / "partner_sensitivity.csv", index=False)

    # ---- 8. paired contrasts ----------------------------------------------
    full = {m["model"]: summary.set_index("model").loc[m["model"]] for m in models}
    pairs = []
    for f in features:
        for a, b in combinations(datasets, 2):
            pairs.append(("dataset", f"{a}__{f}", f"{b}__{f}"))
        for ds in datasets:
            pairs.append(("vs_rsa_only", f"{ds}__{f}", "rsa_only"))
    for ds in datasets:
        for a, b in combinations(features, 2):
            pairs.append(("centrality", f"{ds}__{a}", f"{ds}__{b}"))
    metrics = ["median_mcc", "mean_mcc", "pooled_sensitivity", "pooled_precision",
               "auc_windowed", "ap_windowed", "auc_unwindowed", "ap_unwindowed"]
    c_rows = []
    for kind, a, b in pairs:
        for k in metrics:
            if k not in rep[a] or k not in rep[b]:
                continue
            diff = rep[a][k] - rep[b][k]
            est = full[a][k] - full[b][k]
            lo, hi = ci(diff, ci_level)
            # two-sided bootstrap p; floored at 1/B (report as "< 1/B")
            p = 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))
            c_rows.append({"comparison": kind, "model_a": a, "model_b": b,
                           "metric": k, "difference": est, "lo": lo, "hi": hi,
                           "p_boot": min(1.0, max(p, 1 / len(diff)))})
    pd.DataFrame(c_rows).to_csv(out_dir / "bootstrapB_contrasts.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        cols = ["model", "median_mcc", "median_mcc_lo", "median_mcc_hi",
                "pooled_sensitivity", "pooled_precision", "auc_windowed",
                "random_a_p_median_mcc", "random_b_p_median_mcc",
                "loo_top_centrality_lower", "loo_top_rsa_lower", "loo_top_rsa_upper"]
        cols = [c for c in cols if c in summary.columns]
        logger.info("Summary:\n" + summary[cols].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
    logger.info(f"Stage 2 complete. Outputs in {out_dir}")
    logger.info("=" * 70)
    return summary


if __name__ == "__main__":

    # ── Edit these values ─────────────────────────────────────────────────────
    # Same OUT_ROOT as Stage 1 (Stage 1 tables are read from <OUT_ROOT>/stage1)
    OUT_ROOT = Path("...0_4_revisions/2_experiments/8.8_jackknife")
    GRID_NAME = "percentile"            # Stage 1 grid folder to analyse -- UPDATE
    DATASETS = ["arp_iac", "arp_no_iac", "ring_iac", "ring_no_iac", "pcn5", "pcn6", "pcn7"]
    FEATURES = ["StdDegree", "StdCloseness", "StdBetweenness", "StdPageRank"]

    N_RANDOM = 1000                 # random-classifier replicates
    N_BOOT_A = 1000                 # bootstrap A (threshold stability)
    N_BOOT_B = 1000                 # bootstrap B (CIs, ROC/PR bands, contrasts)
    CI_LEVEL = 0.95
    SEED = 20041                    # fixed seed -> identical results on rerun
    N_PROCESSES = 4                 # parallel workers for bootstrap A + curves (1 = serial)
    # ──────────────────────────────────────────────────────────────────────────

    main(
        out_root=OUT_ROOT,
        grid_name=GRID_NAME,
        datasets=DATASETS,
        features=FEATURES,
        n_random=N_RANDOM,
        n_boot_a=N_BOOT_A,
        n_boot_b=N_BOOT_B,
        seed=SEED,
        n_processes=N_PROCESSES,
        ci_level=CI_LEVEL,
    )
