"""
amitai_jackknife_stage1_ver1.py
===============================
Stage 1 of the Amitai (2004)-style jackknife: build the data tables that
every later stage (leave-one-out, bootstraps, random classifier, ROC/PR)
reads. Nothing in this stage selects thresholds for the jackknife.

What it does
------------
1. Residue table (one row per canonical residue of every test protein)
   - Centralities from <std_root>/<dataset>/<set_name>/<PDBid>_std.csv for
     every dataset (graph construction): StdDegree, StdCloseness,
     StdBetweenness, StdPageRank. Canonical residues only.
   - The residue list of a protein is the UNION over all datasets, so every
     dataset is scored against the same residues and the same functional
     residues. A residue missing from one dataset's network gets NaN
     centrality there and is therefore never predicted functional (it still
     counts as a false negative if it is functional). Counts are logged.
   - Only proteins present in ALL datasets are kept (dropped ones logged).
   - RSA-empirical = SASA-abs / empirical max SASA of the amino acid
     (Tien et al. 2013, same table as SASA.py), from the Lee-Richards rows of
     <sasa_dir>/<PDBid>_sasa.csv. Stored as a fraction (0-1).
   - Functional labels from the BioLiP partner CSV
     (pdb_id, chain, residue, role, partner_type, ligand_ids): a residue
     listed there is functional with that role (binding / catalytic / dual);
     every other residue is non-functional. Each partner type a residue
     contacts becomes a True in its partner_<type> column ('none' is not a
     partner type).

2. Confusion tables (one per dataset x centrality, plus one RSA-only)
   For every protein and every threshold combination of the grid:
       predicted functional  <=>  centrality >= c_lower
                                  and rsa_lower <= RSA <= rsa_upper
   TP, FP, FN are stored (TN = n_residues - TP - FP - FN). Missing values
   compare False, so a residue with NaN centrality or RSA is never predicted.

3. Full-data preview (NOT the jackknife result): the best combination using
   ALL proteins, chosen by the same rule the jackknife will use
   (highest median per-protein MCC; ties -> fewer predicted residues ->
   narrower RSA window -> lowest grid index). Used only to check that the
   tables look sensible and that the optimum is not on the grid edge.

Outputs (under <out_root>/stage1/)
----------------------------------
    residue_table.csv           one row per residue: keys, aa, SASA-abs,
                                RSA-empirical, labels, partner_<type>
                                columns, and <dataset>__<Std feature> columns
    proteins.csv                one row per protein: counts, which datasets
                                had it, whether it is used, drop reason
    join_report.csv             counts from every join / check
    partner_type_counts.csv     residues per partner type x role
    network_coverage.csv        per dataset: residues / functional residues
                                absent from that dataset's network
    grids/<grid_name>/
        combos_centrality_<feature>.csv  combination table per centrality
        combos_rsa_only.csv
        confusion_<dataset>__<feature>.npz
        confusion_rsa_only.npz
            arrays: tp, fp, fn (n_proteins x n_combos, int32),
                    n_res, n_pos (n_proteins), pdb_ids, combo_id
        preview_full_data.csv   best combination per model on all proteins
    stage1.log

All settings live in the `if __name__ == "__main__":` block at the bottom.
"""

import logging
import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tien MZ et al. (2013) PLoS ONE 8(11): e80635 -- empirical maxima (same as SASA.py)
MAX_SASA_EMP = {
    "ALA": 121, "ARG": 265, "ASN": 187, "ASP": 187, "CYS": 148,
    "GLN": 214, "GLU": 214, "GLY":  97, "HIS": 216, "ILE": 195,
    "LEU": 191, "LYS": 230, "MET": 203, "PHE": 228, "PRO": 154,
    "SER": 143, "THR": 163, "TRP": 264, "TYR": 255, "VAL": 165,
}

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# If one residue is given different roles on different rows, keep the highest
ROLE_PRIORITY = {"dual": 3, "catalytic": 2, "binding": 1}

KEY = ["pdb_id", "chain", "resnum", "icode"]
RESNUM_RE = re.compile(r"^(-?\d+)([A-Za-z]?)$")
PARTNER_RES_RE = re.compile(r"^([A-Za-z])(-?\d+)([A-Za-z]?)$")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("amitai_stage1")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(out_dir / "stage1.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _clean_str(s: pd.Series) -> pd.Series:
    """String column with NaN -> '' and surrounding whitespace removed."""
    return s.astype("string").fillna("").str.strip().astype(str)


def parse_resnum(s: pd.Series):
    """
    '527' -> (527, ''), '27A' -> (27, 'A'), 527.0 -> (527, '').
    Unparseable values become (NaN, '').
    """
    txt = _clean_str(s).str.replace(r"\.0$", "", regex=True)
    parts = txt.str.extract(RESNUM_RE)
    num = pd.to_numeric(parts[0], errors="coerce")
    icode = parts[1].fillna("").str.upper()
    return num, icode


def to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return _clean_str(s).str.lower().isin(["true", "1", "yes", "t"])


def add_report(report: list, step: str, item: str, value) -> None:
    report.append({"step": step, "item": item, "value": value})


# ---------------------------------------------------------------------------
# Loading: standardized centralities
# ---------------------------------------------------------------------------

def load_std_dataset(std_root: Path, dataset: str, set_name: str,
                     features: list, logger, report) -> pd.DataFrame:
    """Canonical residues of one dataset, with KEY columns, aa3 and features."""
    folder = Path(std_root) / dataset / set_name
    paths = sorted(folder.glob("*_std.csv"))
    if not paths:
        raise FileNotFoundError(f"[{dataset}] no *_std.csv files in {folder}")

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    step = f"std:{dataset}"
    add_report(report, step, "files", len(paths))
    add_report(report, step, "rows_read", len(df))

    missing = [c for c in ["PDBid", "Chain", "Residue", "Residue number",
                           "IsCanonical"] + features if c not in df.columns]
    if missing:
        raise KeyError(f"[{dataset}] missing columns in _std.csv files: {missing}")

    # sanity checks on the bookkeeping columns
    if "dataset" in df.columns:
        bad = _clean_str(df["dataset"]) != dataset
        if bad.any():
            logger.warning(f"[{dataset}] {bad.sum():,} rows have a 'dataset' value "
                           f"different from the folder name "
                           f"(e.g. {df.loc[bad, 'dataset'].iloc[0]!r})")
        add_report(report, step, "rows_dataset_mismatch", int(bad.sum()))
    if "set_name" in df.columns:
        bad = _clean_str(df["set_name"]) != set_name
        if bad.any():
            logger.warning(f"[{dataset}] dropping {bad.sum():,} rows whose "
                           f"set_name is not '{set_name}'")
        add_report(report, step, "rows_wrong_set_name_dropped", int(bad.sum()))
        df = df[~bad]

    canon = to_bool(df["IsCanonical"])
    add_report(report, step, "rows_noncanonical_dropped", int((~canon).sum()))
    df = df[canon].copy()

    out = pd.DataFrame({
        "pdb_id": _clean_str(df["PDBid"]).str.lower(),
        "chain":  _clean_str(df["Chain"]),
        "aa3":    _clean_str(df["Residue"]).str.upper(),
    })
    out["resnum"], out["icode"] = parse_resnum(df["Residue number"])
    for f in features:
        out[f] = pd.to_numeric(df[f], errors="coerce").to_numpy()

    bad_num = out["resnum"].isna()
    if bad_num.any():
        logger.warning(f"[{dataset}] {bad_num.sum()} rows with unparseable "
                       f"'Residue number' dropped")
    add_report(report, step, "rows_bad_resnum_dropped", int(bad_num.sum()))
    out = out[~bad_num]
    out["resnum"] = out["resnum"].astype(int)

    not_std_aa = ~out["aa3"].isin(AA3_TO_1)
    if not_std_aa.any():
        logger.warning(f"[{dataset}] {not_std_aa.sum()} canonical rows with a "
                       f"residue name outside the 20 amino acids "
                       f"({sorted(out.loc[not_std_aa, 'aa3'].unique())[:10]}); dropped")
    add_report(report, step, "rows_nonstandard_aa_dropped", int(not_std_aa.sum()))
    out = out[~not_std_aa]

    dup = out.duplicated(KEY, keep="first")
    if dup.any():
        logger.warning(f"[{dataset}] {dup.sum()} duplicate residue keys "
                       f"(pdb, chain, number, insertion code); first kept")
    add_report(report, step, "rows_duplicate_key_dropped", int(dup.sum()))
    out = out[~dup]

    n_icode = (out["icode"] != "").sum()
    if n_icode:
        logger.info(f"[{dataset}] {n_icode} residues carry an insertion code")

    chains = out.groupby("pdb_id")["chain"].nunique()
    multi = chains[chains > 1]
    if len(multi):
        logger.warning(f"[{dataset}] {len(multi)} proteins have more than one "
                       f"chain: {list(multi.index[:10])}")
    add_report(report, step, "proteins_multichain", int(len(multi)))
    add_report(report, step, "canonical_residues_kept", len(out))
    add_report(report, step, "proteins", out["pdb_id"].nunique())

    nan_counts = {f: int(out[f].isna().sum()) for f in features}
    for f, n in nan_counts.items():
        add_report(report, step, f"nan_{f}", n)
        if n:
            logger.info(f"[{dataset}] {f}: {n:,} NaN among canonical residues")

    logger.info(f"[{dataset}] {out['pdb_id'].nunique()} proteins, "
                f"{len(out):,} canonical residues")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Loading: SASA -> RSA-empirical
# ---------------------------------------------------------------------------

def load_sasa(sasa_dir: Path, pdb_ids: list, algorithm: str,
              logger, report) -> tuple:
    """
    RSA-empirical for the given proteins.
    Returns (table with KEY + aa3_sasa + sasa_abs + rsa_empirical,
             set of pdb_ids that had no SASA file).
    """
    files = {p.name[: -len("_sasa.csv")].lower(): p
             for p in Path(sasa_dir).glob("*_sasa.csv")}
    frames, no_file = [], set()
    for pid in pdb_ids:
        path = files.get(pid)
        if path is None:
            no_file.add(pid)
            continue
        frames.append(pd.read_csv(path).assign(_pid=pid))

    step = "sasa"
    add_report(report, step, "proteins_without_sasa_file", len(no_file))
    if no_file:
        logger.warning(f"No SASA file for {len(no_file)} proteins: "
                       f"{sorted(no_file)[:10]}")
    if not frames:
        raise FileNotFoundError(f"No *_sasa.csv files found for any protein in {sasa_dir}")

    df = pd.concat(frames, ignore_index=True)
    need = ["Algorithm", "Chain", "Residue", "Residue ID", "SASA-abs"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"SASA files are missing columns: {missing}")

    algo_ok = _clean_str(df["Algorithm"]).str.lower() == algorithm.lower()
    add_report(report, step, "rows_other_algorithm_dropped", int((~algo_ok).sum()))
    df = df[algo_ok]
    if df.empty:
        raise ValueError(f"No SASA rows with Algorithm == '{algorithm}'")

    out = pd.DataFrame({
        "pdb_id":   df["_pid"].to_numpy(),
        "chain":    _clean_str(df["Chain"]).to_numpy(),
        "aa3_sasa": _clean_str(df["Residue"]).str.upper().to_numpy(),
        "sasa_abs": pd.to_numeric(df["SASA-abs"], errors="coerce").to_numpy(),
    })
    out["resnum"], _ = parse_resnum(df["Residue ID"].reset_index(drop=True))
    if "Insertion code" in df.columns:
        out["icode"] = _clean_str(df["Insertion code"].reset_index(drop=True)).str.upper()
    else:
        out["icode"] = ""
    out = out[out["resnum"].notna()].copy()
    out["resnum"] = out["resnum"].astype(int)

    dup = out.duplicated(KEY, keep="first")
    add_report(report, step, "rows_duplicate_key_dropped", int(dup.sum()))
    if dup.any():
        logger.warning(f"SASA: {dup.sum()} duplicate residue keys; first kept")
    out = out[~dup]

    max_sasa = out["aa3_sasa"].map(MAX_SASA_EMP)
    out["rsa_empirical"] = out["sasa_abs"] / max_sasa
    add_report(report, step, "rows_kept", len(out))
    return out.reset_index(drop=True), no_file


# ---------------------------------------------------------------------------
# Loading: functional labels + partner types
# ---------------------------------------------------------------------------

def load_partners(partner_csv: Path, logger, report) -> tuple:
    """
    One row per functional residue: KEY, aa1_partner, functional_type, and the
    set of partner types (excluding 'none').
    Returns (residue table, sorted list of partner types seen).
    """
    df = pd.read_csv(partner_csv)
    need = ["pdb_id", "chain", "residue", "role", "partner_type"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Partner CSV is missing columns: {missing}")

    step = "partners"
    add_report(report, step, "rows_read", len(df))

    res = _clean_str(df["residue"]).str.extract(PARTNER_RES_RE)
    bad = res[1].isna()
    if bad.any():
        logger.warning(f"Partner CSV: {bad.sum()} rows with an unparseable "
                       f"'residue' (e.g. {df.loc[bad, 'residue'].iloc[0]!r}); dropped")
    add_report(report, step, "rows_bad_residue_dropped", int(bad.sum()))

    p = pd.DataFrame({
        "pdb_id":       _clean_str(df["pdb_id"]).str.lower(),
        "chain":        _clean_str(df["chain"]),
        "aa1_partner":  res[0].str.upper(),
        "resnum":       pd.to_numeric(res[1], errors="coerce"),
        "icode":        res[2].fillna("").str.upper(),
        "role":         _clean_str(df["role"]).str.lower(),
        "partner_type": _clean_str(df["partner_type"]),
    })[~bad.to_numpy()]
    p["resnum"] = p["resnum"].astype(int)

    unknown_role = ~p["role"].isin(ROLE_PRIORITY)
    if unknown_role.any():
        logger.warning(f"Partner CSV: {unknown_role.sum()} rows with unknown role "
                       f"{sorted(p.loc[unknown_role, 'role'].unique())}; dropped")
    add_report(report, step, "rows_unknown_role_dropped", int(unknown_role.sum()))
    p = p[~unknown_role]

    # role should be the same on every row of a residue
    n_roles = p.groupby(KEY)["role"].nunique()
    conflict = n_roles[n_roles > 1]
    if len(conflict):
        logger.warning(f"Partner CSV: {len(conflict)} residues have conflicting "
                       f"roles; the highest of dual > catalytic > binding is kept")
    add_report(report, step, "residues_role_conflict", int(len(conflict)))

    n_letters = p.groupby(KEY)["aa1_partner"].nunique()
    n_letter_conflict = int((n_letters > 1).sum())
    if n_letter_conflict:
        logger.warning(f"Partner CSV: {n_letter_conflict} residues are listed with "
                       f"different amino-acid letters on different rows; they "
                       f"will show up as letter mismatches")
    add_report(report, step, "residues_letter_conflict", n_letter_conflict)

    p["_rank"] = p["role"].map(ROLE_PRIORITY)
    agg = (p.groupby(KEY, sort=False)
             .agg(aa1_partner=("aa1_partner", lambda s: "".join(sorted(set(s)))),
                  _rank=("_rank", "max"),
                  partner_types=("partner_type",
                                 lambda s: frozenset(t for t in s if t.lower() != "none")))
             .reset_index())
    inv = {v: k for k, v in ROLE_PRIORITY.items()}
    agg["functional_type"] = agg["_rank"].map(inv)
    agg = agg.drop(columns="_rank")

    types = sorted({t for s in agg["partner_types"] for t in s})
    add_report(report, step, "functional_residues", len(agg))
    add_report(report, step, "partner_types_seen", ";".join(types))

    # binding/dual residues should have a partner, catalytic ones should not
    no_partner = agg["functional_type"].isin(["binding", "dual"]) & \
        (agg["partner_types"].map(len) == 0)
    if no_partner.any():
        logger.warning(f"Partner CSV: {no_partner.sum()} binding/dual residues "
                       f"have no partner type other than 'none'")
    add_report(report, step, "binding_or_dual_without_partner", int(no_partner.sum()))
    cat_with = (agg["functional_type"] == "catalytic") & \
        (agg["partner_types"].map(len) > 0)
    add_report(report, step, "catalytic_with_partner", int(cat_with.sum()))

    logger.info(f"Partner CSV: {len(agg):,} functional residues in "
                f"{agg['pdb_id'].nunique()} proteins; partner types: {types}")
    return agg, types


# ---------------------------------------------------------------------------
# Residue table
# ---------------------------------------------------------------------------

def build_residue_table(std_root, datasets, set_name, features, sasa_dir,
                        sasa_algorithm, partner_csv, logger, report) -> tuple:
    """
    Returns (residue_table, proteins_table, partner_types, coverage_table).
    """
    # 1. centralities for every dataset
    per_ds = {ds: load_std_dataset(std_root, ds, set_name, features, logger, report)
              for ds in datasets}

    # 2. proteins present in every dataset
    present = {ds: set(t["pdb_id"]) for ds, t in per_ds.items()}
    all_ids = sorted(set().union(*present.values()))
    in_all = set.intersection(*present.values())
    prot = pd.DataFrame({"pdb_id": all_ids})
    for ds in datasets:
        prot[f"in_{ds}"] = prot["pdb_id"].isin(present[ds])
    prot["in_all_datasets"] = prot["pdb_id"].isin(in_all)
    n_missing = (~prot["in_all_datasets"]).sum()
    if n_missing:
        logger.warning(f"{n_missing} proteins are missing from at least one "
                       f"dataset and are dropped: "
                       f"{list(prot.loc[~prot['in_all_datasets'], 'pdb_id'][:10])}")
    add_report(report, "proteins", "union_over_datasets", len(all_ids))
    add_report(report, "proteins", "in_all_datasets", len(in_all))

    # 3. master residue list = union over datasets (same residues for every dataset)
    keys = pd.concat([t[KEY + ["aa3"]] for t in per_ds.values()], ignore_index=True)
    keys = keys[keys["pdb_id"].isin(in_all)]
    aa_conflict = keys.groupby(KEY)["aa3"].nunique()
    aa_conflict = aa_conflict[aa_conflict > 1]
    if len(aa_conflict):
        logger.warning(f"{len(aa_conflict)} residue keys have different residue "
                       f"names in different datasets; first seen is kept")
    add_report(report, "residues", "aa_name_conflict_across_datasets", int(len(aa_conflict)))
    master = keys.drop_duplicates(KEY, keep="first").reset_index(drop=True)
    add_report(report, "residues", "master_residues", len(master))

    # 4. RSA-empirical
    sasa, no_sasa = load_sasa(sasa_dir, sorted(in_all), sasa_algorithm, logger, report)
    master = master.merge(sasa, on=KEY, how="left")
    unmatched = master["aa3_sasa"].isna() & ~master["pdb_id"].isin(no_sasa)
    add_report(report, "sasa", "master_residues_without_sasa_row", int(unmatched.sum()))
    if unmatched.any():
        logger.warning(f"{unmatched.sum():,} residues (in proteins that have a "
                       f"SASA file) have no matching SASA row -> RSA NaN")
    aa_mismatch = master["aa3_sasa"].notna() & (master["aa3_sasa"] != master["aa3"])
    add_report(report, "sasa", "residue_name_mismatch", int(aa_mismatch.sum()))
    if aa_mismatch.any():
        logger.warning(f"{aa_mismatch.sum()} residues have a different residue "
                       f"name in the SASA file -> RSA set to NaN")
        master.loc[aa_mismatch, ["sasa_abs", "rsa_empirical"]] = np.nan
    master = master.drop(columns="aa3_sasa")
    n_rsa_nan = int(master["rsa_empirical"].isna().sum())
    add_report(report, "sasa", "residues_rsa_nan", n_rsa_nan)
    n_gt1 = int((master["rsa_empirical"] > 1).sum())
    add_report(report, "sasa", "residues_rsa_above_1", n_gt1)
    logger.info(f"RSA-empirical: {n_rsa_nan:,} NaN, {n_gt1:,} above 1.0 "
                f"(possible with empirical maxima)")

    # 5. functional labels + partner types
    partners, ptypes = load_partners(partner_csv, logger, report)
    part_in = partners[partners["pdb_id"].isin(in_all)]
    add_report(report, "partners", "functional_residues_in_used_proteins", len(part_in))
    master = master.merge(part_in, on=KEY, how="left", indicator=True)
    matched = master["_merge"] == "both"
    master = master.drop(columns="_merge")

    # partner rows that did not land on a residue
    lab_keys = part_in.merge(master[KEY], on=KEY, how="left", indicator=True)
    lost = lab_keys[lab_keys["_merge"] == "left_only"]
    chains_by_prot = master.groupby("pdb_id")["chain"].agg(set)
    other_chain = lost.apply(lambda r: r["chain"] not in chains_by_prot.get(r["pdb_id"], set()),
                             axis=1) if len(lost) else pd.Series(dtype=bool)
    n_other_chain = int(other_chain.sum()) if len(lost) else 0
    n_lost_same_chain = len(lost) - n_other_chain
    add_report(report, "partners", "unmatched_on_other_chain", n_other_chain)
    add_report(report, "partners", "unmatched_on_network_chain", n_lost_same_chain)
    if n_other_chain:
        logger.info(f"{n_other_chain} partner rows are on chains not in the "
                    f"networks (expected for multi-chain PDB entries); ignored")
    if n_lost_same_chain:
        logger.warning(f"{n_lost_same_chain} functional residues on a network "
                       f"chain have no matching canonical residue (non-canonical, "
                       f"numbering mismatch, or not in any network); they are "
                       f"lost. See join_report.csv / stage1.log.")
        for _, r in lost[~other_chain.to_numpy()].head(20).iterrows():
            logger.debug(f"  lost: {r['pdb_id']} {r['chain']} "
                         f"{r['aa1_partner']}{r['resnum']}{r['icode']} ({r['functional_type']})")

    aa1 = master["aa3"].map(AA3_TO_1)
    letter_mismatch = matched & (master["aa1_partner"] != aa1)
    add_report(report, "partners", "amino_acid_letter_mismatch", int(letter_mismatch.sum()))
    if letter_mismatch.any():
        logger.warning(f"{letter_mismatch.sum()} functional residues: the letter in "
                       f"the partner CSV differs from the structure's residue "
                       f"(numbering problem?). They are kept as functional.")
        for _, r in master[letter_mismatch].head(20).iterrows():
            logger.debug(f"  letter mismatch: {r['pdb_id']} {r['chain']} "
                         f"{r['resnum']}{r['icode']} structure={r['aa3']} "
                         f"partner={r['aa1_partner']}")

    master["functional_type"] = master["functional_type"].fillna("non_functional")
    master["is_functional"] = (master["functional_type"] != "non_functional").astype(int)
    pt = master["partner_types"].apply(lambda s: s if isinstance(s, frozenset) else frozenset())
    for t in ptypes:
        master[f"partner_{t}"] = pt.map(lambda s, t=t: t in s)
    master = master.drop(columns=["partner_types", "aa1_partner"])

    # 6. centralities per dataset as <dataset>__<feature>
    coverage = []
    for ds, t in per_ds.items():
        cols = {f: f"{ds}__{f}" for f in features}
        master = master.merge(t[KEY + features].rename(columns=cols),
                              on=KEY, how="left", indicator="_in_net")
        absent = master["_in_net"] == "left_only"
        master = master.drop(columns="_in_net")
        coverage.append({
            "dataset": ds,
            "residues": len(master),
            "residues_absent_from_network": int(absent.sum()),
            "functional_absent_from_network": int((absent & (master["is_functional"] == 1)).sum()),
            **{f"nan_{f}": int(master[cols[f]].isna().sum()) for f in features},
        })
        if absent.any():
            logger.info(f"[{ds}] {absent.sum():,} residues absent from this network "
                        f"({int((absent & (master['is_functional'] == 1)).sum())} functional); "
                        f"never predicted functional in this dataset")

    master = master.sort_values(KEY).reset_index(drop=True)

    # 7. protein table
    g = master.groupby("pdb_id")
    counts = pd.DataFrame({
        "n_residues":   g.size(),
        "n_functional": g["is_functional"].sum(),
        "n_catalytic":  g["functional_type"].apply(lambda s: (s == "catalytic").sum()),
        "n_binding":    g["functional_type"].apply(lambda s: (s == "binding").sum()),
        "n_dual":       g["functional_type"].apply(lambda s: (s == "dual").sum()),
        "n_rsa_nan":    g["rsa_empirical"].apply(lambda s: s.isna().sum()),
        "chains":       g["chain"].agg(lambda s: ";".join(sorted(set(s)))),
    }).reset_index()
    prot = prot.merge(counts, on="pdb_id", how="left")
    prot["has_sasa_file"] = ~prot["pdb_id"].isin(no_sasa)

    reason = np.where(~prot["in_all_datasets"], "missing_from_a_dataset",
             np.where(~prot["has_sasa_file"], "no_sasa_file",
             np.where(prot["n_functional"].fillna(0) == 0, "no_functional_residues", "")))
    prot["drop_reason"] = reason
    prot["used"] = prot["drop_reason"] == ""
    n_nofunc = int((prot["drop_reason"] == "no_functional_residues").sum())
    if n_nofunc:
        logger.warning(f"{n_nofunc} proteins have no functional residues after the "
                       f"join and are excluded from MCC medians (as in Amitai)")
    if (prot["drop_reason"] == "no_sasa_file").any():
        logger.warning(f"{(prot['drop_reason'] == 'no_sasa_file').sum()} proteins "
                       f"dropped for lack of a SASA file")

    used = set(prot.loc[prot["used"], "pdb_id"])
    master = master[master["pdb_id"].isin(used)].reset_index(drop=True)
    add_report(report, "proteins", "used", len(used))
    add_report(report, "residues", "used", len(master))
    logger.info(f"Residue table: {len(used)} proteins, {len(master):,} residues, "
                f"{int(master['is_functional'].sum()):,} functional "
                f"({100 * master['is_functional'].mean():.2f}%)")
    return master, prot, ptypes, pd.DataFrame(coverage)


def partner_counts(res: pd.DataFrame, ptypes: list) -> pd.DataFrame:
    rows = []
    for t in ptypes:
        col = res[f"partner_{t}"]
        for role in ["binding", "dual", "catalytic"]:
            m = col & (res["functional_type"] == role)
            rows.append({"partner_type": t, "role": role,
                         "n_residues": int(m.sum()),
                         "n_proteins": int(res.loc[m, "pdb_id"].nunique())})
        m = col & res["functional_type"].isin(["binding", "dual"])
        rows.append({"partner_type": t, "role": "binding+dual",
                     "n_residues": int(m.sum()),
                     "n_proteins": int(res.loc[m, "pdb_id"].nunique())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def axis_values(spec) -> np.ndarray:
    """(start, stop, step) inclusive, or an explicit list of values."""
    if isinstance(spec, (list, np.ndarray)):
        return np.round(np.asarray(spec, dtype=float), 6)
    start, stop, step = spec
    n = int(round((stop - start) / step)) + 1
    return np.round(start + step * np.arange(n), 6)


def build_combos(c_vals, lo_vals, hi_vals) -> pd.DataFrame:
    """
    One row per combination in C order (centrality, rsa_lower, rsa_upper),
    matching the flattening of the confusion arrays. c_vals None = RSA-only.
    """
    c_axis = [np.nan] if c_vals is None else list(c_vals)
    rows = list(product(c_axis, lo_vals, hi_vals))
    combos = pd.DataFrame(rows, columns=["centrality_lower", "rsa_lower", "rsa_upper"])
    combos.insert(0, "combo_id", np.arange(len(combos)))
    combos["valid"] = combos["rsa_lower"] < combos["rsa_upper"]
    combos["rsa_width"] = combos["rsa_upper"] - combos["rsa_lower"]
    return combos


# ---------------------------------------------------------------------------
# Confusion counts
# ---------------------------------------------------------------------------

def confusion_counts(res: pd.DataFrame, value_col, c_vals, lo_vals, hi_vals,
                     pdb_ids: list) -> dict:
    """
    For every protein and combination: TP, FP, FN (int32, n_proteins x n_combos).
    predicted = (value >= c) & (rsa >= lo) & (rsa <= hi); NaN compares False.
    value_col None -> RSA-only (no centrality condition).
    """
    n_c = 1 if value_col is None else len(c_vals)
    n_combo = n_c * len(lo_vals) * len(hi_vals)
    n_p = len(pdb_ids)
    tp = np.zeros((n_p, n_combo), dtype=np.int32)
    pred = np.zeros((n_p, n_combo), dtype=np.int32)
    n_res = np.zeros(n_p, dtype=np.int32)
    n_pos = np.zeros(n_p, dtype=np.int32)

    groups = res.groupby("pdb_id", sort=False).indices
    rsa_all = res["rsa_empirical"].to_numpy(float)
    y_all = res["is_functional"].to_numpy(float)
    v_all = None if value_col is None else res[value_col].to_numpy(float)

    with np.errstate(invalid="ignore"):
        for i, pid in enumerate(pdb_ids):
            idx = groups[pid]
            rsa, y = rsa_all[idx], y_all[idx]
            L = (rsa[:, None] >= lo_vals[None, :]).astype(float)
            U = (rsa[:, None] <= hi_vals[None, :]).astype(float)
            if value_col is None:
                A = np.ones((len(idx), 1))
            else:
                A = (v_all[idx][:, None] >= c_vals[None, :]).astype(float)
            P = np.einsum("ra,rb,rc->abc", A, L, U, optimize=True)
            T = np.einsum("r,ra,rb,rc->abc", y, A, L, U, optimize=True)
            pred[i] = np.rint(P).astype(np.int32).ravel()
            tp[i] = np.rint(T).astype(np.int32).ravel()
            n_res[i] = len(idx)
            n_pos[i] = int(y.sum())

    fp = pred - tp
    fn = n_pos[:, None] - tp
    return {"tp": tp, "fp": fp, "fn": fn, "n_res": n_res, "n_pos": n_pos}


def mcc_from_counts(tp, fp, fn, n_res) -> np.ndarray:
    """MCC; 0 where undefined (no predictions, or all predicted)."""
    tp, fp, fn = (np.asarray(a, dtype=float) for a in (tp, fp, fn))
    n = np.asarray(n_res, dtype=float)
    if n.ndim == 1 and tp.ndim == 2:
        n = n[:, None]
    tn = n - tp - fp - fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (tp * tn - fp * fn) / den
    return np.where(den > 0, m, 0.0)


def select_best(median_mcc, n_pred_total, rsa_width, valid) -> int:
    """
    Highest median MCC; ties -> fewer predicted residues -> narrower RSA window
    -> lowest grid index. Invalid combinations (rsa_lower >= rsa_upper) excluded.
    """
    score = np.where(valid, np.round(median_mcc, 12), -np.inf)
    order = np.lexsort((np.arange(len(score)), rsa_width, n_pred_total, -score))
    return int(order[0])


def edge_flags(combos: pd.DataFrame, best: int) -> str:
    row = combos.iloc[best]
    flags = []
    for col in ["centrality_lower", "rsa_lower", "rsa_upper"]:
        vals = combos[col].dropna().unique()
        if len(vals) < 2 or pd.isna(row[col]):
            continue
        if row[col] == vals.min():
            flags.append(f"{col}=min")
        elif row[col] == vals.max():
            flags.append(f"{col}=max")
    return ";".join(flags)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(std_root, set_name, datasets, features, sasa_dir, sasa_algorithm,
         partner_csv, out_root, grid_name, grid_centrality_lower,
         grid_rsa_lower, grid_rsa_upper):
    out_dir = Path(out_root) / "stage1"
    grid_dir = out_dir / "grids" / grid_name
    grid_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)
    report = []

    logger.info("=" * 70)
    logger.info("AMITAI JACKKNIFE -- STAGE 1: residue table + confusion tables")
    logger.info(f"  std_root    : {std_root}  (set '{set_name}')")
    logger.info(f"  datasets    : {datasets}")
    logger.info(f"  features    : {features}")
    logger.info(f"  sasa_dir    : {sasa_dir}  (algorithm '{sasa_algorithm}')")
    logger.info(f"  partner_csv : {partner_csv}")
    logger.info(f"  out_dir     : {out_dir}")
    logger.info("=" * 70)

    # ---- residue table --------------------------------------------------
    res, prot, ptypes, coverage = build_residue_table(
        std_root, datasets, set_name, features, sasa_dir, sasa_algorithm,
        partner_csv, logger, report)

    res.to_csv(out_dir / "residue_table.csv", index=False)
    prot.to_csv(out_dir / "proteins.csv", index=False)
    coverage.to_csv(out_dir / "network_coverage.csv", index=False)
    pc = partner_counts(res, ptypes)
    pc.to_csv(out_dir / "partner_type_counts.csv", index=False)
    logger.info("Partner types (binding + dual residues):\n" +
                pc[pc["role"] == "binding+dual"]
                .drop(columns="role").to_string(index=False))
    logger.info("Functional types:\n" +
                res["functional_type"].value_counts().to_string())

    # ---- grid -----------------------------------------------------------
    lo_vals = axis_values(grid_rsa_lower)
    hi_vals = axis_values(grid_rsa_upper)
    c_vals = {f: axis_values(grid_centrality_lower.get(f, grid_centrality_lower["default"]))
              for f in features}
    logger.info(f"Grid '{grid_name}': RSA lower {lo_vals.min():g}-{lo_vals.max():g} "
                f"(n={len(lo_vals)}), RSA upper {hi_vals.min():g}-{hi_vals.max():g} "
                f"(n={len(hi_vals)})")
    for f in features:
        logger.info(f"  {f:<15} lower {c_vals[f].min():g}-{c_vals[f].max():g} "
                    f"(n={len(c_vals[f])}) -> "
                    f"{len(c_vals[f]) * len(lo_vals) * len(hi_vals):,} combinations")

    pdb_ids = sorted(res["pdb_id"].unique())
    preview = []

    def save_and_preview(name, model_label, dataset, value_col, cv):
        combos = build_combos(cv, lo_vals, hi_vals)
        n_invalid = int((~combos["valid"]).sum())
        if n_invalid:
            logger.warning(f"[{name}] {n_invalid} combinations have rsa_lower >= "
                           f"rsa_upper; kept in the arrays but never selected")
        cc = confusion_counts(res, value_col, cv, lo_vals, hi_vals, pdb_ids)
        np.savez_compressed(grid_dir / f"confusion_{name}.npz",
                            pdb_ids=np.array(pdb_ids), combo_id=combos["combo_id"].to_numpy(),
                            **cc)
        mcc = mcc_from_counts(cc["tp"], cc["fp"], cc["fn"], cc["n_res"])
        med = np.median(mcc, axis=0)
        n_pred = (cc["tp"] + cc["fp"]).sum(axis=0)
        best = select_best(med, n_pred, combos["rsa_width"].to_numpy(),
                           combos["valid"].to_numpy())
        b = combos.iloc[best]
        n_tied = int((np.round(med[combos["valid"]], 12) == np.round(med[best], 12)).sum())
        n_near = int((med[combos["valid"]] >= med[best] - 0.01).sum())
        tp_b, fp_b, fn_b = (cc[k][:, best].sum() for k in ("tp", "fp", "fn"))
        preview.append({
            "dataset": dataset, "model": model_label,
            "best_combo_id": best,
            "centrality_lower": b["centrality_lower"],
            "rsa_lower": b["rsa_lower"], "rsa_upper": b["rsa_upper"],
            "median_mcc": med[best], "mean_mcc": mcc[:, best].mean(),
            "pooled_sensitivity": tp_b / (tp_b + fn_b),
            "pooled_precision": tp_b / (tp_b + fp_b) if tp_b + fp_b else np.nan,
            "n_predicted_total": int(n_pred[best]),
            "n_combos_tied_at_best": n_tied,
            "n_combos_within_0.01": n_near,
            "on_grid_edge": edge_flags(combos, best),
        })
        if preview[-1]["on_grid_edge"]:
            logger.warning(f"[{name}] full-data optimum on grid edge: "
                           f"{preview[-1]['on_grid_edge']}")
        return combos

    for f in features:
        combos = None
        for ds in datasets:
            combos = save_and_preview(f"{ds}__{f}", f, ds, f"{ds}__{f}", c_vals[f])
        combos.to_csv(grid_dir / f"combos_centrality_{f}.csv", index=False)
        logger.info(f"Confusion tables written for {f} ({len(datasets)} datasets)")

    combos = save_and_preview("rsa_only", "RSA_only", "all", None, None)
    combos.to_csv(grid_dir / "combos_rsa_only.csv", index=False)

    pv = pd.DataFrame(preview)
    pv.to_csv(grid_dir / "preview_full_data.csv", index=False)
    pd.DataFrame(report).to_csv(out_dir / "join_report.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        logger.info("Full-data preview (NOT jackknife; sanity check only):\n" +
                    pv[["dataset", "model", "centrality_lower", "rsa_lower",
                        "rsa_upper", "median_mcc", "pooled_sensitivity",
                        "n_combos_within_0.01", "on_grid_edge"]]
                    .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    logger.info(f"Stage 1 complete. Outputs in {out_dir}")
    logger.info("=" * 70)
    return res, prot, pv


if __name__ == "__main__":

    # ── Edit these values ─────────────────────────────────────────────────────
    # OUTPUT_ROOT of standardization_ver3.py (holds <dataset>/test/<PDBid>_std.csv)
    STD_ROOT = Path("...0_4_revisions/2_experiments/5_standardization/...")
    SET_NAME = "test"
    DATASETS = ["arp_iac", "arp_no_iac", "ring_iac", "ring_no_iac", "pcn5", "pcn6", "pcn7"]
    FEATURES = ["StdDegree", "StdCloseness", "StdBetweenness", "StdPageRank"]

    # Folder of <PDBid>_sasa.csv files (FreeSASA output with SASA-abs)   <-- EDIT
    ### This folder will be provided within ZENODO
    SASA_DIR = Path("...0_4_revisions/2_experiments/7_rsa/lee_richards")
    SASA_ALGORITHM = "lee_richards"

    # Long partner table from biolip_partner_types.py (one row per residue x
    # partner type: pdb_id, chain, residue, role, partner_type, ligand_ids)  <-- EDIT
    PARTNER_CSV = Path("...0_4_revisions/2_experiments/1_binding_partners/partner_types_per_residue.csv")

    # Everything from the jackknife goes under here                     <-- EDIT
    OUT_ROOT = Path("...0_4_revisions/2_experiments/8_jackknife")

    # # Threshold grid. Each axis is (start, stop, step), inclusive, or a list.
    # # Stage 1 = Amitai et al. (2004) grid with RSA as a fraction:
    # #   centrality lower 0.9-2.1 by 0.1, RSA lower 2.5-8.5% by 0.5%,
    # #   RSA upper 33-45% by 1%  -> 13 x 13 x 13 = 2,197 combinations.
    # # For the informed grid, change GRID_NAME (new folder) and the ranges;
    # # GRID_CENTRALITY_LOWER can hold a separate range per feature.
    # GRID_NAME = "amitai"
    # GRID_CENTRALITY_LOWER = {
    #     "default": (0.9, 2.1, 0.1),
    #     # "StdBetweenness": (0.5, 4.0, 0.25),   # example per-feature override
    # }
    # GRID_RSA_LOWER = (0.025, 0.085, 0.005)
    # GRID_RSA_UPPER = (0.33, 0.45, 0.01)

    # Label-free percentile grid (amitai_grid_percentiles_ver1.py)
    #   centrality lower: p50-p98 of all residues (pooled over 7 datasets), 13 steps
    #   RSA lower: p1-p30, RSA upper: p40-p95 of all residues
    #   combinations: StdDegree 2,028, StdCloseness 2,028, StdBetweenness 2,028, StdPageRank 2,028
    # GRID_NAME = "percentile"
    # GRID_CENTRALITY_LOWER = {
    #     "default": (0.9, 2.1, 0.1),   # unused when every feature is listed below
    #     "StdDegree": [-0.264, -0.135, -0.014, 0.139, 0.299, 0.422, 0.541, 0.669, 0.828, 1.007, 1.192, 1.529, 2.170],
    #     "StdCloseness": [-0.223, -0.116, -0.009, 0.098, 0.206, 0.321, 0.437, 0.563, 0.700, 0.857, 1.048, 1.307, 1.773],
    #     "StdBetweenness": [-0.293, -0.214, -0.125, -0.025, 0.092, 0.223, 0.377, 0.558, 0.789, 1.097, 1.527, 2.256, 4.148],
    #     "StdPageRank": [-0.034, 0.070, 0.176, 0.288, 0.404, 0.525, 0.655, 0.795, 0.948, 1.122, 1.342, 1.645, 2.240],
    # }
    # GRID_RSA_LOWER = [0.000, 0.001, 0.006, 0.015, 0.026, 0.040, 0.056, 0.075, 0.095, 0.116, 0.137, 0.159]
    # GRID_RSA_UPPER = [0.253, 0.292, 0.331, 0.370, 0.407, 0.446, 0.483, 0.519, 0.561, 0.609, 0.659, 0.718, 0.799]


    #### Amitai style standardization 

    # Label-free percentile grid (amitai_grid_percentiles_ver1.py)
    #   centrality lower: p50-p98 of all residues (pooled over 7 datasets), 13 steps
    #   RSA lower: p1-p30, RSA upper: p40-p95 of all residues
    #   combinations: StdDegree 2,028, StdCloseness 2,028, StdBetweenness 2,028, StdPageRank 2,028
    GRID_NAME = "percentile"
    GRID_CENTRALITY_LOWER = {
        "default": (0.9, 2.1, 0.1),   # unused when every feature is listed below
        "StdDegree": [-0.264, -0.135, -0.014, 0.139, 0.299, 0.422, 0.541, 0.669, 0.828, 1.007, 1.192, 1.529, 2.170],
        "StdCloseness": [0.064, 0.166, 0.266, 0.366, 0.467, 0.574, 0.686, 0.802, 0.932, 1.075, 1.242, 1.459, 1.831],
        "StdBetweenness": [-0.361, -0.277, -0.182, -0.077, 0.043, 0.186, 0.346, 0.537, 0.767, 1.052, 1.415, 1.923, 2.912],
        "StdPageRank": [-0.066, 0.028, 0.134, 0.245, 0.365, 0.492, 0.624, 0.768, 0.929, 1.115, 1.342, 1.636, 2.198],
    }
    GRID_RSA_LOWER = [0.000, 0.001, 0.006, 0.015, 0.026, 0.040, 0.056, 0.075, 0.095, 0.116, 0.137, 0.159]
    GRID_RSA_UPPER = [0.253, 0.292, 0.331, 0.370, 0.407, 0.446, 0.483, 0.519, 0.561, 0.609, 0.659, 0.718, 0.799]


    # ──────────────────────────────────────────────────────────────────────────

    main(
        std_root=STD_ROOT,
        set_name=SET_NAME,
        datasets=DATASETS,
        features=FEATURES,
        sasa_dir=SASA_DIR,
        sasa_algorithm=SASA_ALGORITHM,
        partner_csv=PARTNER_CSV,
        out_root=OUT_ROOT,
        grid_name=GRID_NAME,
        grid_centrality_lower=GRID_CENTRALITY_LOWER,
        grid_rsa_lower=GRID_RSA_LOWER,
        grid_rsa_upper=GRID_RSA_UPPER,
    )
