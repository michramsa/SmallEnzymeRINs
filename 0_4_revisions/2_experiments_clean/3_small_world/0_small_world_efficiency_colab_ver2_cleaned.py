"""
Small-world test for residue interaction networks (RINs) using the efficiency
framework of Latora & Marchiori (2001), "Efficient behavior of small-world
networks", Phys. Rev. Lett. 87, 198701.

Efficiency measures (unweighted, undirected graphs)
    E_glob : mean of 1/d_ij over all node pairs, with 1/d_ij = 0 for pairs in
             different components. Disconnected graphs are therefore handled
             directly: no largest-connected-component (LCC) restriction is
             needed, and fragmentation is itself counted as lost efficiency.
    E_loc  : mean over nodes i of E_glob of the subgraph induced by the
             neighbours of i (node i removed). Nodes with < 2 neighbours
             contribute 0.

Latora-Marchiori criterion
    A small-world network has E_glob close to that of a random graph (short
    paths) AND E_loc close to that of a lattice (fault-tolerant local
    structure), well above the random graph's E_loc.

Null models (built for every graph, from that same graph)
    random   : degree-preserving rewiring (nx.random_reference) by default, or
               G(n, m) Erdos-Renyi with `random_null="gnm"`.
    lattice  : degree-preserving lattice rewiring (nx.lattice_reference).

Per-graph outputs (one CSV row per graph)
    E_glob, E_loc                         real graph
    E_*_rand / E_*_lat (+ _sd)            null means and standard deviations
    Eg_ratio_rand  = E_glob / E_glob_rand        (near 1  -> random-like paths)
    El_ratio_rand  = E_loc  / E_loc_rand         (>> 1    -> more local than random)
    El_ratio_lat   = E_loc  / E_loc_lat          (near 1  -> lattice-like locality)
    Eg_ratio_lat   = E_glob / E_glob_lat         (> 1     -> shorter paths than lattice)
    Eg_z_rand, El_z_rand                  z-scores against the random ensemble
    El_p_vs_rand                          one-sided empirical p: fraction of random
                                          nulls at least as locally efficient as the
                                          real graph, (1 + k) / (N + 1)
    rand_ncomp_mean, lat_ncomp_mean       mean number of components in the nulls;
                                          should equal n_components (diagnostic)

References
    Latora V, Marchiori M (2001). Efficient behavior of small-world networks.
        Phys Rev Lett 87:198701. (E_glob, E_loc; disconnected graphs allowed)
    Maslov S, Sneppen K (2002). Specificity and stability in topology of protein
        networks. Science 296:910-913. (degree-preserving edge swaps;
        nx.random_reference)

Outputs (one pair of CSVs per PDB id, grouped by dataset)
    <output_dir>/<label>/<PDBID>_<type>_efficiency_<random_null>[_lcc].csv
        one row per graph file belonging to that PDB id
    <output_dir>/<label>/<PDBID>_<type>_summary_<random_null>[_lcc].csv
        median / q25 / q75 of each summary metric over that PDB id's graphs
    <type> is the graph-folder name minus its "_graphml" suffix, e.g. the folder
    "iac_test_graphml" gives type "iac_test". The PDB id is the part of the graph
    file name before the first "_", upper-cased. The <label> sub-folder keeps
    datasets whose folders share a name (e.g. arpeggio/iac vs ring/iac) apart.

Runs are resumable: a PDB id whose efficiency CSV already exists is skipped on
re-run. Set overwrite=True (or delete the CSVs) after changing any setting.
"""

import csv
import os
import random
import multiprocessing as mp
from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm   # works in notebooks and terminals
except ImportError:  # progress bar is optional
    tqdm = None


FIELDS = [
    "graph", "pdb_id", "n_nodes", "n_edges", "density", "n_components",
    "lcc_node_frac", "E_glob", "E_loc",
    "E_glob_rand", "E_glob_rand_sd", "E_loc_rand", "E_loc_rand_sd",
    "E_glob_lat", "E_glob_lat_sd", "E_loc_lat", "E_loc_lat_sd",
    "Eg_ratio_rand", "El_ratio_rand", "Eg_ratio_lat", "El_ratio_lat",
    "Eg_z_rand", "El_z_rand", "El_p_vs_rand",
    "rand_ncomp_mean", "lat_ncomp_mean", "n_random", "n_lattice",
]

SUMMARY_METRICS = [
    "Eg_ratio_rand", "El_ratio_rand", "Eg_ratio_lat", "El_ratio_lat",
    "Eg_z_rand", "El_z_rand", "El_p_vs_rand",
]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_graphml(path, node_type=str):
    """Load a GraphML file as a bare, simple, undirected graph (attributes and
    self-loops dropped, isolated nodes kept)."""
    G = nx.read_graphml(path, node_type=node_type)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from((u, v) for u, v in G.edges() if u != v)
    return H


def largest_cc(G):
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def split_components(G):
    """Connected components as integer-labelled graphs, largest first.
    Node order inside each component follows the order in G, so results are
    reproducible regardless of Python's string-hash randomisation."""
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    comp_of = {n: k for k, c in enumerate(comps) for n in c}
    members = [[] for _ in comps]
    for n in G:
        members[comp_of[n]].append(n)

    parts = []
    for nodes in members:
        idx = {n: i for i, n in enumerate(nodes)}
        H = nx.Graph()
        H.add_nodes_from(range(len(nodes)))
        H.add_edges_from((idx[u], idx[v]) for u in nodes for v in G[u]
                         if idx[u] < idx[v])
        parts.append(H)
    return parts


# ----------------------------------------------------------------------------
# Null models (one connected component at a time)
# ----------------------------------------------------------------------------
def _rewire_component(H, ref_func, niter, rng):
    """Degree-preserving rewiring of one connected component. Components that
    are too small (or complete) to rewire are returned unchanged."""
    n, m = H.number_of_nodes(), H.number_of_edges()
    if n < 4 or m < 2 or m == n * (n - 1) // 2:
        return H.copy()
    return ref_func(H, niter=niter, seed=rng)  # connectivity=True is the default


def _gnm_component(n, m, rng, max_tries=200):
    """Connected G(n, m) with exactly n nodes and m edges."""
    if n < 3 or m >= n * (n - 1) // 2:
        return nx.gnm_random_graph(n, m, seed=rng)
    if m == n - 1:
        return nx.random_labeled_tree(n, seed=rng)
    for _ in range(max_tries):
        H = nx.gnm_random_graph(n, m, seed=rng)
        if nx.is_connected(H):
            return H
    # Fallback: random spanning tree plus (m - n + 1) random extra edges.
    H = nx.random_labeled_tree(n, seed=rng)
    extra = rng.sample(list(nx.non_edges(H)), m - (n - 1))
    H.add_edges_from(extra)
    return H


def make_null(components, kind, niter, rng):
    """One null graph with the same per-component node and edge counts.
    kind: 'degree_preserving', 'gnm', or 'lattice'."""
    parts = []
    for H in components:
        if kind == "degree_preserving":
            parts.append(_rewire_component(H, nx.random_reference, niter, rng))
        elif kind == "lattice":
            parts.append(_rewire_component(H, nx.lattice_reference, niter, rng))
        elif kind == "gnm":
            parts.append(_gnm_component(H.number_of_nodes(),
                                        H.number_of_edges(), rng))
        else:
            raise ValueError(f"Unknown null kind: {kind!r}")
    return nx.disjoint_union_all(parts)


def null_ensemble(components, kind, n_graphs, niter, rng, n_nodes, n_edges,
                  real_degrees):
    """E_glob, E_loc and component count for each of n_graphs null graphs.
    Every null must match n and m; the degree-preserving kinds ('degree_preserving'
    and 'lattice') must also match the real degree sequence."""
    Eg, El, ncomp = [], [], []
    for _ in range(n_graphs):
        R = make_null(components, kind, niter, rng)
        if R.number_of_nodes() != n_nodes or R.number_of_edges() != n_edges:
            raise RuntimeError(
                f"{kind} null has n={R.number_of_nodes()}, m={R.number_of_edges()}; "
                f"expected n={n_nodes}, m={n_edges}")
        if kind != "gnm" and sorted(d for _, d in R.degree()) != real_degrees:
            raise RuntimeError(f"{kind} null does not preserve the degree sequence")
        Eg.append(nx.global_efficiency(R))
        El.append(nx.local_efficiency(R))
        ncomp.append(nx.number_connected_components(R))
    return np.array(Eg), np.array(El), np.array(ncomp)


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def _mean(a):
    return float(np.mean(a)) if len(a) else np.nan


def _sd(a):
    return float(np.std(a, ddof=1)) if len(a) > 1 else np.nan


def _ratio(a, b):
    return a / b if b and np.isfinite(b) else np.nan


def _z(x, a):
    s = _sd(a)
    return (x - _mean(a)) / s if np.isfinite(s) and s > 0 else np.nan


def small_world_efficiency(G, seed_key, random_null="degree_preserving",
                           n_random=100, n_lattice=20, niter_random=10,
                           niter_lattice=5, seed=0):
    """Latora-Marchiori efficiency statistics for one graph against matched nulls."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    comps = split_components(G)
    lcc_n = comps[0].number_of_nodes() if comps else 0

    Eg, El = nx.global_efficiency(G), nx.local_efficiency(G)
    real_degrees = sorted(d for _, d in G.degree())

    # Separate, deterministic random streams per graph and per ensemble.
    rng_r = random.Random(f"{seed}|{seed_key}|random")
    rng_l = random.Random(f"{seed}|{seed_key}|lattice")
    Egr, Elr, ncr = null_ensemble(comps, random_null, n_random, niter_random,
                                  rng_r, n, m, real_degrees)
    Egl, Ell, ncl = null_ensemble(comps, "lattice", n_lattice, niter_lattice,
                                  rng_l, n, m, real_degrees)

    Eg_ratio_rand = _ratio(Eg, _mean(Egr))
    El_ratio_rand = _ratio(El, _mean(Elr))
    Eg_ratio_lat = _ratio(Eg, _mean(Egl))
    El_ratio_lat = _ratio(El, _mean(Ell))
    p_loc = ((1 + int(np.sum(Elr >= El - 1e-12))) / (len(Elr) + 1)
             if len(Elr) else np.nan)

    return {
        "n_nodes": n, "n_edges": m, "density": nx.density(G),
        "n_components": len(comps), "lcc_node_frac": lcc_n / n if n else np.nan,
        "E_glob": Eg, "E_loc": El,
        "E_glob_rand": _mean(Egr), "E_glob_rand_sd": _sd(Egr),
        "E_loc_rand": _mean(Elr), "E_loc_rand_sd": _sd(Elr),
        "E_glob_lat": _mean(Egl), "E_glob_lat_sd": _sd(Egl),
        "E_loc_lat": _mean(Ell), "E_loc_lat_sd": _sd(Ell),
        "Eg_ratio_rand": Eg_ratio_rand, "El_ratio_rand": El_ratio_rand,
        "Eg_ratio_lat": Eg_ratio_lat, "El_ratio_lat": El_ratio_lat,
        "Eg_z_rand": _z(Eg, Egr), "El_z_rand": _z(El, Elr),
        "El_p_vs_rand": p_loc,
        "rand_ncomp_mean": _mean(ncr), "lat_ncomp_mean": _mean(ncl),
        "n_random": n_random, "n_lattice": n_lattice,
    }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def dataset_type(graphs_dir):
    """Dataset 'type' taken from the graph folder name, minus a trailing
    '_graphml': .../iac_test_graphml -> 'iac_test'."""
    name = Path(graphs_dir).name
    return name[:-len("_graphml")] if name.endswith("_graphml") else name


def pdb_id_of(path):
    """PDB id = text before the first underscore of the file name, upper-cased."""
    return Path(path).stem.split("_")[0].upper()


def summarise_graphs(df):
    """Median and quartiles of each summary metric over the rows of df."""
    rows = []
    for metric in SUMMARY_METRICS:
        x = df[metric].dropna()
        rows.append({
            "metric": metric, "n_graphs": len(x),
            "median": x.median(), "q25": x.quantile(0.25),
            "q75": x.quantile(0.75),
        })
    return pd.DataFrame(rows)


def _write_csv_atomic(df, path):
    """Write to a temp file then rename, so a half-written CSV never exists."""
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def analyse_pdb(task, eff_path, summ_path, random_null, n_random, n_lattice,
                niter_random, niter_lattice, seed, use_largest_component):
    """Worker: analyse every graph file of one PDB id and write that PDB id's
    efficiency and summary CSVs. Returns (pdb_id, None) or (pdb_id, error).
    The efficiency CSV is written last, so its existence marks the PDB id done."""
    pdb_id, paths = task
    try:
        rows = []
        for path in sorted(paths):
            name = Path(path).stem
            G = load_graphml(path)
            if use_largest_component:
                G = largest_cc(G)
            row = small_world_efficiency(
                G, name, random_null=random_null, n_random=n_random,
                n_lattice=n_lattice, niter_random=niter_random,
                niter_lattice=niter_lattice, seed=seed)
            row["graph"] = name
            row["pdb_id"] = pdb_id
            rows.append(row)
        eff = pd.DataFrame(rows, columns=FIELDS)
        _write_csv_atomic(summarise_graphs(eff), summ_path)
        _write_csv_atomic(eff, eff_path)
        return pdb_id, None
    except Exception as exc:
        return pdb_id, f"{type(exc).__name__}: {exc}"


def run_dataset(label, graphs_dir, out_dir, random_null, n_random, n_lattice,
                niter_random, niter_lattice, seed, use_largest_component,
                num_processes, overwrite, suffix):
    """Process every .graphml in graphs_dir, writing two CSVs per PDB id."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = dataset_type(graphs_dir)

    by_pdb = {}
    for p in Path(graphs_dir).glob("*.graphml"):
        by_pdb.setdefault(pdb_id_of(p), []).append(p)
    print(f"\n[{label}] {sum(map(len, by_pdb.values()))} GraphML files, "
          f"{len(by_pdb)} PDB ids in {graphs_dir}  (type: {dtype})")

    def eff_path(pdb):
        return out_dir / f"{pdb}_{dtype}_efficiency{suffix}.csv"

    def summ_path(pdb):
        return out_dir / f"{pdb}_{dtype}_summary{suffix}.csv"

    todo = [(pdb, paths) for pdb, paths in by_pdb.items()
            if overwrite or not eff_path(pdb).exists()]
    todo.sort(key=lambda t: -sum(p.stat().st_size for p in t[1]))  # biggest first
    print(f"[{label}] {len(by_pdb) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        return

    kw = dict(random_null=random_null, n_random=n_random, n_lattice=n_lattice,
              niter_random=niter_random, niter_lattice=niter_lattice,
              seed=seed, use_largest_component=use_largest_component)
    tasks = [((pdb, [str(p) for p in paths]), str(eff_path(pdb)),
              str(summ_path(pdb)), kw) for pdb, paths in todo]

    errors = []
    try:
        with mp.Pool(processes=num_processes or mp.cpu_count()) as pool:
            results = pool.imap_unordered(_run_task, tasks)
            if tqdm is not None:
                results = tqdm(results, total=len(tasks), desc=label)
            for pdb, err in results:
                if err:
                    errors.append((pdb, err))
                    print(f"  FAILED {pdb}: {err}")
    except KeyboardInterrupt:
        print(f"[{label}] Interrupted. Re-run to resume; finished PDB ids are kept.")
        raise
    if errors:
        print(f"[{label}] {len(errors)} PDB id(s) failed and will be retried on re-run")


def _run_task(arg):
    """Pool entry point (top-level so it can be pickled)."""
    task, eff, summ, kw = arg
    return analyse_pdb(task, eff_path=eff, summ_path=summ, **kw)


def main(datasets, output_dir, random_null="degree_preserving", n_random=100,
         n_lattice=20, niter_random=10, niter_lattice=5, seed=0,
         use_largest_component=False, num_processes=None, overwrite=False):
    """
    datasets              : {label: folder of .graphml files}
    output_dir            : root folder; each dataset gets <output_dir>/<label>/
    random_null           : 'degree_preserving' (nx.random_reference) or 'gnm'
    n_random, n_lattice   : null graphs per real graph
    niter_random/lattice  : rewiring passes (each edge rewired ~niter times)
    seed                  : base seed; each graph gets its own stream from
                            (seed, graph name), so runs are reproducible
    use_largest_component : True = analyse only each graph's LCC (sensitivity check)
    num_processes         : worker processes (None = all CPUs)
    overwrite             : True = recompute and replace existing PDB-id CSVs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{random_null}" + ("_lcc" if use_largest_component else "")

    for label, graphs_dir in datasets.items():
        run_dataset(
            label, graphs_dir, output_dir / label, random_null, n_random,
            n_lattice, niter_random, niter_lattice, seed,
            use_largest_component, num_processes, overwrite, suffix)

    print(f"\nSaved results to: {output_dir}")


def mount_drive_if_colab(mount_point="/content/drive"):
    """Mount Google Drive when running inside Colab; do nothing elsewhere."""
    try:
        from google.colab import drive
    except ImportError:
        return False
    if not os.path.ismount(mount_point):
        drive.mount(mount_point)
    return True


if __name__ == "__main__":
    # ---- Edit these values ----
    # Root folder on Google Drive that holds your data and receives results.
    drive_root = "/content/drive/MyDrive/AcrNetworks-clean-v2/0_4_revisions"

    ### yes - this means all of the graphs have to be inside of your google drive unfortunately
    ### this will also change based on how your graphs are organized
    datasets = {
        "arp_iac": f"{drive_root}/1_data/arpeggio/iac/iac_test_graphml",
        "arp_no_iac": f"{drive_root}/1_data/arpeggio/no_iac/no_iac_test_graphml",
        "ring_iac": f"{drive_root}/1_data/ring/iac/iac_test_graphml",
        "ring_no_iac": f"{drive_root}/1_data/ring/no_iac/no_iac_test_graphml",
        "pcn5": f"{drive_root}/1_data/ring/pcns/5a_pcns_test_graphml",
        "pcn6": f"{drive_root}/1_data/ring/pcns/6a_pcns_test_graphml",
        "pcn7": f"{drive_root}/1_data/ring/pcns/7a_pcns_test_graphml",
    }
    output_dir = f"{drive_root}/2_experiments/3_small_world"
    random_null = "degree_preserving"   # or "gnm" (density-only Erdos-Renyi null)
    n_random = 100                      # random null graphs per real graph
    n_lattice = 20                      # lattice null graphs per real graph
    niter_random = 10
    niter_lattice = 5
    seed = 0
    use_largest_component = False       # True = LCC-only sensitivity analysis
    num_processes = os.cpu_count()      # 8 on your Colab Pro+ runtime
    overwrite = False                   # True = discard existing CSVs and redo
    mount_drive = True                  # mounts Drive when running in Colab
    # ---------------------------

    if mount_drive:
        mount_drive_if_colab()

    main(
        datasets=datasets,
        output_dir=output_dir,
        random_null=random_null,
        n_random=n_random,
        n_lattice=n_lattice,
        niter_random=niter_random,
        niter_lattice=niter_lattice,
        seed=seed,
        use_largest_component=use_largest_component,
        num_processes=num_processes,
        overwrite=overwrite,
    )
