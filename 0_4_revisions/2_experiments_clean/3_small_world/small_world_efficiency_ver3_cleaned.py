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

Outputs
    <output_dir>/<label>_efficiency_<random_null>[_lcc].csv   per-graph results
    <output_dir>/summary_efficiency_<random_null>[_lcc].csv   median / IQR per dataset

Runs are resumable: finished graphs are skipped on re-run. Set overwrite=True
(or delete the CSV) after changing any setting.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imports import *

try:
    from tqdm import tqdm
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
def analyse_graph(path, random_null, n_random, n_lattice, niter_random,
                  niter_lattice, seed, use_largest_component):
    """Worker: returns (name, row_dict, None) or (name, None, error_message)."""
    path = Path(path)
    name = path.stem
    try:
        G = load_graphml(path)
        if use_largest_component:
            G = largest_cc(G)
        row = small_world_efficiency(
            G, name, random_null=random_null, n_random=n_random,
            n_lattice=n_lattice, niter_random=niter_random,
            niter_lattice=niter_lattice, seed=seed)
        row["graph"] = name
        row["pdb_id"] = name.split("_")[0].upper()
        return name, row, None
    except Exception as exc:
        return name, None, f"{type(exc).__name__}: {exc}"


def run_dataset(label, graphs_dir, out_csv, random_null, n_random, n_lattice,
                niter_random, niter_lattice, seed, use_largest_component,
                num_processes, overwrite):
    """Process every .graphml in graphs_dir, appending one CSV row per graph."""
    out_csv = Path(out_csv)
    files = sorted(Path(graphs_dir).glob("*.graphml"),
                   key=lambda p: -p.stat().st_size)  # biggest first
    print(f"\n[{label}] {len(files)} GraphML files in {graphs_dir}")

    if overwrite and out_csv.exists():
        out_csv.unlink()
    done = set()
    if out_csv.exists():
        done = set(pd.read_csv(out_csv)["graph"].astype(str))
    else:
        with open(out_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    todo = [p for p in files if p.stem not in done]
    print(f"[{label}] {len(done)} already done, {len(todo)} to run")

    if todo:
        worker = partial(
            analyse_graph, random_null=random_null, n_random=n_random,
            n_lattice=n_lattice, niter_random=niter_random,
            niter_lattice=niter_lattice, seed=seed,
            use_largest_component=use_largest_component)
        n_workers = num_processes or mp.cpu_count()
        errors = []
        try:
            with mp.Pool(processes=n_workers) as pool, \
                    open(out_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                results = pool.imap_unordered(worker, [str(p) for p in todo])
                if tqdm is not None:
                    results = tqdm(results, total=len(todo), desc=label)
                for name, row, err in results:
                    if err:
                        errors.append((name, err))
                        print(f"  FAILED {name}: {err}")
                        continue
                    writer.writerow(row)
                    f.flush()
        except KeyboardInterrupt:
            print(f"[{label}] Interrupted. Re-run to resume from {out_csv}")
            return None
        if errors:
            print(f"[{label}] {len(errors)} graph(s) failed and will be retried on re-run")

    return pd.read_csv(out_csv)


def summarise(results):
    """Median and IQR of each metric, per dataset."""
    rows = []
    for label, df in results.items():
        for metric in SUMMARY_METRICS:
            x = df[metric].dropna()
            rows.append({
                "dataset": label, "metric": metric, "n_graphs": len(x),
                "median": x.median(), "q25": x.quantile(0.25),
                "q75": x.quantile(0.75),
            })
    return pd.DataFrame(rows)


def main(datasets, output_dir, random_null="degree_preserving", n_random=100,
         n_lattice=20, niter_random=10, niter_lattice=5, seed=0,
         use_largest_component=False, num_processes=None, overwrite=False):
    """
    datasets              : {label: folder of .graphml files}
    output_dir            : where the CSVs are written
    random_null           : 'degree_preserving' (nx.random_reference) or 'gnm'
    n_random, n_lattice   : null graphs per real graph
    niter_random/lattice  : rewiring passes (each edge rewired ~niter times)
    seed                  : base seed; each graph gets its own stream from
                            (seed, graph name), so runs are reproducible
    use_largest_component : True = analyse only each graph's LCC (sensitivity check)
    num_processes         : worker processes (None = all CPUs)
    overwrite             : True = ignore and replace existing CSVs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{random_null}" + ("_lcc" if use_largest_component else "")

    results = {}
    for label, graphs_dir in datasets.items():
        out_csv = output_dir / f"{label}_efficiency{suffix}.csv"
        df = run_dataset(
            label, graphs_dir, out_csv, random_null, n_random, n_lattice,
            niter_random, niter_lattice, seed, use_largest_component,
            num_processes, overwrite)
        if df is None:
            return
        results[label] = df

    summary = summarise(results)
    summary_path = output_dir / f"summary_efficiency{suffix}.csv"
    summary.to_csv(summary_path, index=False)
    with pd.option_context("display.width", 120, "display.float_format", "{:.3f}".format):
        print("\n" + summary.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    # ---- Edit these values ----
    datasets = {
        "arp_iac": "...0_4_revisions/1_data/arpeggio/iac/iac_test_graphml",
        "arp_no_iac": "...0_4_revisions/1_data/arpeggio/no_iac/no_iac_test_graphml",
        "ring_iac": "...0_4_revisions/1_data/ring/iac/iac_test_graphml",
        "ring_no_iac": "...0_4_revisions/1_data/ring/no_iac/no_iac_test_graphml",
        "pcn5": "...0_4_revisions/1_data/ring/pcns/5a_pcns_test_graphml",
        "pcn6": "...0_4_revisions/1_data/ring/pcns/6a_pcns_test_graphml",
        "pcn7": "...0_4_revisions/1_data/ring/pcns/7a_pcns_test_graphml"
    }
    output_dir = "...0_4_revisions/2_experiments/3_small_world"
    random_null = "degree_preserving"   # or "gnm" (density-only Erdos-Renyi null)
    n_random = 100                      # random null graphs per real graph
    n_lattice = 20                      # lattice null graphs per real graph
    niter_random = 10
    niter_lattice = 5
    seed = 0
    use_largest_component = False       # True = LCC-only sensitivity analysis
    num_processes = 4
    overwrite = False                   # True = discard existing CSVs and redo
    # ---------------------------

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
