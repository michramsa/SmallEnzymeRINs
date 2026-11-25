"""
Efficiency vs. rewiring probability for one residue interaction network (RIN),
in the spirit of Figure 1 of Latora & Marchiori (2001), "Efficient behavior of
small-world networks", Phys. Rev. Lett. 87, 198701.

The starting point (p = 0) is the ORIGINAL RIN, not an ideal ring lattice. The
curves therefore show how global and local efficiency change as the real
contact network is progressively randomised.

Two rewiring experiments are run on the same graph
    watts_strogatz     : each edge is rewired with probability p. One endpoint
                         (chosen at random) is kept and the other end is moved
                         to a uniformly random node, never creating a self-loop
                         or a duplicate edge. Edge count is preserved; degrees
                         are NOT preserved; the graph may fragment.
    degree_preserving  : Maslov-Sneppen double edge swaps,
                         (a-b, c-d) -> (a-d, c-b), which keep every node's
                         degree. The number of successful swaps is
                             round(p * n_edges * swaps_per_edge)
                         so p = 1 is a fully randomised, degree-matched graph
                         (at the default swaps_per_edge = 10) and p -> 0 leaves
                         the RIN essentially untouched. Swaps do not enforce
                         connectivity, so the graph may fragment.

Efficiency measures (unweighted, undirected graphs; Latora & Marchiori 2001)
    E_glob : mean of 1/d_ij over all node pairs (1/d_ij = 0 across components,
             so fragmentation is counted as lost efficiency; no LCC needed).
    E_loc  : mean over nodes i of E_glob of the subgraph induced by the
             neighbours of i (node i removed). Nodes with < 2 neighbours
             contribute 0.
Both are computed with networkx (nx.global_efficiency, nx.local_efficiency),
exactly as in small_world_efficiency_ver3.py.

For every p in a log-spaced grid, n_reps independent rewired graphs are made.

Reference graphs (same two sweeps, each starting from its own graph)
    Each figure shows up to six lines: E_glob and E_loc for
      protein (solid, filled markers)
          the original RIN.
      random  (dashed, open markers; include_random=True)
          a random graph that keeps the RIN's degree sequence
          (random_kind="degree_preserving", the default): a fresh copy of the
          RIN is fully randomised with Maslov-Sneppen swaps for every
          replicate, then rewired like the protein. random_kind="gnm" uses a
          G(n, m) random graph instead, which keeps only n and m.
      lattice (dotted, half-filled markers; include_lattice=True)
          a ring lattice with the RIN's n and exactly its m edges: each node is
          joined to its d = m // n nearest neighbours on each side, and the
          remaining m - n*d edges join nodes at distance d + 1, spread evenly
          around the ring. Degrees therefore differ by at most 2 (they cannot
          match the RIN's degrees). This is the p = 0 start of Fig. 1 in
          Latora & Marchiori; its curves rise (E_glob) and fall (E_loc) toward
          the random level as p grows.
    The random lines are roughly flat for degree-preserving swaps, because the
    graph is already randomised. Under watts_strogatz rewiring the random line
    can drift slightly, since that move does not keep degrees. Neither reference
    is forced to be connected.

Y axis (as in Fig. 1 of Latora & Marchiori)
    By default (normalize=False) the raw efficiencies are plotted on a y axis
    from 0 to 1. This is the paper's scale: networkx divides E by N(N-1), which
    is the same as dividing by the efficiency of the ideal graph with an edge
    between every pair of nodes, so 0 <= E <= 1. With normalize=True the curves
    are instead E(p) / E(0), where E(0) is the value for the original RIN (both
    start at 1; a horizontal dotted line at 1 marks the original RIN).

Crossing point
    The p at which the mean E_glob and E_loc curves cross is found by linear
    interpolation of the curves in log10(p), marked on the figure, printed, and
    saved to a CSV. It is computed on the plotted values, so it depends on
    `normalize`. If the curves do not cross inside the p range, none is drawn.

Outputs (output_dir)
    <label>_efficiency_sweep_raw.csv       one row per rewired graph
                                           (column "system": protein / random /
                                           lattice)
    <label>_efficiency_sweep_summary.csv   mean / SD per system, experiment, p
    <label>_efficiency_crossings.csv       protein crossing p and efficiency,
                                           per experiment
    <label>_efficiency_watts_strogatz.<fmt>        Figure, one file
    <label>_efficiency_degree_preserving.<fmt>     Figure, one file

Reproducible: every rewired graph gets its own random stream seeded from
(seed, system, experiment, p, replicate). The protein streams do not depend on
the reference options, so the protein curves match those of efficiency_ver3.py
for the same seed.

References
    Latora V, Marchiori M (2001). Phys Rev Lett 87:198701.
    Watts DJ, Strogatz SH (1998). Nature 393:440-442.
    Maslov S, Sneppen K (2002). Science 296:910-913.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imports import *

import numpy as np
from matplotlib import ticker

from figure_style_ver2 import (apply_style, save_fig, FIG_SIZE, COLORS, COLOR_CYCLE,
                                   MARKER_CYCLE, LAYOUT, FONT)

EXPERIMENTS = {
    "watts_strogatz": {
        "xlabel": "Edge rewiring probability, $p$",
    },
    "degree_preserving": {
        "xlabel": "Degree-preserving swap fraction, $p$",
    },
}

METRICS = [
    # column, legend label, colour, marker
    ("E_glob", r"$E_{glob}$", COLORS["blue"], "o"),
    ("E_loc", r"$E_{loc}$", COLORS["orange"], "s"),
]

# Colour identifies the metric; line style, marker shape and marker fill
# identify the graph the sweep starts from.
#   marker       : None = use the metric's marker from METRICS (protein);
#                  otherwise one shape for both metrics of that graph
#   marker_scale : multiplies LAYOUT["marker_size"]
#   alpha        : opacity of line, markers and error bars (1 = opaque;
#                  lower = more transparent)
SYSTEM_STYLES = {
    "protein": {"name": "protein", "linestyle": "-", "marker": None,
                "marker_scale": 1.0, "alpha": 1.0, "marker_kw": {}},
    "random":  {"name": "random", "linestyle": "--", "marker": "D",
                "marker_scale": 0.6, "alpha": 0.5,
                "marker_kw": {"markerfacecolor": "white"}},
    "lattice": {"name": "lattice", "linestyle": ":", "marker": "^",
                "marker_scale": 0.7, "alpha": 0.5,
                "marker_kw": {"fillstyle": "left",
                              "markerfacecoloralt": "white"}},
}


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


# ----------------------------------------------------------------------------
# Regular lattice reference
# ----------------------------------------------------------------------------
def ring_lattice(n, m):
    """Ring lattice with exactly n nodes (labelled 0..n-1) and m edges.

    Every node is joined to its d = m // n nearest neighbours on each side
    (n*d edges). The remaining r = m - n*d edges join node i to node i+d+1 for r
    nodes spread evenly around the ring, so node degrees are 2d, 2d+1 or 2d+2.
    """
    d, r = divmod(m, n)
    longest = d + 1 if r else d
    if longest > (n - 1) // 2:
        raise ValueError(f"A ring lattice with n={n} and m={m} edges needs "
                         f"neighbours at distance {longest}, which is too dense "
                         "for a ring; set include_lattice=False")
    L = nx.Graph()
    L.add_nodes_from(range(n))
    for i in range(n):
        for step in range(1, d + 1):
            L.add_edge(i, (i + step) % n)
    for k in range(r):
        i = (k * n) // r
        L.add_edge(i, (i + d + 1) % n)
    assert L.number_of_edges() == m
    return L


# ----------------------------------------------------------------------------
# Rewiring
# ----------------------------------------------------------------------------
def rewire_watts_strogatz(G, p, rng):
    """Watts-Strogatz style rewiring of the original graph G.

    Each edge is visited once; with probability p it is rewired by keeping one
    randomly chosen endpoint and moving the other to a uniformly random node
    that is not already a neighbour. Edge count is preserved."""
    H = G.copy()
    nodes = list(H.nodes())
    n = len(nodes)
    for u, v in list(G.edges()):
        if rng.random() >= p:
            continue
        if rng.random() < 0.5:           # u is the endpoint that stays
            u, v = v, u
        if H.degree(u) >= n - 1:         # u is already linked to every node
            continue
        while True:
            w = rng.choice(nodes)
            if w != u and not H.has_edge(u, w):   # also excludes w == v
                break
        H.remove_edge(u, v)
        H.add_edge(u, w)
    return H


def rewire_degree_preserving(G, p, rng, swaps_per_edge):
    """Degree-preserving double edge swaps of the original graph G.

    The number of successful swaps is round(p * m * swaps_per_edge). If the
    graph is too constrained to finish them all, the swaps already made are
    kept."""
    H = G.copy()
    m = H.number_of_edges()
    n_swaps = int(round(p * m * swaps_per_edge))
    if n_swaps < 1 or H.number_of_nodes() < 4 or m < 2:
        return H
    try:
        nx.double_edge_swap(H, nswap=n_swaps, max_tries=100 * n_swaps + 1000,
                            seed=rng)
    except nx.NetworkXAlgorithmError:
        pass  # swaps done so far stay in H (it is modified in place)
    return H


# ----------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------
_G = None        # original graph, set once per worker process
_LATTICE = None  # ring lattice with the same n and m (or None)


def _init_worker(G, lattice=None):
    global _G, _LATTICE
    _G, _LATTICE = G, lattice


def _run_task(task, swaps_per_edge, seed, random_kind):
    """One rewired graph -> one row of efficiencies.

    system 'protein': rewire the original RIN.
    system 'random' : build a random graph (degree-preserving randomisation of
                      the RIN, or G(n, m)), then rewire it.
    system 'lattice': rewire the ring lattice with the RIN's n and m."""
    system, experiment, p, rep = task
    if system == "protein":   # same stream as efficiency_ver3.py
        rng = random.Random(f"{seed}|{experiment}|{p!r}|{rep}")
        start = _G
    elif system == "random":
        rng = random.Random(f"{seed}|random|{experiment}|{p!r}|{rep}")
        if random_kind == "degree_preserving":
            start = rewire_degree_preserving(_G, 1.0, rng, swaps_per_edge)
        elif random_kind == "gnm":
            start = nx.gnm_random_graph(_G.number_of_nodes(),
                                        _G.number_of_edges(), seed=rng)
        else:
            raise ValueError(f"Unknown random_kind: {random_kind!r}")
    elif system == "lattice":
        rng = random.Random(f"{seed}|lattice|{experiment}|{p!r}|{rep}")
        start = _LATTICE
    else:
        raise ValueError(f"Unknown system: {system!r}")

    if experiment == "watts_strogatz":
        H = rewire_watts_strogatz(start, p, rng)
    elif experiment == "degree_preserving":
        H = rewire_degree_preserving(start, p, rng, swaps_per_edge)
    else:
        raise ValueError(f"Unknown experiment: {experiment!r}")

    return {
        "system": system, "experiment": experiment, "p": p, "rep": rep,
        "n_edges": H.number_of_edges(),
        "n_components": nx.number_connected_components(H),
        "E_glob": nx.global_efficiency(H),
        "E_loc": nx.local_efficiency(H),
    }


# ----------------------------------------------------------------------------
# Sweep and summary
# ----------------------------------------------------------------------------
def run_sweep(G, p_values, n_reps, swaps_per_edge, seed, num_processes,
              include_random, include_lattice, random_kind, lattice):
    """All (system, experiment, p, replicate) tasks -> DataFrame of raw results."""
    systems = (["protein"] + (["random"] if include_random else [])
               + (["lattice"] if include_lattice else []))
    tasks = [(sys_, exp, float(p), rep)
             for sys_ in systems for exp in EXPERIMENTS
             for p in p_values for rep in range(n_reps)]
    worker = partial(_run_task, swaps_per_edge=swaps_per_edge, seed=seed,
                     random_kind=random_kind)
    rows = []

    if num_processes is None or num_processes > 1:
        n_workers = num_processes or mp.cpu_count()
        with mp.Pool(processes=n_workers, initializer=_init_worker,
                     initargs=(G, lattice)) as pool:
            results = pool.imap_unordered(worker, tasks, chunksize=4)
            if tqdm is not None:
                results = tqdm(results, total=len(tasks), desc="rewiring")
            rows.extend(results)
    else:
        _init_worker(G, lattice)
        results = map(worker, tasks)
        if tqdm is not None:
            results = tqdm(results, total=len(tasks), desc="rewiring")
        rows.extend(results)

    return (pd.DataFrame(rows)
            .sort_values(["system", "experiment", "p", "rep"])
            .reset_index(drop=True))


def summarise(raw):
    """Mean and SD (ddof=1) of each metric per system, experiment and p."""
    rows = []
    for (system, exp, p), g in raw.groupby(["system", "experiment", "p"]):
        row = {"system": system, "experiment": exp, "p": p, "n_reps": len(g),
               "n_components_mean": g["n_components"].mean()}
        for metric, *_ in METRICS:
            row[f"{metric}_mean"] = g[metric].mean()
            row[f"{metric}_sd"] = g[metric].std(ddof=1) if len(g) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures (one figure per file)
# ----------------------------------------------------------------------------
def set_decade_ticks(ax):
    """Log x-axis with one labelled tick per power of ten (0.001, 0.01, 0.1, 1),
    which are equally spaced on a log axis, plain decimal labels (no scientific
    notation), and no minor ticks."""
    lo, hi = ax.get_xlim()
    first, last = int(np.ceil(np.log10(lo))), int(np.floor(np.log10(hi)))
    ticks = [10.0 ** k for k in range(first, last + 1)]
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: np.format_float_positional(x, trim="-")))
    ax.xaxis.set_minor_locator(ticker.NullLocator())


def plotted_curves(summary, system, experiment, baseline, normalize):
    """Mean and SD of E_glob and E_loc for one system ('protein' or 'random')
    and experiment, sorted by p, scaled exactly as they are plotted (raw, or
    divided by the ORIGINAL RIN's value, for both systems)."""
    df = summary[(summary["system"] == system)
                 & (summary["experiment"] == experiment)].sort_values("p")
    out = pd.DataFrame({"p": df["p"].to_numpy()})
    for metric, *_ in METRICS:
        scale = baseline[metric] if normalize else 1.0
        out[f"{metric}_mean"] = df[f"{metric}_mean"].to_numpy() / scale
        out[f"{metric}_sd"] = df[f"{metric}_sd"].fillna(0).to_numpy() / scale
    return out


def find_crossings(curves):
    """Where the mean E_glob and E_loc curves cross.

    Returns a list of (p, efficiency) pairs, one per sign change of
    E_glob - E_loc between neighbouring grid points, interpolated linearly in
    log10(p). Empty if the curves do not cross inside the p range."""
    logp = np.log10(curves["p"].to_numpy())
    g = curves["E_glob_mean"].to_numpy()
    d = g - curves["E_loc_mean"].to_numpy()
    crossings = []
    for i in range(len(d) - 1):
        if d[i] * d[i + 1] < 0:
            t = d[i] / (d[i] - d[i + 1])
            crossings.append((10.0 ** (logp[i] + t * (logp[i + 1] - logp[i])),
                              g[i] + t * (g[i + 1] - g[i])))
    return crossings


def plot_experiment(curves, crossings, experiment, normalize, out_path,
                    fig_size_key, reference_curves=None, show_crossing=True):
    """E_glob and E_loc versus p for one experiment, saved as its own file.

    curves           : protein curves (solid lines)
    reference_curves : optional dict {'random': curves, 'lattice': curves},
                       drawn with the styles in SYSTEM_STYLES
    show_crossing    : True = mark and label the protein crossing point(s) with a
                       dashed drop line; False = leave them off the figure (the
                       crossing is still printed and saved to the CSV)."""
    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE[fig_size_key])

    series = [("protein", curves)] + list((reference_curves or {}).items())
    for system, cv in series:
        style = SYSTEM_STYLES[system]
        for metric, label, color, marker in METRICS:
            ax.errorbar(cv["p"], cv[f"{metric}_mean"],
                        yerr=cv[f"{metric}_sd"], color=color,
                        marker=style["marker"] or marker,
                        linestyle=style["linestyle"],
                        alpha=style["alpha"],
                        label=f"{label}, {style['name']}" if reference_curves
                        else label,
                        markersize=LAYOUT["marker_size"] * style["marker_scale"],
                        linewidth=LAYOUT["line_width"],
                        capsize=LAYOUT["cap_size"], **style["marker_kw"])

    ax.set_xscale("log")
    set_decade_ticks(ax)
    lo, hi = ax.get_xlim()

    if normalize:
        ax.axhline(1.0, color=COLORS["mid_gray"], linestyle=":", linewidth=1.0)
        ax.set_ylabel(r"$E(p)\,/\,E(0)$")
    else:   # Fig. 1 of Latora & Marchiori: efficiency from 0 to 1
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylabel("Efficiency")

    for p_c, e_c in (crossings if show_crossing else []):
        ax.plot([p_c, p_c], [0 if not normalize else ax.get_ylim()[0], e_c],
                color=COLORS["dark_gray"], linestyle="--", linewidth=0.75)
        ax.plot(p_c, e_c, marker="o", markersize=LAYOUT["marker_size"] + 2,
                markerfacecolor="white", markeredgecolor=COLORS["dark_gray"],
                markeredgewidth=1.0, linestyle="none", zorder=5)
        right_side = (np.log10(p_c) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)) > 0.6
        ax.annotate(f"$p$ = {p_c:.3g}", xy=(p_c, e_c),
                    xytext=(-8 if right_side else 8, 10),
                    textcoords="offset points",
                    ha="right" if right_side else "left",
                    fontsize=FONT["size_annot"],
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="none", alpha=0.85))

    ax.set_xlabel(EXPERIMENTS[experiment]["xlabel"])
    if not reference_curves:
        ax.legend(loc=LAYOUT["legend_loc"], framealpha=LAYOUT["legend_framealpha"])
    else:   # one column per graph (E_glob over E_loc), below the axis
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
                  ncol=len(series), frameon=False, columnspacing=1.2,
                  handlelength=2.6)

    save_fig(fig, out_path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main(graph_path, output_dir, label=None, p_min=1e-3, p_max=1.0, n_p=25,
         p_values=None, n_reps=20, swaps_per_edge=10, normalize=False, seed=0,
         num_processes=4, fig_size_key="half_page", fig_format="tiff",
         include_random=True, include_lattice=True,
         random_kind="degree_preserving", show_crossing=True):
    """
    graph_path     : GraphML file of the protein's RIN (the p = 0 graph)
    output_dir     : where CSVs and figures are written
    label          : name used in output file names (None = graph file stem)
    p_min, p_max   : ends of the log-spaced p grid (p_min > 0)
    n_p            : number of p values in the grid
    p_values       : explicit list of p values; overrides p_min/p_max/n_p
    n_reps         : rewired graphs per p and per experiment
    swaps_per_edge : degree-preserving swaps per edge at p = 1
    normalize      : False = raw efficiencies, y axis 0 to 1 (Latora-Marchiori
                     Fig. 1); True = plot E(p) / E(0)
    seed           : base seed (each rewired graph gets its own stream)
    num_processes  : worker processes (None = all CPUs, 1 = run serially)
    fig_size_key   : key into FIG_SIZE from the figure style file
    fig_format     : 'tiff' or 'eps' (PLOS-accepted formats)
    include_random : True = also sweep a random reference graph (dashed lines)
    include_lattice: True = also sweep a ring lattice with the RIN's n and m
                     (dotted lines)
    random_kind    : 'degree_preserving' = random graph with the RIN's degree
                     sequence (default); 'gnm' = G(n, m), keeps only n and m
    show_crossing  : True = draw the crossing point, its dashed drop line and
                     the p label on the figures; False = omit them (the value
                     is still printed in the terminal and saved to the CSV)
    """
    graph_path = Path(graph_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = label or graph_path.stem

    G = load_graphml(graph_path)
    if G.number_of_edges() == 0:
        raise ValueError(f"{graph_path} has no edges")

    if p_values is None:
        p_values = np.logspace(np.log10(p_min), np.log10(p_max), n_p)
    p_values = [float(p) for p in p_values]
    if min(p_values) <= 0 or max(p_values) > 1:
        raise ValueError("p values must lie in (0, 1]")

    baseline = {"E_glob": nx.global_efficiency(G),
                "E_loc": nx.local_efficiency(G)}
    print(f"[{label}] n={G.number_of_nodes()}, m={G.number_of_edges()}, "
          f"components={nx.number_connected_components(G)}, "
          f"E_glob(0)={baseline['E_glob']:.4f}, E_loc(0)={baseline['E_loc']:.4f}")
    if normalize and min(baseline.values()) <= 0:
        raise ValueError("A baseline efficiency is 0 (no edges or no triangles); "
                         "set normalize=False")

    if random_kind not in ("degree_preserving", "gnm"):
        raise ValueError(f"random_kind must be 'degree_preserving' or 'gnm', "
                         f"got {random_kind!r}")
    lattice = (ring_lattice(G.number_of_nodes(), G.number_of_edges())
               if include_lattice else None)

    raw = run_sweep(G, p_values, n_reps, swaps_per_edge, seed, num_processes,
                    include_random, include_lattice, random_kind, lattice)
    summary = summarise(raw)

    raw_path = output_dir / f"{label}_efficiency_sweep_raw.csv"
    summary_path = output_dir / f"{label}_efficiency_sweep_summary.csv"
    raw.to_csv(raw_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(summary_path, index=False)

    crossing_rows = []
    for experiment in EXPERIMENTS:
        curves = plotted_curves(summary, "protein", experiment, baseline,
                                normalize)
        reference_curves = {}
        if include_random:
            reference_curves["random"] = plotted_curves(
                summary, "random", experiment, baseline, normalize)
        if include_lattice:
            reference_curves["lattice"] = plotted_curves(
                summary, "lattice", experiment, baseline, normalize)
        crossings = find_crossings(curves)
        if crossings:
            for p_c, e_c in crossings:
                print(f"[{label}] {experiment}: E_glob and E_loc cross at "
                      f"p = {p_c:.4g} (efficiency = {e_c:.3f})")
                crossing_rows.append({"experiment": experiment, "p_cross": p_c,
                                      "efficiency_at_cross": e_c})
        else:
            print(f"[{label}] {experiment}: E_glob and E_loc do not cross "
                  f"for p in [{min(p_values):.3g}, {max(p_values):.3g}]")
            crossing_rows.append({"experiment": experiment, "p_cross": np.nan,
                                  "efficiency_at_cross": np.nan})

        out_path = output_dir / f"{label}_efficiency_{experiment}.{fig_format}"
        plot_experiment(curves, crossings, experiment, normalize, out_path,
                        fig_size_key, reference_curves=reference_curves,
                        show_crossing=show_crossing)

    pd.DataFrame(crossing_rows).to_csv(
        output_dir / f"{label}_efficiency_crossings.csv", index=False)

    print(f"Saved results to: {output_dir}")
    return raw, summary


if __name__ == "__main__":
    # ---- Edit these values ----
    graph_path = "...1BOR_clean.graphml"
    output_dir = "...0_4_revisions/0_new_data/efficiency"
    label = None                # None = use the graph file name
    p_min = 1e-3                # smallest p (log grid)
    p_max = 1.0                 # largest p
    n_p = 25                    # number of p values
    p_values = None             # or an explicit list, e.g. [0.001, 0.01, 0.1, 1]
    n_reps = 20                 # rewired graphs per p, per experiment
    swaps_per_edge = 10         # degree-preserving swaps per edge at p = 1
    normalize = False           # False = raw efficiency, y from 0 to 1 (paper Fig. 1)
                                # True  = E(p) / E(0)
    seed = 0
    num_processes = 4           # 1 = serial, None = all CPUs
    fig_size_key = "half_page"
    fig_format = "png"         # 'tiff' or 'eps'
    include_random = True       # dashed lines: random reference graph
    include_lattice = True      # dotted lines: ring lattice with the same n and m
    random_kind = "degree_preserving"   # or "gnm" (keeps only n and m)
    show_crossing = False        # True = crossing point, dashed line and p label
                                # on the figure; False = hide them (still printed
                                # in the terminal and saved to the CSV)
    # ---------------------------

    main(
        graph_path=graph_path,
        output_dir=output_dir,
        label=label,
        p_min=p_min,
        p_max=p_max,
        n_p=n_p,
        p_values=p_values,
        n_reps=n_reps,
        swaps_per_edge=swaps_per_edge,
        normalize=normalize,
        seed=seed,
        num_processes=num_processes,
        fig_size_key=fig_size_key,
        fig_format=fig_format,
        include_random=include_random,
        include_lattice=include_lattice,
        random_kind=random_kind,
        show_crossing=show_crossing,
    )
