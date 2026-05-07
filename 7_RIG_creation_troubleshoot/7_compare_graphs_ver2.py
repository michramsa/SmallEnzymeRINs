"""
Compare two NetworkX GraphML files for debugging/validation.

Checks performed (in order, coarse to fine):
  1. Byte-level identity (fast short-circuit if files are identical)
  2. Basic stats (node/edge counts, density, directedness, multigraph, self-loops)
  3. Set-level diff on nodes and edges (symmetric difference)
  4. Attribute comparison for shared nodes and shared edges
  5. Topological checks (degree sequence, connected components)

Configure inputs and options inside the `if __name__ == "__main__"` block.
"""

import hashlib
from pathlib import Path

import networkx as nx


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def file_hash(path):
    """Return MD5 hash of file contents."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_byte_identity(path_a, path_b):
    """Return True if the two files are byte-identical."""
    hash_a = file_hash(path_a)
    hash_b = file_hash(path_b)
    identical = hash_a == hash_b
    print("=" * 70)
    print("1. BYTE-LEVEL IDENTITY")
    print("=" * 70)
    print(f"  {Path(path_a).name}: {hash_a}")
    print(f"  {Path(path_b).name}: {hash_b}")
    print(f"  Identical: {identical}")
    print()
    return identical


def check_basic_stats(g_a, g_b, label_a, label_b):
    """Print and compare basic graph statistics."""
    print("=" * 70)
    print("2. BASIC STATS")
    print("=" * 70)

    def stats(g):
        return {
            "type": type(g).__name__,
            "directed": g.is_directed(),
            "multigraph": g.is_multigraph(),
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "density": nx.density(g) if g.number_of_nodes() > 1 else 0.0,
            "self_loops": nx.number_of_selfloops(g),
        }

    s_a = stats(g_a)
    s_b = stats(g_b)

    print(f"  {'metric':<15} {label_a:<20} {label_b:<20} match")
    print(f"  {'-' * 15} {'-' * 20} {'-' * 20} {'-' * 5}")
    all_match = True
    for key in s_a:
        match = s_a[key] == s_b[key]
        all_match = all_match and match
        print(f"  {key:<15} {str(s_a[key]):<20} {str(s_b[key]):<20} {match}")
    print()
    return all_match


def check_node_edge_sets(g_a, g_b, label_a, label_b, max_show=10):
    """Compare node and edge sets via symmetric difference."""
    print("=" * 70)
    print("3. NODE & EDGE SET COMPARISON")
    print("=" * 70)

    nodes_a = set(g_a.nodes())
    nodes_b = set(g_b.nodes())
    only_a_nodes = nodes_a - nodes_b
    only_b_nodes = nodes_b - nodes_a

    # For undirected graphs, normalize edge tuples so (u,v) == (v,u)
    def edge_set(g):
        if g.is_directed():
            return set(g.edges())
        return {tuple(sorted(e)) for e in g.edges()}

    edges_a = edge_set(g_a)
    edges_b = edge_set(g_b)
    only_a_edges = edges_a - edges_b
    only_b_edges = edges_b - edges_a

    print(f"  Nodes only in {label_a}: {len(only_a_nodes)}")
    if only_a_nodes:
        sample = list(only_a_nodes)[:max_show]
        print(f"    sample: {sample}")
    print(f"  Nodes only in {label_b}: {len(only_b_nodes)}")
    if only_b_nodes:
        sample = list(only_b_nodes)[:max_show]
        print(f"    sample: {sample}")
    print(f"  Edges only in {label_a}: {len(only_a_edges)}")
    if only_a_edges:
        sample = list(only_a_edges)[:max_show]
        print(f"    sample: {sample}")
    print(f"  Edges only in {label_b}: {len(only_b_edges)}")
    if only_b_edges:
        sample = list(only_b_edges)[:max_show]
        print(f"    sample: {sample}")
    print()

    sets_match = (
        not only_a_nodes
        and not only_b_nodes
        and not only_a_edges
        and not only_b_edges
    )
    return sets_match, edges_a & edges_b


def check_attributes(g_a, g_b, shared_edges, label_a, label_b, max_show=5):
    """Compare attributes for nodes and edges present in both graphs."""
    print("=" * 70)
    print("4. ATTRIBUTE COMPARISON (shared nodes & edges)")
    print("=" * 70)

    shared_nodes = set(g_a.nodes()) & set(g_b.nodes())
    node_mismatches = []
    for n in shared_nodes:
        attrs_a = dict(g_a.nodes[n])
        attrs_b = dict(g_b.nodes[n])
        if attrs_a != attrs_b:
            node_mismatches.append((n, attrs_a, attrs_b))

    print(f"  Shared nodes: {len(shared_nodes)}")
    print(f"  Nodes with attribute mismatches: {len(node_mismatches)}")
    for n, a, b in node_mismatches[:max_show]:
        print(f"    node {n!r}:")
        print(f"      {label_a}: {a}")
        print(f"      {label_b}: {b}")

    edge_mismatches = []
    for e in shared_edges:
        # Re-derive an edge accessor that works for both directed and undirected
        u, v = e
        attrs_a = dict(g_a.get_edge_data(u, v, default={}))
        attrs_b = dict(g_b.get_edge_data(u, v, default={}))
        if attrs_a != attrs_b:
            edge_mismatches.append((e, attrs_a, attrs_b))

    print(f"  Shared edges: {len(shared_edges)}")
    print(f"  Edges with attribute mismatches: {len(edge_mismatches)}")
    for e, a, b in edge_mismatches[:max_show]:
        print(f"    edge {e!r}:")
        print(f"      {label_a}: {a}")
        print(f"      {label_b}: {b}")
    print()

    return not node_mismatches and not edge_mismatches


def write_node_record(graph, output_path, include_node_attrs=True,
                      include_edge_attrs=True, sort_nodes=True):
    """
    Write a human-readable record of every node and its connections.

    Format per node:
        node_id [node_attrs]
            -> neighbor_id [edge_attrs]
            -> neighbor_id [edge_attrs]
            ...

    Parameters
    ----------
    graph : networkx.Graph
        The graph to record.
    output_path : str or Path
        Where to write the text file.
    include_node_attrs : bool
        Whether to include each node's attribute dict on its header line.
    include_edge_attrs : bool
        Whether to include each edge's attribute dict next to the neighbor.
    sort_nodes : bool
        If True, sort nodes alphabetically (helps when diffing two records).
        If False, preserve insertion order from the GraphML file.
    """
    nodes = sorted(graph.nodes()) if sort_nodes else list(graph.nodes())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"# Node-and-connections record\n")
        f.write(f"# nodes: {graph.number_of_nodes()}, "
                f"edges: {graph.number_of_edges()}, "
                f"directed: {graph.is_directed()}\n\n")

        for n in nodes:
            if include_node_attrs:
                node_attrs = dict(graph.nodes[n])
                f.write(f"{n}  {node_attrs}\n")
            else:
                f.write(f"{n}\n")

            # Sort neighbors for stable, diffable output
            neighbors = sorted(graph.neighbors(n))
            for nbr in neighbors:
                if include_edge_attrs:
                    edge_attrs = dict(graph.get_edge_data(n, nbr, default={}))
                    f.write(f"    -> {nbr}  {edge_attrs}\n")
                else:
                    f.write(f"    -> {nbr}\n")
            f.write("\n")

    print(f"  Wrote node record: {output_path}")


def check_topology(g_a, g_b, label_a, label_b):
    """Compare degree sequences and connected components."""
    print("=" * 70)
    print("5. TOPOLOGY")
    print("=" * 70)

    deg_a = sorted((d for _, d in g_a.degree()), reverse=True)
    deg_b = sorted((d for _, d in g_b.degree()), reverse=True)
    deg_match = deg_a == deg_b
    print(f"  Degree sequences match: {deg_match}")
    if not deg_match:
        print(f"    {label_a} top-10: {deg_a[:10]}")
        print(f"    {label_b} top-10: {deg_b[:10]}")

    # Use the appropriate components function for directed vs undirected
    if g_a.is_directed():
        comp_a = nx.number_weakly_connected_components(g_a)
        comp_b = nx.number_weakly_connected_components(g_b)
        comp_label = "weakly connected components"
    else:
        comp_a = nx.number_connected_components(g_a)
        comp_b = nx.number_connected_components(g_b)
        comp_label = "connected components"

    comp_match = comp_a == comp_b
    print(f"  {comp_label}: {label_a}={comp_a}, {label_b}={comp_b}, match={comp_match}")
    print()
    return deg_match and comp_match


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def compare_graphs(path_a, path_b, label_a="A", label_b="B",
                   stop_on_byte_match=False, max_show=10,
                   record_dir=None, include_node_attrs=True,
                   include_edge_attrs=True):
    """
    Run the full comparison pipeline.

    Parameters
    ----------
    path_a, path_b : str or Path
        Paths to the two GraphML files.
    label_a, label_b : str
        Short labels used in the printed output.
    stop_on_byte_match : bool
        If True, skip the structural checks when files are byte-identical.
        Set False to always run all checks (useful for sanity-checking the
        comparison code itself).
    max_show : int
        Max number of differing items to display per category.
    record_dir : str or Path or None
        If provided, write a human-readable node/connections record for each
        graph into this directory. If None, no records are written.
    include_node_attrs, include_edge_attrs : bool
        Passed through to write_node_record when record_dir is set.
    """
    print(f"\nComparing:\n  {label_a}: {path_a}\n  {label_b}: {path_b}\n")

    bytes_match = check_byte_identity(path_a, path_b)
    if bytes_match and stop_on_byte_match:
        print("Files are byte-identical; skipping structural checks.")
        return

    g_a = nx.read_graphml(path_a)
    g_b = nx.read_graphml(path_b)

    stats_match = check_basic_stats(g_a, g_b, label_a, label_b)
    sets_match, shared_edges = check_node_edge_sets(
        g_a, g_b, label_a, label_b, max_show=max_show
    )
    attrs_match = check_attributes(
        g_a, g_b, shared_edges, label_a, label_b, max_show=max_show
    )
    topo_match = check_topology(g_a, g_b, label_a, label_b)

    if record_dir is not None:
        print("=" * 70)
        print("6. NODE/CONNECTIONS RECORDS")
        print("=" * 70)
        record_dir = Path(record_dir)
        out_a = record_dir / f"{label_a}_node_record.txt"
        out_b = record_dir / f"{label_b}_node_record.txt"
        write_node_record(g_a, out_a, include_node_attrs, include_edge_attrs)
        write_node_record(g_b, out_b, include_node_attrs, include_edge_attrs)
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Byte-identical:        {bytes_match}")
    print(f"  Basic stats match:     {stats_match}")
    print(f"  Node/edge sets match:  {sets_match}")
    print(f"  Attributes match:      {attrs_match}")
    print(f"  Topology matches:      {topo_match}")
    print()
    overall = stats_match and sets_match and attrs_match and topo_match
    print(f"  OVERALL EQUIVALENT:    {overall}")
    print()


# ---------------------------------------------------------------------------
# Configure runs here
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Edit these paths and options to run a comparison.
    path_a = "/home/epi2melabs/AcrNetworks-clean-v2/7_RIG_creation_troubleshoot/new_6AVD_clean.graphml"
    path_b = "/home/epi2melabs/AcrNetworks-clean-v2/7_RIG_creation_troubleshoot/6AVD_clean.graphml"

    label_a = "new_6AVD"
    label_b = "old_6AVD"

    # Set False to always run structural checks even when bytes match.
    stop_on_byte_match = False

    # How many differing items to print per category (when there are diffs).
    max_show = 10

    # Where to write per-graph node/connections records.
    # Set to None to skip writing records.
    record_dir = "/home/epi2melabs/AcrNetworks-clean-v2/7_RIG_creation_troubleshoot"
    include_node_attrs = True
    include_edge_attrs = True

    compare_graphs(
        path_a=path_a,
        path_b=path_b,
        label_a=label_a,
        label_b=label_b,
        stop_on_byte_match=stop_on_byte_match,
        max_show=max_show,
        record_dir=record_dir,
        include_node_attrs=include_node_attrs,
        include_edge_attrs=include_edge_attrs,
    )
