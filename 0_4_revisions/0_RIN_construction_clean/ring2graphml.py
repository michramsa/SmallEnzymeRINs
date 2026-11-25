#!/usr/bin/env python3
"""
ring2graphml.py - Convert RING (Residue Interaction Network Generator, v4)
output files into simple, undirected NetworkX graphs and GraphML.

RING writes two tab-separated files per structure:
    <name>_ringNodes   one row per residue
    <name>_ringEdges   one row per interaction (a residue pair can have several)

Usage (CLI):
    # whole folder: every X_ringNodes + X_ringEdges pair -> OUTPUT_FOLDER/X.graphml
    python ring2graphml.py path/to/ring_output -o path/to/graphs
    # single pair
    python ring2graphml.py 1AC0.cif_ringNodes 1AC0.cif_ringEdges -o out/1AC0
    # options (work in both modes)
    --model 3                                  only model 3 of an NMR/MD file
    --priority SSBOND,IONIC,HBOND,VDW          which interaction wins on a residue pair

Usage (library):
    from ring2graphml import load_ring_graphs
    graphs = load_ring_graphs("x_ringNodes", "x_ringEdges")   # {model: nx.Graph}
    G = graphs[1]

Design
------
* Graph is a simple undirected nx.Graph: at most ONE edge between any two
  residues, no self-loops.
* Only residues are kept. Nodes whose Type is not "RES" (e.g. ligands) are
  dropped, and so is any edge that touches a node not in the kept set.
* When a residue pair has several RING interactions (e.g. HBOND and VDW), one
  is chosen for the edge: the type that comes first in the priority list
  (default: DEFAULT_PRIORITY below, strongest/most specific first); ties
  within the same type go to the shortest distance. The edge carries that
  interaction's attributes. The types that lost are only recorded in the
  `other_interactions` edge attribute (comma-separated; absent if none).
* Missing values ("nan", empty) are omitted from attributes, since GraphML has
  no null. Blank DSSP codes (coil) are written as "-".
* Node ids are RING's own ids, e.g. "A:509:_:CYS" (chain:resnum:insertion:resname).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imports import *

# Strongest / most specific first. Anything not listed ranks after these.
DEFAULT_PRIORITY = ["SSBOND", "METAL_ION", "IONIC", "HALOGEN", "PICATION",
                    "PIPISTACK", "PIHBOND", "HBOND", "IAC", "VDW"]

NODE_FLOAT = {"Bfactor_CA", "x", "y", "z"}
NODE_INT = {"Position", "Degree"}
EDGE_FLOAT = {"Distance", "Angle"}
NODE_RENAME = {"Chain": "chain", "Position": "position", "Residue": "residue",
               "Type": "type", "Degree": "degree"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _clean(value):
    """Return None for empty / nan values, else the raw string."""
    if value is None or value.strip() == "" or value.strip().lower() == "nan":
        return None
    return value


def _convert(key, value, floats, ints):
    value = _clean(value)
    if value is None:
        return None
    if key in floats:
        try:
            f = float(value)
        except ValueError:
            return None
        return None if math.isnan(f) else f
    if key in ints:
        return int(value)
    return value


def _read_tsv(path):
    # Explicit tab delimiter; do NOT strip fields (DSSP can be " ").
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = [h.strip() for h in next(reader)]
        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue
            row = row + [""] * (len(header) - len(row))  # pad trailing empties
            yield dict(zip(header, row))


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def parse_nodes(path):
    """Yield (model, node_id, attrs) for each residue row of a ringNodes file."""
    for row in _read_tsv(path):
        if row.get("Type", "RES").strip() != "RES":
            continue                                  # drop ligands / non-residues
        attrs = {}
        for k, v in row.items():
            if k in ("NodeId", "Model"):
                continue
            if k == "Dssp":
                attrs["dssp"] = v.strip() or "-"
                continue
            conv = _convert(k, v, NODE_FLOAT, NODE_INT)
            if conv is not None:
                attrs[NODE_RENAME.get(k, k)] = conv
        yield int(row.get("Model") or 1), row["NodeId"], attrs


def parse_edges(path):
    """Yield (model, node1, node2, attrs) for each row of a ringEdges file."""
    for row in _read_tsv(path):
        full = row["Interaction"]                     # e.g. HBOND:MC_SC
        itype, _, subtype = full.partition(":")
        attrs = {"interaction": itype, "interaction_full": full}
        if subtype:
            attrs["subtype"] = subtype                # MC_MC / MC_SC / SC_SC / ...
        for k in ("Distance", "Angle"):
            conv = _convert(k, row.get(k, ""), EDGE_FLOAT, set())
            if conv is not None:
                attrs[k.lower()] = conv
        for k in ("Atom1", "Atom2", "Donor", "Positive", "Cation", "Orientation"):
            v = _clean(row.get(k, ""))
            if v is not None:
                attrs[k.lower()] = v
        yield int(row.get("Model") or 1), row["NodeId1"], row["NodeId2"], attrs


# --------------------------------------------------------------------------
# graph building
# --------------------------------------------------------------------------
def load_ring_graphs(nodes_path, edges_path, priority=None):
    """Return {model_number: nx.Graph}, one edge max per residue pair."""
    priority = list(priority or DEFAULT_PRIORITY)
    rank = {t: i for i, t in enumerate(priority)}

    def sort_key(attrs):
        return (rank.get(attrs["interaction"], len(rank)),
                attrs.get("distance", float("inf")))

    graphs = {}
    for model, nid, attrs in parse_nodes(nodes_path):
        graphs.setdefault(model, nx.Graph(model=model)).add_node(nid, **attrs)

    # Gather all candidate interactions per residue pair first.
    pairs = {}   # (model, frozenset({a, b})) -> [attrs, ...]
    for model, a, b, attrs in parse_edges(edges_path):
        G = graphs.get(model)
        if G is None or a == b or a not in G or b not in G:
            continue                                  # ligand / unknown node / self-loop
        pairs.setdefault((model, frozenset((a, b))), []).append((a, b, attrs))

    for (model, _), items in pairs.items():
        items.sort(key=lambda t: sort_key(t[2]))
        a, b, best = items[0]
        best = dict(best)
        others = []
        for _, _, d in items[1:]:
            if d["interaction"] != best["interaction"] and d["interaction"] not in others:
                others.append(d["interaction"])
        if others:
            best["other_interactions"] = ",".join(others)
        graphs[model].add_edge(a, b, **best)

    return graphs


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
STRUCT_EXTS = {".cif", ".mmcif", ".pdb", ".ent"}


def find_pairs(folder):
    """Return ({stem: (nodes_path, edges_path)}, [warnings]) for a RING output folder.

    '1AC0.cif_ringNodes' + '1AC0.cif_ringEdges' -> stem '1AC0'.
    """
    nodes, edges = {}, {}
    for fn in sorted(os.listdir(folder)):
        full = os.path.join(folder, fn)
        if not os.path.isfile(full):
            continue
        for suffix, store in (("_ringNodes", nodes), ("_ringEdges", edges)):
            if fn.endswith(suffix):
                base = fn[:-len(suffix)]
                stem, ext = os.path.splitext(base)
                store[base if ext.lower() not in STRUCT_EXTS else stem] = full
    warnings = [f"no _ringEdges file for '{k}' - skipped" for k in sorted(nodes.keys() - edges.keys())]
    warnings += [f"no _ringNodes file for '{k}' - skipped" for k in sorted(edges.keys() - nodes.keys())]
    return {k: (nodes[k], edges[k]) for k in sorted(nodes.keys() & edges.keys())}, warnings


def convert_one(nodes_path, edges_path, prefix, model=None, priority=None):
    """Convert one nodes/edges pair; write <prefix>[_modelN].graphml. Returns number of files written."""
    graphs = load_ring_graphs(nodes_path, edges_path, priority=priority)
    if model is not None:
        if model not in graphs:
            raise ValueError(f"model {model} not found; available: {sorted(graphs)}")
        graphs = {model: graphs[model]}
    for m, G in sorted(graphs.items()):
        path = f"{prefix}.graphml" if len(graphs) == 1 else f"{prefix}_model{m}.graphml"
        nx.write_graphml(G, path)
        print(f"{os.path.basename(path)}: model {m}, {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return len(graphs)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Convert RING nodes/edges files to simple-graph GraphML.",
        epilog="Folder mode:  ring2graphml.py INPUT_FOLDER -o OUTPUT_FOLDER\n"
               "File mode:    ring2graphml.py X_ringNodes X_ringEdges -o OUTPUT_PREFIX",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="folder containing the _ringNodes/_ringEdges files, "
                                 "or a single <name>_ringNodes file")
    p.add_argument("edges", nargs="?", help="<name>_ringEdges file (file mode only)")
    p.add_argument("-o", "--output",
                   help="folder mode: output folder, created if needed (default: <input>_graphml); "
                        "file mode: output path prefix (default: derived from the nodes file name)")
    p.add_argument("--model", type=int, help="only export this model number")
    p.add_argument("--priority", help="comma-separated interaction types, highest priority first "
                                      f"(default: {','.join(DEFAULT_PRIORITY)})")
    args = p.parse_args(argv)
    priority = args.priority.split(",") if args.priority else None

    if os.path.isdir(args.input):                                    # ---- folder mode
        if args.edges:
            p.error("give either a folder or a nodes/edges file pair, not both")
        outdir = args.output or args.input.rstrip("/\\") + "_graphml"
        pairs, warnings = find_pairs(args.input)
        for w in warnings:
            print("warning:", w, file=sys.stderr)
        if not pairs:
            sys.exit(f"No matching _ringNodes/_ringEdges pairs found in {args.input}")
        os.makedirs(outdir, exist_ok=True)
        done, failed = 0, []
        for stem, (npath, epath) in pairs.items():
            try:
                convert_one(npath, epath, os.path.join(outdir, stem), args.model, priority)
                done += 1
            except Exception as exc:                                  # keep going on a bad file
                failed.append(stem)
                print(f"error: {stem}: {exc}", file=sys.stderr)
        print(f"\nConverted {done}/{len(pairs)} structures -> {outdir}")
        if failed:
            sys.exit(f"Failed: {', '.join(failed)}")
    else:                                                             # ---- file mode
        if not args.edges:
            p.error("file mode needs both a _ringNodes and a _ringEdges file")
        prefix = args.output
        if not prefix:
            base = os.path.basename(args.input)
            prefix = base[:-len("_ringNodes")] if base.endswith("_ringNodes") else base
        if os.path.dirname(prefix):
            os.makedirs(os.path.dirname(prefix), exist_ok=True)
        try:
            convert_one(args.input, args.edges, prefix, args.model, priority)
        except ValueError as exc:
            sys.exit(str(exc))


if __name__ == "__main__":
    main()
