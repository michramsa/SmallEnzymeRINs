"""
check_graphml_connectivity.py

Iterates through a folder of GraphML files and checks whether each graph
is connected. For directed graphs, "weak connectivity" is checked (i.e.
connectivity if edge direction is ignored), since this is usually what
people mean when checking RIN/structural graphs for disconnection.

Outputs a summary to the console and optionally writes a CSV report.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

import os
import csv
import networkx as nx


def check_graph_connectivity(filepath):
    """
    Load a single GraphML file and determine if it's connected.

    Returns a dict with results, or None if the file could not be read.
    """
    try:
        G = nx.read_graphml(filepath)
    except Exception as e:
        return {
            "file": os.path.basename(filepath),
            "error": str(e),
            "n_nodes": None,
            "n_edges": None,
            "is_connected": None,
            "n_components": None,
        }

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        is_connected = None
        n_components = 0
    elif G.is_directed():
        is_connected = nx.is_weakly_connected(G)
        n_components = nx.number_weakly_connected_components(G)
    else:
        is_connected = nx.is_connected(G)
        n_components = nx.number_connected_components(G)

    return {
        "file": os.path.basename(filepath),
        "error": None,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "is_connected": is_connected,
        "n_components": n_components,
    }


def scan_folder(input_dir, output_csv=None, verbose=True):
    """
    Iterate through all .graphml files in input_dir, check connectivity,
    print a summary, and optionally write results to output_csv.
    """
    graphml_files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith(".graphml")
    )

    if not graphml_files:
        print(f"No .graphml files found in: {input_dir}")
        return []

    results = []
    for fname in graphml_files:
        fpath = os.path.join(input_dir, fname)
        result = check_graph_connectivity(fpath)
        results.append(result)

        if verbose:
            if result["error"]:
                print(f"[ERROR]   {result['file']}: {result['error']}")
            elif result["is_connected"] is None:
                print(f"[EMPTY]   {result['file']}: 0 nodes")
            elif result["is_connected"]:
                print(
                    f"[OK]      {result['file']}: connected "
                    f"({result['n_nodes']} nodes, {result['n_edges']} edges)"
                )
            else:
                print(
                    f"[DISCONN] {result['file']}: "
                    f"{result['n_components']} components "
                    f"({result['n_nodes']} nodes, {result['n_edges']} edges)"
                )

    disconnected = [r for r in results if r["is_connected"] is False]
    errored = [r for r in results if r["error"] is not None]

    print("\n--- Summary ---")
    print(f"Total files scanned: {len(results)}")
    print(f"Disconnected graphs: {len(disconnected)}")
    print(f"Files with read errors: {len(errored)}")

    if output_csv:
        fieldnames = ["file", "error", "n_nodes", "n_edges", "is_connected", "n_components"]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to: {output_csv}")

    return results


def main(input_dir, output_csv=None, verbose=True):
    scan_folder(input_dir, output_csv=output_csv, verbose=verbose)


if __name__ == "__main__":
    # --- Edit these variables to configure the run ---
    input_dir = DATA_DIR_3 / "ref" / "RINs"   # folder containing .graphml files
    output_csv = None       # set to None to skip writing CSV
    verbose = True                                # print per-file status to console

    main(input_dir, output_csv=output_csv, verbose=verbose)
