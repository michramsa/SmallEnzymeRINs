from imports import *
import networkx as nx
from small_world_networkx import analyze_small_world_properties, print_small_world_analysis

# turn PDBs into RIGs

# these are the proteins
G = nx.read_graphml(KNOWN_DIR / "saved graphml graphs/2LW5.graphml")
G_2 = nx.read_graphml(KNOWN_DIR / "saved graphml graphs/5Y6A.graphml")
G_3 = nx.read_graphml(KNOWN_DIR / "saved graphml graphs/6vrb.graphml")

# run for analysis of small-world props.
# results = analyze_small_world_properties(G)
# print_small_world_analysis(results)

# results2 = analyze_small_world_properties(G_2)
# print_small_world_analysis(results2)

# results3 = analyze_small_world_properties(G_3)
# print_small_world_analysis(results3)

print("Trying Networkx sigma")
try:
    nx_sigma = nx.sigma(G, niter=5, nrand=5)
except:
    nx_sigma = None
    print("Warning: NetworkX sigma calculation failed")