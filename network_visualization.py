from imports import *

source_dir = "/home/epi2melabs/projects/AcrMich/known acrs"
save_dir = "/home/epi2melabs/projects/AcrMich/known acrs/saved graphs/"
# file_path = os.path.join(save_dir, file)

# graph = nx.read_adjlist(file_path)

# nx.write_graphml_lxml(graph, "/home/epi2melabs/projects/AcrMich/known acrs/saved graphml graphs/6anv.graphml")

# # network visualization options
# options = {
#     "font_size": 36,
#     "node_size": 100,
#     "node_color": "white",
#     "edgecolors": "black",
#     "linewidths": 5,
#     "width": 5,
# }# draw network
# nx.draw_networkx(graph, **options)

# # plot setting and show
# ax = plt.gca()
# ax.margins(0.20)
# plt.axis("off")
# plt.show()

########################################################################################

# parser = PDBParser()
# try:
#     structure = parser.get_structure("6ANV", "/home/epi2melabs/projects/AcrMich/known acrs/6anv.pdb")
# except Exception as e:
#     print(e)

# for chain in structure[0]:  # Access first model
#     print(f"Chain ID: {chain.id}")

# chain_A = structure[0]["B"]  # Select chain A from the first model
# res_list = Selection.unfold_entities(chain_A, "R")  # Extract only chain A residues

# # res_list gives a list of residues
# # res_list = Selection.unfold_entities(structure, "R")
# # print("Res list unclean", res_list)

# # only interested in the alpha carbons, separate these residues
# res_list_clean = []
# for res in res_list:
#     if res.has_id("CA"):
#         res_list_clean.append(res)

# print(len(res_list_clean))

##########################################################################################
# file = "/home/epi2melabs/projects/AcrMich/known acrs/6vrb.pdb"

# graph, dipoles, volumes = create_graph("6VRB", file)

# save_path = os.path.join(save_dir, f"6VRB.adjlist")
# print(f"Saving adjacency list to: {save_path}")

# nx.write_adjlist(graph, save_path)  # Save adjacency list

graph = nx.read_adjlist("/home/epi2melabs/projects/AcrMich/known acrs/saved graphs/6VRB.adjlist")

nx.write_graphml_lxml(graph, "/home/epi2melabs/projects/AcrMich/known acrs/saved graphml graphs/6VRB.graphml")