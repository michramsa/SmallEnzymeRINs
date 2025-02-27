from imports import *
from constants import *
from protein_contact import *
from Bio.PDB.SASA import ShrakeRupley

def calc_residue_distance(res1, res2, choice):
    if choice == 'alpha':
        ca1 = res1["CA"]
        ca2 = res2["CA"]
        # subtraction is overloaded to provide distance
        difference = ca1 - ca2
        return difference
    elif choice == 'beta':
        cb1 = res1["CB"]
        cb2 = res2["CB"]
        # subtraction is overloaded to provide distance
        difference = cb1 - cb2
        return difference
    else:
        return 0


def calc_dist_matrix(chain1, chain2, threshold, choice):
    # this just makes some array of size of chain 1 by size of chain 2, specifies the data type as a float
    ans = numpy.zeros((len(chain1), len(chain2)), float)
    for row, res1 in enumerate(chain1):
        for col, res2 in enumerate(chain2):
            if choice == 'alpha':
                dist = calc_residue_distance(res1,res2, 'alpha')
            elif choice == 'beta':
                dist = calc_residue_distance(res1,res2, 'beta')
            else:
                break

            if dist <= threshold:
                ans[row,col] = 1
    
    return ans

# takes in the name of your structure, and the associated PDB file (check your path) - returns a Networkx graph
# much of the coding in this function utilizes syntax specific to Biopython - refer to their documentation for clarifications
# pick based on interest of finding alpha or beta distances
def create_graph(structure_name, structure_file, choice, threshold):
    # to help with dealing with the PDB file structure, we use a parser
    parser = PDBParser()
    try:
        structure = parser.get_structure(structure_name, structure_file)
    except Exception as e:
        print(e)
        return None, None, None
    # res_list gives a list of residues
    res_list = Selection.unfold_entities(structure, "R")
    # print("Res list unclean", res_list)

    res_list_clean = []
    if choice == 'alpha':
        # only interested in the alpha carbons, separate these residues
        for res in res_list:
            if res.has_id("CA"):
                res_list_clean.append(res)
    elif choice == 'beta':
        # only interested in the beta carbons, separate these residues
        for res in res_list:
            if res.has_id("CB"):
                res_list_clean.append(res)

    # print("Res list cleaned:", res_list_clean)
    # print("Number of res: ", len(res_list_clean))

    # add in the names as an attribute
    res_names_dict = {}

    # add in residues only
    for i in range(0,len(res_list_clean)):
        res_names = {}
        res_names["residue"]=res_list_clean[i].get_resname()
        res_names_dict[i]=res_names

    #create contact map
    contact_map = calc_dist_matrix(res_list_clean, res_list_clean, threshold, choice)

    # can check the dimensions of the contact map but it should be the (length of sequence)^2
    # print(np.shape(contact_map))

    # generate graph
    g = nx.Graph()

    # generate nodes for the graph
    g.add_nodes_from(list(range(0,len(res_list_clean))))

    # apply residue labels to graph
    nx.set_node_attributes(g,res_names_dict)

    dipole_dict = {}
    volume_dict = {}
    # add in acranker labels
    for i in range(0, len(res_list_clean)):
        # get the residue (3-letter name) from the graph
        current_node_residue = g.nodes[i]["residue"]

        # get the associated single letter abbreviation for that residue
        single_letter_res = amino_acids.get(current_node_residue)

        # find the group it belongs to from the acrankers
        for j in range(1,8):
            group_aminos = acranker_labels[j]['amino']
            for letter in group_aminos:
                if single_letter_res == letter:
                    dipole_scale = {}
                    volume = {}
                    dipole_scale['dipole_scale'] = acranker_labels[j]['dipole_scale']
                    volume['volume'] = acranker_labels[j]['volume']
                    dipole_dict[i] = dipole_scale
                    volume_dict[i] = volume

    # apply residue labels to graph
    nx.set_node_attributes(g,dipole_dict)
    nx.set_node_attributes(g,volume_dict)

    # check 
    # for i in range(0, len(res_list_clean)):
    #     print("TESTER 2: ", g.nodes[i])

    # generate edges
    edge_list = []
    # assign an index for each residue, iterate through the contact map
    for index, res in enumerate(contact_map):
        # print(res)
        indices = np.where(res==1)
        for contact_pt_index in indices[0]:
            if index != contact_pt_index:
                new_edge = (index, contact_pt_index)
                edge_list.append(new_edge)
    # see edge list
    # print(edge_list)

    # add edges into the graph
    g.add_edges_from(edge_list)

    return g, dipole_dict, volume_dict

if __name__ == '__main__':

    source_dir = "/home/epi2melabs/projects/AcrMich/known acrs"
    save_dir = "/home/epi2melabs/projects/AcrMich/known acrs/NAPS_beta_graphs"

    # # for saving the graphs for alpha
    # for file in os.listdir(source_dir):
    #     print(file)
    #     if ".pdb" in file:
    #         new_file_name = file.split('.')[0]
    #         graph, dipoles, volumes = create_graph(new_file_name, source_dir+"/"+file, 'beta', 7)

    #         save_path = os.path.join(save_dir, f"{new_file_name}.adjlist")
    #         print(f"Saving adjacency list to: {save_path}")

    #         nx.write_adjlist(graph, save_path)  # Save adjacency list

    ## loop to open graphs
    for file in os.listdir(save_dir):
        if file.endswith('.adjlist'):
            file_path = os.path.join(save_dir, file)
            
            graph = nx.read_adjlist(file_path)
            
            print(f"Loaded graph: {file}")
            print(f"Number of nodes: {graph.number_of_nodes()}")
            print(f"Number of edges: {graph.number_of_edges()}")
            print("---")

    # # loop for degree centrality measures
    # for file in os.listdir(save_dir):
    #     if file.endswith('.adjlist'):
    #         file_path = os.path.join(save_dir, file)
    #         graph = nx.read_adjlist(file_path)

    #         central_dict = nx.degree_centrality(graph)

    #         data = [{"Node": node, "DegreeCentrality": centrality} for node, centrality in central_dict.items()]

    #         output_filename = os.path.join("/home/epi2melabs/projects/AcrMich/known acrs/NAPS_beta_graphs/degree_centrality/", file + ".csv")

    #         with open(output_filename, 'w', newline='') as csvfile:
    #             writer = csv.DictWriter(csvfile, fieldnames=["Node", "DegreeCentrality"])
    #             writer.writeheader()
    #             writer.writerows(data)


    # # loop for closeness centrality measures
    # for file in os.listdir(save_dir):
    #     if file.endswith('.adjlist'):
    #         file_path = os.path.join(save_dir, file)
    #         graph = nx.read_adjlist(file_path)

    #         central_dict = nx.closeness_centrality(graph)

    #         data = [{"Node": node, "ClosenessCentrality": centrality} for node, centrality in central_dict.items()]

    #         output_filename = os.path.join("/home/epi2melabs/projects/AcrMich/known acrs/NAPS_beta_graphs/closeness_centrality/", file + ".csv")

    #         with open(output_filename, 'w', newline='') as csvfile:
    #             writer = csv.DictWriter(csvfile, fieldnames=["Node", "ClosenessCentrality"])
    #             writer.writeheader()
    #             writer.writerows(data)

    # # loop for creating CSV file with SASA and RSA values
    # # Tien MZ, Meyer AG, Sydykova DK, Spielman SJ, Wilke CO (2013) Maximum Allowed Solvent Accessibilites of Residues in Proteins. PLoS ONE 8(11):
    # # e80635. doi:10.1371/journal.pone.0080635
    # max_sasa_theo_dict = {
    #     "ALA": 129,
    #     "ARG": 274,
    #     "ASN": 195,
    #     "ASP": 193,
    #     "CYS": 167,
    #     "GLN": 225,
    #     "GLU": 223,
    #     "GLY": 104,
    #     "HIS": 224,
    #     "ILE": 197,
    #     "LEU": 201,
    #     "LYS": 236,
    #     "MET": 224,
    #     "PHE": 240,
    #     "PRO": 159,
    #     "SER": 155,
    #     "THR": 172,
    #     "TRP": 285,
    #     "TYR": 263,
    #     "VAL": 174
    # }

    # max_sasa_emp_dict = {
    #     "ALA": 121,
    #     "ARG": 265,
    #     "ASN": 187,
    #     "ASP": 187,
    #     "CYS": 148,
    #     "GLN": 214,
    #     "GLU": 214,
    #     "GLY": 97,
    #     "HIS": 216,
    #     "ILE": 195,
    #     "LEU": 191,
    #     "LYS": 230,
    #     "MET": 203,
    #     "PHE": 228,
    #     "PRO": 154,
    #     "SER": 143,
    #     "THR": 163,
    #     "TRP": 264,
    #     "TYR": 255,
    #     "VAL": 165
    # }

    # omissions = ['closeness_centrality', 'degree_centrality', 'SASA', 'saved graphs']

    # # ## i believe this is unchanged regardless of the graph
    # # for file in os.listdir(source_dir):
    # #     print(file)
    # #     new_file_name = file.split('.')[0]
    # #     # create the graph
    # #     # graph, dipoles, volumes = create_graph(new_file_name, source_dir+"/"+file)
    # #     # get the order and the identity of each amino acid
    # #     # node_res_dict = nx.get_node_attributes(graph, "residue")

    # #     # get the SASA values on the PDB file for that structure
    # #     if file not in omissions:
    # #         p = PDBParser(QUIET=1)
    # #         struct = p.get_structure(new_file_name, source_dir+"/"+file)
    # #         sr = ShrakeRupley()
    # #         sr.compute(struct, level="R")
    # #         sasa_data = []
    # #         for model in struct:
    # #             for chain in model:
    # #                 for residue in chain:
    # #                     if "CA" in residue:
    # #                         print(file)
    # #                         print(max_sasa_theo_dict.get(residue.resname))
    # #                         if max_sasa_theo_dict.get(residue.resname) == None or max_sasa_emp_dict.get(residue.resname) == None:
    # #                             sasa_data.append({
    # #                                 "Model": model.id,
    # #                                 "Chain": chain.id,
    # #                                 "Residue": residue.resname,
    # #                                 "Residue ID": residue.id[1],  
    # #                                 "SASA (Å²)": round(residue.sasa, 2),
    # #                                 "RSA-theoretical (Å²)": -1,
    # #                                 "RSA-empirical (Å²)": -1
    # #                             })
    # #                         else:
    # #                             sasa_data.append({
    # #                                 "Model": model.id,
    # #                                 "Chain": chain.id,
    # #                                 "Residue": residue.resname,
    # #                                 "Residue ID": residue.id[1],  
    # #                                 "SASA (Å²)": round(residue.sasa, 2),
    # #                                 "RSA-theoretical (Å²)": round(residue.sasa, 2) / max_sasa_theo_dict.get(residue.resname),
    # #                                 "RSA-empirical (Å²)": round(residue.sasa, 2) / max_sasa_emp_dict.get(residue.resname)
    # #                             })

    # #         df = pd.DataFrame(sasa_data)
    # #         df.to_csv(source_dir+"/SASA/"+new_file_name+".csv")

    # ## calculate standardized closeness centrality
    # close_dir = "/home/epi2melabs/projects/AcrMich/known acrs/NAPS_beta_graphs/closeness_centrality"
    # for file in os.listdir(close_dir):
    #     print(file)
    #     new_file_name = file.split('.')[0]
    #     df = pd.read_csv(close_dir+"/"+file)
    #     # print(df.head())
    #     df_sorted = df.sort_values(by='Node')
    #     # print(df_sorted.head())

    #     closeness_vals = df_sorted['ClosenessCentrality'].tolist()
    #     # print(closeness_vals)

    #     mean_closeness = np.mean(closeness_vals)
    #     std_closeness = np.std(closeness_vals)

    #     # Z = (observed value - mean of the sample) / standard deviation of the sample
    #     standardized_closeness = [(c - mean_closeness) / std_closeness for c in closeness_vals]

    #     # add to existing df and make new file
    #     df_sorted['StandardizedCloseness'] = standardized_closeness

    #     df_sorted.to_csv("/home/epi2melabs/projects/AcrMich/known acrs/NAPS_beta_graphs/standardized_closeness/stdclose_"+new_file_name+".csv")

    # #### calculate standardized degree centrality
    # degree_dir = "/home/epi2melabs/projects/AcrMich/known acrs/NAPS_beta_graphs/degree_centrality"
    # for file in os.listdir(degree_dir):
    #     print(file)
    #     new_file_name = file.split('.')[0]
    #     df = pd.read_csv(degree_dir+"/"+file)
    #     # print(df.head())
    #     df_sorted = df.sort_values(by='Node')
    #     # print(df_sorted.head())

    #     degree_vals = df_sorted['DegreeCentrality'].tolist()
    #     # print(degree_vals)

    #     mean_degree = np.mean(degree_vals)
    #     std_degree = np.std(degree_vals)

    #     # Z = (observed value - mean of the sample) / standard deviation of the sample
    #     standardized_degree = [(c - mean_degree) / std_degree for c in degree_vals]

    #     # add to existing df and make new file
    #     df_sorted['StandardizedDegreeCent'] = standardized_degree

    #     df_sorted.to_csv("/home/epi2melabs/projects/AcrMich/known acrs/NAPS_beta_graphs/standardized_degree_cent/stddegcent_"+new_file_name+".csv")