# Modified create_PSN.py to include RNA residues
# Key changes: Use different reference atoms for protein vs RNA/DNA

# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *
from constants import *
from Bio.PDB.SASA import ShrakeRupley
import numpy as np
import networkx as nx
from Bio.PDB import PDBParser, Selection

# Define residue types
PROTEIN_RESIDUES = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                   'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                   'THR', 'TRP', 'TYR', 'VAL'}

RNA_RESIDUES = {'A', 'U', 'G', 'C'}
DNA_RESIDUES = {'DA', 'DT', 'DG', 'DC'}
NUCLEIC_ACID_RESIDUES = RNA_RESIDUES.union(DNA_RESIDUES)

def get_reference_atom(residue):
    """
    Get the appropriate reference atom for distance calculations.
    For proteins: CA (alpha carbon)
    For RNA/DNA: C1' (anomeric carbon of ribose/deoxyribose)
    """
    res_name = residue.get_resname().strip()
    
    if res_name in PROTEIN_RESIDUES:
        if residue.has_id("CA"):
            return residue["CA"]
    elif res_name in NUCLEIC_ACID_RESIDUES:
        # Try C1' first (standard), then C1* (alternate naming)
        if residue.has_id("C1'"):
            return residue["C1'"]
        elif residue.has_id("C1*"):
            return residue["C1*"]
    
    return None

def calc_residue_distance(res1, res2):
    """Calculate distance between two residues using appropriate reference atoms."""
    try:
        atom1 = get_reference_atom(res1)
        atom2 = get_reference_atom(res2)
        
        if atom1 is None or atom2 is None:
            return float('inf')
        
        difference = atom1 - atom2
        return difference
    except Exception as e:
        return float('inf')

def calc_dist_matrix(residues, threshold):
    """Calculate distance matrix between all residues."""
    n_residues = len(residues)
    ans = np.zeros((n_residues, n_residues), float)
    
    for row, res1 in enumerate(residues):
        for col, res2 in enumerate(residues):
            if row != col:  # Skip self-comparisons
                dist = calc_residue_distance(res1, res2)
                if dist <= threshold:
                    ans[row, col] = 1
    
    return ans

def create_multi_chain_graph_with_rna(structure_name, structure_file, chain_choice, threshold, include_rna=True):
    """
    Create a protein structure network that includes both protein and RNA/DNA residues.
    
    Parameters:
    - structure_name: Name identifier for the structure
    - structure_file: Path to PDB file
    - chain_choice: 'both' for all chains, or specific chain ID
    - threshold: Distance threshold in Angstroms
    - include_rna: Whether to include RNA/DNA residues (default True)
    
    Returns:
    - graph: NetworkX graph object
    - dipole_dict: Dictionary of dipole moments (placeholder)
    - volume_dict: Dictionary of volumes (placeholder)
    """
    
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(structure_name, structure_file)
    except Exception as e:
        print(f"Error parsing PDB file: {e}")
        return None, {}, {}
    
    # Print available chains
    print("Available chains:")
    for chain in structure[0]:  # Access first model
        print(f"Chain ID: {chain.id}")
    
    # Get residues based on chain choice
    if chain_choice == "both":
        print("Processing all chains")
        res_list = Selection.unfold_entities(structure, "R")
    else:
        print(f"Processing chain {chain_choice}")
        try:
            chain_obj = structure[0][chain_choice]
            res_list = Selection.unfold_entities(chain_obj, "R")
        except KeyError:
            print(f"Chain {chain_choice} not found in structure")
            return None, {}, {}

    # Filter residues based on available reference atoms and residue type
    res_list_clean = []
    protein_count = 0
    rna_count = 0
    dna_count = 0
    other_count = 0
    
    for res in res_list:
        res_name = res.get_resname().strip()
        
        # Skip water and other unwanted residues
        if res_name in ['HOH', 'WAT']:
            continue
        
        # Check if we have appropriate reference atom
        ref_atom = get_reference_atom(res)
        if ref_atom is not None:
            res_list_clean.append(res)
            
            # Count residue types
            if res_name in PROTEIN_RESIDUES:
                protein_count += 1
            elif res_name in RNA_RESIDUES:
                rna_count += 1
            elif res_name in DNA_RESIDUES:
                dna_count += 1
            else:
                other_count += 1

    print(f"Residue composition:")
    print(f"  Protein residues: {protein_count}")
    print(f"  RNA residues: {rna_count}")
    print(f"  DNA residues: {dna_count}")
    print(f"  Other residues: {other_count}")
    print(f"  Total residues: {len(res_list_clean)}")

    # Create residue names dictionary with chain information
    res_names_dict = {}
    chain_counters = {}  # Keep track of residue numbers per chain
    
    for i, res in enumerate(res_list_clean):
        chain_id = res.get_parent().id
        res_name = res.get_resname().strip()
        
        # Initialize counter for new chains
        if chain_id not in chain_counters:
            chain_counters[chain_id] = 1
        
        # Determine residue type
        if res_name in PROTEIN_RESIDUES:
            res_type = "protein"
        elif res_name in NUCLEIC_ACID_RESIDUES:
            res_type = "nucleic_acid"
        else:
            res_type = "other"
        
        # Create residue identifier
        if chain_choice == "both":
            res_identifier = f"{res_name}_{chain_id}_{chain_counters[chain_id]}"
        else:
            res_identifier = f"{res_name}_{chain_counters[chain_id]}"
        
        res_names_dict[i] = {
            "residue": res_identifier,
            "chain_id": chain_id,
            "res_number": res.get_id()[1],
            "res_name": res_name,
            "res_type": res_type
        }
        
        chain_counters[chain_id] += 1

    # Create contact map
    print("Calculating contact map...")
    contact_map = calc_dist_matrix(res_list_clean, threshold)
    print(f"Contact map dimensions: {contact_map.shape}")

    # Create NetworkX graph
    g = nx.Graph()
    
    # Add nodes
    g.add_nodes_from(list(range(len(res_list_clean))))
    
    # Set node attributes
    nx.set_node_attributes(g, res_names_dict)

    # Generate edges from contact map
    edge_list = []
    for index, res in enumerate(contact_map):
        indices = np.where(res == 1)[0]
        for contact_pt_index in indices:
            if index != contact_pt_index:  # Avoid self-loops
                new_edge = (index, contact_pt_index)
                edge_list.append(new_edge)

    print(f"Number of edges: {len(edge_list)}")
    
    # Add edges to graph
    g.add_edges_from(edge_list)

    # Create placeholder dictionaries for dipoles and volumes
    dipole_dict = {i: 0.0 for i in range(len(res_list_clean))}
    volume_dict = {i: 0.0 for i in range(len(res_list_clean))}

    print(f"Graph created with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")
    
    return g, dipole_dict, volume_dict

def analyze_graph_by_chain_and_type(graph):
    """Analyze the graph and print statistics by chain and residue type."""
    if graph is None:
        return
    
    chain_stats = {}
    type_stats = {'protein': 0, 'nucleic_acid': 0, 'other': 0}
    
    for node_id, node_data in graph.nodes(data=True):
        chain_id = node_data.get('chain_id', 'Unknown')
        res_type = node_data.get('res_type', 'other')
        
        if chain_id not in chain_stats:
            chain_stats[chain_id] = {
                'node_count': 0,
                'residue_types': set(),
                'degrees': [],
                'protein_count': 0,
                'nucleic_acid_count': 0,
                'other_count': 0
            }
        
        chain_stats[chain_id]['node_count'] += 1
        chain_stats[chain_id]['residue_types'].add(node_data.get('res_name', 'Unknown'))
        chain_stats[chain_id]['degrees'].append(graph.degree(node_id))
        chain_stats[chain_id][f'{res_type}_count'] += 1
        
        type_stats[res_type] += 1
    
    print("\n=== Overall Composition ===")
    for res_type, count in type_stats.items():
        print(f"{res_type.replace('_', ' ').title()}: {count} residues")
    
    print("\n=== Chain Analysis ===")
    for chain_id, stats in chain_stats.items():
        print(f"Chain {chain_id}:")
        print(f"  - Total nodes: {stats['node_count']}")
        print(f"  - Protein: {stats['protein_count']}")
        print(f"  - Nucleic acid: {stats['nucleic_acid_count']}")
        print(f"  - Other: {stats['other_count']}")
        print(f"  - Unique residue types: {len(stats['residue_types'])}")
        print(f"  - Average degree: {np.mean(stats['degrees']):.2f}")
        print(f"  - Degree range: {min(stats['degrees'])} - {max(stats['degrees'])}")

if __name__ == "__main__":
    # Example usage
    file_name = IMP_DIR / "3 main Acr PDB files/6vrb.pdb"
    save_dir = IMP_DIR / "3 main Acr files"
    structure_name = "6vrb"
    
    # Create the multi-chain graph including RNA
    print("Creating multi-chain protein structure network with RNA...")
    graph, dipoles, volumes = create_multi_chain_graph_with_rna(
        structure_name, 
        file_name, 
        'both',  # Process all chains
        7,  # 7 Angstrom threshold
        include_rna=True
    )
    
    if graph is not None:
        # Analyze the graph
        analyze_graph_by_chain_and_type(graph)
        
        # Save the graph (you can use your existing save function)
        # save_graph_with_chain_info(graph, save_dir, structure_name)
        
        print(f"\nFinal graph statistics:")
        print(f"Total nodes: {graph.number_of_nodes()}")
        print(f"Total edges: {graph.number_of_edges()}")
        print(f"Graph density: {nx.density(graph):.4f}")
        
        # Check connectivity
        if nx.is_connected(graph):
            print("Graph is connected")
        else:
            components = list(nx.connected_components(graph))
            print(f"Graph has {len(components)} connected components")
            print(f"Largest component size: {len(max(components, key=len))}")
    else:
        print("Failed to create graph")