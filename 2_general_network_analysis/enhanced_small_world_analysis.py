from imports import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# determine if networks display small world properties
# inputs: a graph and the number of random graphs to generate for comparison
def analyze_small_world_properties(G, n_random=100):
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc)
        print(f"Warning: Using largest connected component with {G.number_of_nodes()} nodes")
    
    # Basic network properties
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(n_nodes)
    print(n_edges)
    
    # Calculate clustering coefficient
    C = nx.average_clustering(G)
    
    # Calculate average path length
    L = nx.average_shortest_path_length(G)
    
    # For random graph comparison (G(n,p) model)
    p = 2 * n_edges / (n_nodes * (n_nodes - 1))  # probability for G(n,p) random graph
    
    # Generate random graphs for comparison
    random_clusters = []
    random_paths = []
    
    for i in range(n_random):
        G_rand = nx.gnm_random_graph(n_nodes, n_edges, seed = 1)
        
        if not nx.is_connected(G_rand):
            largest_cc = max(nx.connected_components(G_rand), key=len)
            G_rand = G_rand.subgraph(largest_cc)
            continue
            
        random_clusters.append(nx.average_clustering(G_rand))
        random_paths.append(nx.average_shortest_path_length(G_rand))
    
    C_rand = np.mean(random_clusters)
    L_rand = np.mean(random_paths)
    
    avg_degree = 2 * n_edges / n_nodes
    
    # Method 1: Create a ring lattice with approximately the same number of edges
    # Start with a higher k and then remove edges if needed
    k = int(avg_degree)
    if k % 2 == 1:  # k even required
        k += 1  # Round up to ensure we have enough edges
    
    G_lattice = nx.watts_strogatz_graph(n_nodes, k, 0, seed = 1)
    lattice_edges = G_lattice.number_of_edges()
    
    print(f"Before deletion: {G_lattice.number_of_edges()} edges")
    if lattice_edges > n_edges:
        print("DELETING EDGES")
        # Create a list of edges sorted by the distance between nodes in the ring
        edges_with_distance = []
        for u, v in G_lattice.edges():
            # Calculate the "ring distance" between nodes
            dist = min(abs(u-v), n_nodes - abs(u-v))
            edges_with_distance.append((u, v, dist))
    
        # Sort by distance (descending) to remove the "least lattice-like" edges first
        edges_with_distance.sort(key=lambda x: -x[2])
    
        # Remove the required number of edges
        for i in range(lattice_edges - n_edges):
            u, v, _ = edges_with_distance[i]
            G_lattice.remove_edge(u, v)
    
    print(f"After deletion: {G_lattice.number_of_edges()} edges")

    if lattice_edges < n_edges:
        print("ADDING EDGES")
        # Find node pairs that maintain lattice structure (connect to further neighbors)
        potential_edges = []
        for i in range(n_nodes):
            current_neighbors = set(G_lattice.neighbors(i))
            # Find the next nearest neighbors to connect to in order
            for distance in range(1, n_nodes//2):
                # Try next neighbor clockwise
                next_neighbor = (i + k//2 + distance) % n_nodes
                # And counterclockwise
                prev_neighbor = (i - k//2 - distance) % n_nodes
                
                if next_neighbor not in current_neighbors:
                    potential_edges.append((i, next_neighbor, distance))
                if prev_neighbor not in current_neighbors:
                    potential_edges.append((i, prev_neighbor, distance))
        
        # Sort by distance (increasing) to add nearest neighbors first
        potential_edges.sort(key=lambda x: x[2])
        
        # Add the required number of edges
        edges_needed = n_edges - lattice_edges
        added_edges = set()  # Track which edges we've added
        
        for i, j, _ in potential_edges:
            if len(added_edges) >= edges_needed:
                break
                
            edge = tuple(sorted([i, j]))  # Normalize the edge representation
            if edge not in added_edges and not G_lattice.has_edge(i, j):
                G_lattice.add_edge(i, j)
                added_edges.add(edge)
    
    print(f"Random graph: {G_rand.number_of_nodes()} nodes, {G_rand.number_of_edges()} edges")
    print(f"Lattice graph: {G_lattice.number_of_nodes()} nodes, {G_lattice.number_of_edges()} edges")

    # Use this function after creating both graphs
    visualize_graphs(G, G_rand, title2 = "Random Reference")
    visualize_graphs(G, G_lattice)

    if nx.is_connected(G_lattice):
        L_lattice = nx.average_shortest_path_length(G_lattice)
        C_lattice = nx.average_clustering(G_lattice)
    else:
        largest_cc = max(nx.connected_components(G_lattice), key=len)
        G_lattice_cc = G_lattice.subgraph(largest_cc)
        L_lattice = nx.average_shortest_path_length(G_lattice_cc)
        print(f"Warning: Lattice not connected, using largest component with {G_lattice_cc.number_of_nodes()} nodes")
        
    # sigma
    sigma = (C / C_rand) / (L / L_rand)
    
    # omega
    omega = (L_rand / L) - (C / C_lattice)
    omega_normalized = 1 - abs(omega)  # Normalized to [0,1] where 1 is most small-worldly
    
    # small-world index/SWI
    if L_lattice == L_rand:  # Edge case - prevent division by zero
        swi_L = 0.5  # Middle value
    else:
        swi_L = (L - L_lattice) / (L_rand - L_lattice)
    
    if C_lattice == C_rand:  # Edge case - prevent division by zero
        swi_C = 0.5  # Middle value
    else:
        swi_C = (C - C_rand) / (C_lattice - C_rand)
    
    swi = swi_L * swi_C
    
    # 4. Use NetworkX's built-in small world sigma
    # This implementation may differ slightly from our custom one
    print("Trying Networkx sigma")
    try:
        nx_sigma = nx.sigma(G, niter=10, nrand=5, seed = 1)
    except:
        nx_sigma = None
        print("Warning: NetworkX sigma calculation failed")
    
    # 5. Use NetworkX's built-in small world omega
    print("Trying Networkx omega")
    try:
        nx_omega = nx.omega(G, niter=10, nrand=5, seed = 1)
        nx_omega_normalized = 1 - abs(nx_omega)  # Normalized to [0,1]
    except:
        nx_omega = None
        nx_omega_normalized = None
        print("Warning: NetworkX omega calculation failed")
        
    # Find degree distribution
    degrees = [d for n, d in G.degree()]
    mean_degree = np.mean(degrees)
    max_degree = max(degrees)
    std_degree = np.std(degrees)
    
    # Store and return results
    results = {
        # Network properties
        "nodes": n_nodes,
        "edges": n_edges,
        "avg_degree": mean_degree,
        "max_degree": max_degree,
        "std_degree": std_degree,
        "degrees": degrees,
        
        # Clustering coefficients
        "clustering_coefficient": C,
        "random_clustering": C_rand,
        "lattice_clustering": C_lattice,
        
        # Path lengths
        "avg_path_length": L,
        "random_path_length": L_rand,
        "lattice_path_length": L_lattice,
        
        # Small-world metrics
        "sigma": sigma,                          # Traditional small-world coefficient
        "omega": omega,                          # Telesford et al. (2011) metric
        "omega_normalized": omega_normalized,    # Normalized omega to [0,1]
        "small_world_index": swi,                # Neal (2017) metric
        
        # NetworkX implementations
        "nx_sigma": nx_sigma,                    # NetworkX's sigma implementation
        "nx_omega": nx_omega,                    # NetworkX's omega implementation
        "nx_omega_normalized": nx_omega_normalized  # Normalized NetworkX omega to [0,1]
    }
    
    return results


def print_small_world_analysis(results):
    print("\n" + "="*50)
    print("SMALL WORLD NETWORK ANALYSIS")
    print("="*50)
    
    print("\n--- Network Properties ---")
    print(f"Nodes: {results['nodes']}")
    print(f"Edges: {results['edges']}")
    print(f"Mean Degree: {results['avg_degree']:.2f}")
    print(f"Max Degree: {results['max_degree']}")
    print(f"Standard Deviation of Degrees: {results['std_degree']:.2f}")
    
    print("\n--- Clustering and Path Length ---")
    print(f"Clustering Coefficient (C): {results['clustering_coefficient']:.4f}")
    print(f"Random Graph Clustering (C_rand): {results['random_clustering']:.4f}")
    print(f"Lattice Graph Clustering (C_lattice): {results['lattice_clustering']:.4f}")
    print(f"Average Path Length (L): {results['avg_path_length']:.4f}")
    print(f"Random Graph Path Length (L_rand): {results['random_path_length']:.4f}")
    print(f"Lattice Graph Path Length (L_lattice): {results['lattice_path_length']:.4f}")
    
    print("\n--- Small-World Metrics ---")
    print(f"Sigma (σ): {results['sigma']:.4f}")
    if results['sigma'] > 1:
        print("  ↳ Network shows small-world properties (σ > 1)")
    else:
        print("  ↳ Network does not show small-world properties (σ ≤ 1)")
    
    print(f"Omega (ω): {results['omega']:.4f}")
    print(f"Normalized Omega (ω'): {results['omega_normalized']:.4f}")
    if -0.5 < results['omega'] < 0.5:
        print("  ↳ Network shows small-world properties (ω close to 0)")
    elif results['omega'] <= -0.5:
        print("  ↳ Network shows lattice-like properties (ω negative)")
    else:
        print("  ↳ Network shows random-like properties (ω positive)")
    
    print(f"Small-World Index (SWI): {results['small_world_index']:.4f}")
    if results['small_world_index'] > 0.6:
        print("  ↳ Network shows strong small-world properties (SWI > 0.6)")
    elif results['small_world_index'] > 0.4:
        print("  ↳ Network shows moderate small-world properties (0.4 < SWI < 0.6)")
    else:
        print("  ↳ Network shows weak or no small-world properties (SWI < 0.4)")
    
    if results['nx_sigma'] is not None:
        print(f"NetworkX Sigma: {results['nx_sigma']:.4f}")
    
    if results['nx_omega'] is not None:
        print(f"NetworkX Omega: {results['nx_omega']:.4f}")
        print(f"NetworkX Normalized Omega: {results['nx_omega_normalized']:.4f}")

# input: dictionary containing results from other function, and a title for yoyur graph
def plot_small_world_analysis(results, savefile, name, title="Small World Network Analysis"):
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # 1. Degree distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(results['degrees'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(results['avg_degree'], color='r', linestyle='--', 
                label=f'Mean Degree: {results["avg_degree"]:.2f}')
    ax1.set_title('Degree Distribution')
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(alpha=0.3)
    fig.savefig(name+'_degree.png')
    
    # 2. Small-world metrics comparison
    ax2 = fig.add_subplot(gs[0, 1:])
    metrics = ['sigma', 'omega_normalized', 'small_world_index']
    values = [results['sigma']/max(1, results['sigma']), 
              results['omega_normalized'], 
              results['small_world_index']]
    
    if results['nx_sigma'] is not None and results['nx_omega_normalized'] is not None:
        metrics.extend(['nx_sigma', 'nx_omega_normalized'])
        values.extend([results['nx_sigma']/max(1, results['nx_sigma']), 
                      results['nx_omega_normalized']])
    
    pos = np.arange(len(metrics))
    ax2.bar(pos, values, align='center', alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xticks(pos)
    ax2.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax2.set_title('Small-World Metrics Comparison')
    ax2.set_ylabel('Normalized Value')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Clustering and Path Length comparison with reference networks
    ax3 = fig.add_subplot(gs[1, :2])
    
    # Normalize all values to [0,1] for easier comparison
    c_values = [
        results['clustering_coefficient'],
        results['random_clustering'],
        results['lattice_clustering']
    ]
    c_max = max(c_values)
    c_norm = [c/c_max for c in c_values]
    
    l_values = [
        results['avg_path_length'],
        results['random_path_length'],
        results['lattice_path_length']
    ]
    l_max = max(l_values)
    l_norm = [l/l_max for l in l_values]
    
    width = 0.35
    network_types = ['Observed', 'Random', 'Lattice']
    x = np.arange(len(network_types))
    
    ax3.bar(x - width/2, c_norm, width, label='Clustering Coefficient (normalized)', color='skyblue')
    ax3.bar(x + width/2, l_norm, width, label='Path Length (normalized)', color='salmon')
    
    ax3.set_title('Comparison with Reference Networks')
    ax3.set_xticks(x)
    ax3.set_xticklabels(network_types)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Add actual values as text above the bars
    for i, v in enumerate(c_values):
        ax3.text(i - width/2, c_norm[i] + 0.05, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(l_values):
        ax3.text(i + width/2, l_norm[i] + 0.05, f'{v:.3f}', ha='center')
    
    # 4. Summary stats and interpretation
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    summary = [
        f"Network: {results['nodes']} nodes, {results['edges']} edges",
        f"Mean Degree: {results['avg_degree']:.2f}",
        "\nSmall-world properties:",
        f"Sigma (σ): {results['sigma']:.4f}",
        f"{'Small-world (σ > 1)' if results['sigma'] > 1 else 'Not small-world (σ ≤ 1)'}",
        f"\nOmega (ω): {results['omega']:.4f}",
        f"Normalized: {results['omega_normalized']:.4f}",
        f"{'Small-world (ω ≈ 0)' if -0.5 < results['omega'] < 0.5 else 'Not small-world'}",
        f"\nSmall-World Index: {results['small_world_index']:.4f}",
        f"{'Strong small-world (>0.6)' if results['small_world_index'] > 0.6 else 'Moderate small-world (0.4-0.6)' if results['small_world_index'] > 0.4 else 'Weak/No small-world (<0.4)'}"
    ]
    
    ax4.text(0, 1, '\n'.join(summary), va='top', fontsize=11)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()
    plt.savefig(savefile)

# input: a dictionary of results
def plot_degree_distribution(results):
    plt.figure(figsize=(10, 6))
    plt.hist(results['degrees'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(results['avg_degree'], color='r', linestyle='--', 
                label=f'Mean Degree: {results["avg_degree"]:.2f}')
    
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def visualize_graphs(G, G_lattice, title1="Original Graph", title2="Lattice Reference"):
    plt.figure(figsize=(15, 7))
    
    # Original graph
    plt.subplot(121)
    pos_original = nx.spring_layout(G, seed=42)  # For consistent layout
    nx.draw_networkx(G, pos=pos_original, node_size=30, node_color='blue', 
                    with_labels=False, alpha=0.7)
    plt.title(title1)
    
    # Lattice graph
    plt.subplot(122)
    # For a ring lattice, circular layout is most appropriate
    pos_lattice = nx.circular_layout(G_lattice)
    nx.draw_networkx(G_lattice, pos=pos_lattice, node_size=30, node_color='red',
                    with_labels=False, alpha=0.7)
    plt.title(title2)
    
    plt.tight_layout()
    plt.savefig("graph_comparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    
    # these are the PSN proteins
    # G = nx.read_graphml("./known acrs/saved graphml graphs/2LW5.graphml")
    # G_2 = nx.read_graphml("./known acrs/saved graphml graphs/5Y6A.graphml")
    # G_3 = nx.read_graphml("./known acrs/saved graphml graphs/6vrb.graphml")

    # these are the RIG proteins
    G = nx.read_graphml("/home/epi2melabs/projects/AcrNetworks/PISCES/known acr data/2LW5_clean.graphml")
    G_2 = nx.read_graphml("/home/epi2melabs/projects/AcrNetworks/PISCES/known acr data/5Y6A_clean.graphml")
    G_3 = nx.read_graphml("/home/epi2melabs/projects/AcrNetworks/PISCES/known acr data/6vrb_clean.graphml")

    # results = analyze_small_world_properties(G)
    # # print_small_world_analysis(results)
    # plot_small_world_analysis(results, "2LW5_small_world_RIG.png", "2LW5", "PDB ID: 2LW5 RIG Analysis")

    # results2 = analyze_small_world_properties(G_2)
    # # print_small_world_analysis(results2)
    # plot_small_world_analysis(results2, "5Y6A_small_world_RIG.png", "5Y6A", "PDB ID: 5Y6A RIG Analysis")

    results3 = analyze_small_world_properties(G_3)
    # print_small_world_analysis(results3)
    plot_small_world_analysis(results3, "6VRB_small_world_RIG.png", "6VRB", "PDB ID: 6VRB RIG Analysis")
    
    # tester
    # print("\nAnalyzing Zachary's Karate Club network...")
    # G_karate = nx.karate_club_graph()
    # results_karate = analyze_small_world_properties(G_karate)
    # print_small_world_analysis(results_karate)
    # plot_small_world_analysis(results_karate, "Zachary's Karate Club Network")