import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# determine if networks display small world properties
# inputs: a graph and the number of random graphs to generate for comparison
def analyze_small_world_properties(G, n_random=100):
    # determine if graph connected
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc)
        print(f"Warning: Using largest connected component with {G.number_of_nodes()} nodes")
    
    # calc clustering coefficient
    C = nx.average_clustering(G)
    
    # calc average path length
    L = nx.average_shortest_path_length(G)
    
    # make random graphs for comparison
    random_clusters = []
    random_paths = []
    
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    p = 2 * n_edges / (n_nodes * (n_nodes - 1))  # probability for G(n,p) random graph
    
    for i in range(n_random):
        # the random graphs should have the same amount of nodes and edges roughly
        G_rand = nx.erdos_renyi_graph(n_nodes, p)
        
        # make random graph connected
        if not nx.is_connected(G_rand):
            largest_cc = max(nx.connected_components(G_rand), key=len)
            G_rand = G_rand.subgraph(largest_cc)
            continue
            
        random_clusters.append(nx.average_clustering(G_rand))
        random_paths.append(nx.average_shortest_path_length(G_rand))
    
    # repeat calculations for the random graphs
    C_rand = np.mean(random_clusters)
    L_rand = np.mean(random_paths)
    
    # find small-world coefficient
    sigma = (C / C_rand) / (L / L_rand)
    
    # look more into degree distribution
    degrees = [d for n, d in G.degree()]
    mean_degree = np.mean(degrees)
    max_degree = max(degrees)
    std_degree = np.std(degrees)
    
    # Store results
    results = {
        "clustering_coefficient": C,
        "avg_path_length": L,
        "random_clustering": C_rand,
        "random_path_length": L_rand,
        "small_world_coefficient": sigma,
        "mean_degree": mean_degree,
        "max_degree": max_degree,
        "std_degree": std_degree,
        "degrees": degrees
    }
    
    return results

def print_small_world_analysis(results):
    """Print a summary of small-world analysis results"""
    print("\n--- Small World Network Analysis ---")
    print(f"Clustering Coefficient (C): {results['clustering_coefficient']:.4f}")
    print(f"Average Path Length (L): {results['avg_path_length']:.4f}")
    print(f"Random Graph Clustering (C_rand): {results['random_clustering']:.4f}")
    print(f"Random Graph Path Length (L_rand): {results['random_path_length']:.4f}")
    print(f"Small World Coefficient (σ): {results['small_world_coefficient']:.4f}")
    
    if results['small_world_coefficient'] > 1:
        print("\nThe network shows small-world properties (σ > 1)")
    else:
        print("\nThe network does not show small-world properties (σ ≤ 1)")
    
    print("\n--- Degree Distribution ---")
    print(f"Mean Degree: {results['mean_degree']:.2f}")
    print(f"Max Degree: {results['max_degree']}")
    print(f"Standard Deviation: {results['std_degree']:.2f}")

def plot_degree_distribution(results):
    """Plot the degree distribution of the network"""
    degrees = results['degrees']
    
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=30, alpha=0.7)
    plt.axvline(results['mean_degree'], color='r', linestyle='--', 
                label=f'Mean Degree: {results["mean_degree"]:.2f}')
    
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def multiple_networks():
    # Generate different types of networks
    n = 100  # number of nodes
    
    # 1. Watts-Strogatz small-world network
    G_small_world = nx.watts_strogatz_graph(n=n, k=4, p=0.1)
    
    # 2. Erdős-Rényi random network
    G_random = nx.erdos_renyi_graph(n=n, p=0.04)
    
    # 3. Barabási-Albert scale-free network
    G_scale_free = nx.barabasi_albert_graph(n=n, m=2)
    
    # 4. Regular lattice
    G_regular = nx.watts_strogatz_graph(n=n, k=4, p=0)
    
    # Analyze each network
    print("\n=== Watts-Strogatz Small World Network ===")
    results_small_world = analyze_small_world_properties(G_small_world)
    print_small_world_analysis(results_small_world)
    
    print("\n=== Erdős-Rényi Random Network ===")
    results_random = analyze_small_world_properties(G_random)
    print_small_world_analysis(results_random)
    
    print("\n=== Barabási-Albert Scale-Free Network ===")
    results_scale_free = analyze_small_world_properties(G_scale_free)
    print_small_world_analysis(results_scale_free)
    
    print("\n=== Regular Lattice ===")
    results_regular = analyze_small_world_properties(G_regular)
    print_small_world_analysis(results_regular)
    
    # Plot degree distributions
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.hist(results_small_world['degrees'], bins=15, alpha=0.7)
    plt.title('Small World Network')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 2)
    plt.hist(results_random['degrees'], bins=15, alpha=0.7)
    plt.title('Random Network')
    plt.xlabel('Degree')
    
    plt.subplot(2, 2, 3)
    plt.hist(results_scale_free['degrees'], bins=30, alpha=0.7)
    plt.title('Scale-Free Network')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 4)
    plt.hist(results_regular['degrees'], bins=5, alpha=0.7)
    plt.title('Regular Lattice')
    plt.xlabel('Degree')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    multiple_networks()
