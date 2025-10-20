import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import random
import os

def analyze_graph(dataset_name):
    """
    Loads a graph from an edgelist, computes key statistics,
    and generates required plots.
    """
    edgelist_path = f'{dataset_name}-edgelist.csv'
    if not os.path.exists(edgelist_path):
        print(f"Error: {edgelist_path} not found. Please run the extraction script first.")
        return

    print(f"\n--- Analyzing {dataset_name} graph ---")

    # 1. Load the graph
    df = pd.read_csv(edgelist_path)
    # Create a directed graph from the pandas DataFrame
    G = nx.from_pandas_edgelist(
        df,
        source='Source',
        target='Target',
        edge_attr='Weight',
        create_using=nx.DiGraph()
    )

    # 2. Calculate Statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # --- Components ---
    # Weakly Connected Components (Giant Component)
    wccs = list(nx.weakly_connected_components(G))
    num_wcc = len(wccs)
    giant_component_nodes = max(wccs, key=len)
    size_giant_component = len(giant_component_nodes)

    # Strongly Connected Components
    sccs = list(nx.strongly_connected_components(G))
    num_scc = len(sccs)
    largest_scc_nodes = max(sccs, key=len)
    size_largest_scc = len(largest_scc_nodes)
    
    # --- Density ---
    density = nx.density(G)

    # --- Clustering Coefficient (Approximated) ---
    # For directed graphs, this metric is well-defined.
    # NetworkX's function is efficient and suitable here.
    avg_clustering = nx.average_clustering(G)

    # --- Average Distance and Distance Distribution (Undirected Giant Component) ---
    # Create an undirected subgraph of the giant component
    G_gc_undirected = G.subgraph(giant_component_nodes).to_undirected()

    # For larger graphs, calculating all-pairs shortest paths is very slow.
    # The prompt asks for an approximation. We can sample nodes.
    if G_gc_undirected.number_of_nodes() > 2000:
        # Approximation by sampling for the larger graph
        n_samples = 1000  # Number of nodes to sample for approximation
        nodes_sample = random.sample(list(G_gc_undirected.nodes()), k=n_samples)
        
        total_path_length = 0
        total_paths = 0
        all_path_lengths = []
        
        for node in nodes_sample:
            path_lengths = nx.shortest_path_length(G_gc_undirected, source=node)
            del path_lengths[node] # remove self-loops
            total_path_length += sum(path_lengths.values())
            total_paths += len(path_lengths)
            all_path_lengths.extend(path_lengths.values())
            
        avg_distance = total_path_length / total_paths if total_paths > 0 else 0
        dist_distribution = Counter(all_path_lengths)
        
    else:
        # Exact calculation for the smaller graph
        avg_distance = nx.average_shortest_path_length(G_gc_undirected)
        # Get distance distribution
        all_path_lengths = []
        for node, lengths in nx.shortest_path_length(G_gc_undirected):
            all_path_lengths.extend(l for l in lengths.values() if l > 0)
        dist_distribution = Counter(all_path_lengths)


    # 3. Print Statistics in LaTeX format
    print("--- Statistics (LaTeX Row Format) ---")
    print(f"Number of Nodes & {num_nodes:,} \\\\")
    print(f"Number of Edges & {num_edges:,} \\\\")
    print(f"Weakly Connected Components & {num_wcc:,} (Giant component: {size_giant_component:,} nodes) \\\\")
    print(f"Strongly Connected Components & {num_scc:,} (Largest: {size_largest_scc:,} nodes) \\\\")
    print(f"Density & {density:.2e} \\\\")
    print(f"Avg. Clustering Coefficient & {avg_clustering:.4f} \\\\")
    print(f"Avg. Distance (Giant Component) & {avg_distance:.4f} \\\\")
    print("-" * 35)

    # 4. Generate Plots
    
    # --- Degree Distributions ---
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]

    in_degree_counts = Counter(in_degrees)
    out_degree_counts = Counter(out_degrees)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Indegree
    ax1.loglog(list(in_degree_counts.keys()), list(in_degree_counts.values()), 'bo', markersize=4, alpha=0.7)
    ax1.set_title(f'Indegree Distribution ({dataset_name})')
    ax1.set_xlabel('Indegree (k)')
    ax1.set_ylabel('Frequency P(k)')
    ax1.grid(True, which="both", ls="--", linewidth=0.5)

    # Outdegree
    ax2.loglog(list(out_degree_counts.keys()), list(out_degree_counts.values()), 'ro', markersize=4, alpha=0.7)
    ax2.set_title(f'Outdegree Distribution ({dataset_name})')
    ax2.set_xlabel('Outdegree (k)')
    ax2.set_ylabel('Frequency P(k)')
    ax2.grid(True, which="both", ls="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_degree_distribution.png', dpi=300)
    plt.close()
    print(f"Saved {dataset_name}_degree_distribution.png")

    # --- Distance Distribution ---
    distances = sorted(dist_distribution.keys())
    counts = [dist_distribution[d] for d in distances]
    
    plt.figure(figsize=(8, 6))
    plt.bar(distances, counts, color='skyblue')
    plt.title(f'Distance Distribution of Giant Component ({dataset_name})')
    plt.xlabel('Shortest Path Distance (d)')
    plt.ylabel('Frequency')
    plt.xticks(distances)
    plt.yscale('log') # Log scale is often useful for frequency
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(f'{dataset_name}_distance_distribution.png', dpi=300)
    plt.close()
    print(f"Saved {dataset_name}_distance_distribution.png")


if __name__ == '__main__':
    analyze_graph('twitter-small')
    analyze_graph('twitter-larger')