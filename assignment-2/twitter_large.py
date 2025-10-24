import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import re
import os
import random

# --- Configuration ---
# IMPORTANT: Adjust these paths to match your university environment
INPUT_TSV = '/vol/share/groups/liacs/scratch/SNACS/twitter.tsv' 
# INPUT_TSV = 'twitter.tsv' # For local testing if you download a sample
PASS1_OUTPUT_CSV = 'twitter-full-filtered-edgelist.csv'

# --- Parameters ---
CHUNKSIZE = 1_000_000  # Process 1 million lines at a time
WEIGHT_THRESHOLD = 2   # Keep edges with weight > 2 (i.e., at least 3 mentions)

def run_pass_1(input_path, output_path, chunksize, weight_threshold):
    """
    Streams through the massive TSV file, aggregates edge counts,
    filters by weight, and saves a manageable edgelist.
    """
    if os.path.exists(output_path):
        print(f"{output_path} already exists. Skipping Pass 1.")
        return

    print("--- Starting Pass 1: Streaming, Aggregating, and Filtering ---")
    print(f"Reading from: {input_path}")
    print(f"Chunksize: {chunksize:,} lines")
    print(f"Weight Threshold: Keeping edges with weight > {weight_threshold}")

    edge_counts = Counter()
    username_regex = re.compile(r'@([A-Za-z0-9_]{1,15})\b')
    
    # Use pandas for robust, chunked TSV reading
    reader = pd.read_csv(
        input_path,
        sep='\t',
        header=None,
        names=['timestamp', 'sender', 'content'],
        chunksize=chunksize,
        on_bad_lines='skip', # Important for real-world messy data
        dtype={'content': str} # Ensure content is read as string
    )

    total_lines = 0
    for i, chunk in enumerate(reader):
        chunk.dropna(inplace=True) # Drop rows with missing values
        for row in chunk.itertuples():
            sender = str(row.sender).lower()
            mentions = username_regex.findall(str(row.content))
            for mentioned_user in mentions:
                mentioned_user_lower = mentioned_user.lower()
                if sender != mentioned_user_lower: # Exclude self-mentions
                    edge_counts[(sender, mentioned_user_lower)] += 1
        
        total_lines += len(chunk)
        print(f"  Processed chunk {i+1} ({total_lines:,} lines so far). Edge count: {len(edge_counts):,}")

    print("\n--- Filtering edges by weight threshold ---")
    filtered_edges = {edge: weight for edge, weight in edge_counts.items() if weight > weight_threshold}
    
    print(f"Original unique edges: {len(edge_counts):,}")
    print(f"Filtered unique edges: {len(filtered_edges):,}")

    print(f"--- Writing filtered edgelist to {output_path} ---")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        f.write('Source,Target,Weight\n')
        for (source, target), weight in filtered_edges.items():
            f.write(f'{source},{target},{weight}\n')
    print("--- Pass 1 Complete ---")


def run_pass_2(edgelist_path):
    """
    Loads the filtered graph and computes the required statistics and plots.
    This is nearly identical to the script for twitter-larger.
    """
    print("\n--- Starting Pass 2: In-Memory Analysis and Visualization ---")
    print(f"Loading filtered graph from: {edgelist_path}")

    df = pd.read_csv(edgelist_path)
    G = nx.from_pandas_edgelist(
        df, source='Source', target='Target', edge_attr='Weight', create_using=nx.DiGraph()
    )

    # --- Calculate Statistics ---
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    wccs = list(nx.weakly_connected_components(G))
    giant_component_nodes = max(wccs, key=len)
    G_gc_undirected = G.subgraph(giant_component_nodes).to_undirected()

    print("\n--- Approximating expensive metrics by sampling 2000 nodes from the giant component ---")
    nodes_sample = random.sample(list(G_gc_undirected.nodes()), k=min(2000, G_gc_undirected.number_of_nodes()))
    
    # Approximated Average Clustering
    avg_clustering = nx.average_clustering(G, nodes=nodes_sample)

    # Approximated Average Distance and Distribution
    all_path_lengths = []
    for node in nodes_sample:
        path_lengths = nx.shortest_path_length(G_gc_undirected, source=node)
        del path_lengths[node]
        all_path_lengths.extend(path_lengths.values())
            
    avg_distance = np.mean(all_path_lengths) if all_path_lengths else 0
    dist_distribution = Counter(all_path_lengths)
    
    # --- Print Statistics for LaTeX Report ---
    print("\n--- Statistics Summary (for LaTeX Table) ---")
    print(f"Number of Nodes & {num_nodes:,} \\\\")
    print(f"Number of Edges & {num_edges:,} \\\\")
    print(f"Weakly Connected Components & {len(wccs):,} (Giant: {len(giant_component_nodes):,} nodes) \\\\")
    sccs = nx.strongly_connected_components(G)
    largest_scc_size = len(max(sccs, key=len))
    print(f"Strongly Connected Components & {nx.number_strongly_connected_components(G):,} (Largest: {largest_scc_size:,} nodes) \\\\")
    print(f"Density & {nx.density(G):.2e} \\\\")
    print(f"Avg. Clustering Coefficient (Approx.) & {avg_clustering:.4f} \\\\")
    print(f"Avg. Distance (Giant Component, Approx.) & {avg_distance:.4f} \\\\")
    
    # --- Generate Plots ---
    print("\n--- Generating Plots ---")
    # Degree Distribution
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.loglog(list(Counter(in_degrees).keys()), list(Counter(in_degrees).values()), 'bo', ms=2, alpha=0.5)
    ax1.set_title('Indegree Distribution (Full Dataset, Filtered)')
    ax2.loglog(list(Counter(out_degrees).keys()), list(Counter(out_degrees).values()), 'ro', ms=2, alpha=0.5)
    ax2.set_title('Outdegree Distribution (Full Dataset, Filtered)')
    plt.savefig('twitter-full_degree_distribution.png', dpi=300)
    print("Saved degree distribution plot.")

    # Distance Distribution
    distances, counts = zip(*sorted(dist_distribution.items()))
    plt.figure(figsize=(8, 6))
    plt.bar(distances, counts, color='skyblue')
    plt.title('Distance Distribution of Giant Component (Full Dataset, Approx.)')
    plt.yscale('log')
    plt.savefig('twitter-full_distance_distribution.png', dpi=300)
    print("Saved distance distribution plot.")
    print("--- Pass 2 Complete ---")


if __name__ == '__main__':
    run_pass_1(INPUT_TSV, PASS1_OUTPUT_CSV, CHUNKSIZE, WEIGHT_THRESHOLD)
    run_pass_2(PASS1_OUTPUT_CSV)