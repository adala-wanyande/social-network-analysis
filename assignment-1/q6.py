import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random
import os

# ------------------------
# Load data
# ------------------------
medium_file = 'snacs2025-student4581733-medium.tsv'
large_file = 'snacs2025-student4581733-large.tsv'

medium_edges = pd.read_csv(medium_file, sep='\t', header=None, names=['source', 'target'])
large_edges = pd.read_csv(large_file, sep='\t', header=None, names=['source', 'target'])

# ------------------------
# Create directed graphs
# ------------------------
G_medium = nx.from_pandas_edgelist(medium_edges, source='source', target='target', create_using=nx.DiGraph())
G_large = nx.from_pandas_edgelist(large_edges, source='source', target='target', create_using=nx.DiGraph())

# ------------------------
# Output folder for plots
# ------------------------
output_dir = 'plots_distance'
os.makedirs(output_dir, exist_ok=True)

# ------------------------
# Get largest weakly connected component
# ------------------------
def get_lwcc(G):
    wccs = list(nx.weakly_connected_components(G))
    largest_wcc_nodes = max(wccs, key=len)
    return G.subgraph(largest_wcc_nodes).copy()

# ------------------------
# Distance distribution function
# ------------------------
def distance_distribution(G, name, filename, sample_size=None):
    G_undir = G.to_undirected()
    
    # Compute distances
    if sample_size:
        nodes = list(G_undir.nodes())
        distances = []
        for _ in range(sample_size):
            source = random.choice(nodes)
            sp_lengths = nx.single_source_shortest_path_length(G_undir, source)
            distances.extend(sp_lengths.values())
    else:
        distances = []
        for sp_lengths in nx.all_pairs_shortest_path_length(G_undir):
            distances.extend(sp_lengths[1].values())
    
    # Plot histogram
    plt.figure(figsize=(6,4))
    plt.hist(distances, bins=range(max(distances)+1), color='lightgreen', edgecolor='black', density=True)
    plt.xlabel("Distance (shortest path length)")
    plt.ylabel("Fraction of node pairs")
    plt.yscale('log') 
    plt.title(f"{name}: Distance Distribution in LWCC")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"{name} LWCC: Distance distribution plot saved as '{filename}'")

# ------------------------
# Medium graph
# ------------------------
G_medium_lwcc = get_lwcc(G_medium)
distance_distribution(
    G_medium_lwcc,
    "Medium",
    os.path.join(output_dir, "medium_distance_distribution.png")
)

# ------------------------
# Large graph (approximation)
# ------------------------
G_large_lwcc = get_lwcc(G_large)
distance_distribution(
    G_large_lwcc,
    "Large",
    os.path.join(output_dir, "large_distance_distribution.png"),
    sample_size=1000  # sample 1000 nodes for approximation
)
