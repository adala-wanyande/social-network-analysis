import networkx as nx
import pandas as pd

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
# Compute average clustering coefficient
# ------------------------
def compute_avg_clustering(G, name):
    # Convert to undirected to account for directionality
    G_undir = G.to_undirected()
    avg_clust = nx.average_clustering(G_undir)
    print(f"{name} Graph: Average clustering coefficient = {avg_clust:.4f}")
    print("  (Direction ignored: treated graph as undirected for clustering)\n")

compute_avg_clustering(G_medium, "Medium")
compute_avg_clustering(G_large, "Large")
