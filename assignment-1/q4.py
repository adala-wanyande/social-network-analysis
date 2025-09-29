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
# Function to compute components stats
# ------------------------
def components_stats(G, name):
    # Weakly connected components
    wccs = list(nx.weakly_connected_components(G))
    num_wcc = len(wccs)
    largest_wcc = max(wccs, key=len)
    G_lwcc = G.subgraph(largest_wcc)
    nodes_lwcc = G_lwcc.number_of_nodes()
    edges_lwcc = G_lwcc.number_of_edges()

    # Strongly connected components
    sccs = list(nx.strongly_connected_components(G))
    num_scc = len(sccs)
    largest_scc = max(sccs, key=len)
    G_lscc = G.subgraph(largest_scc)
    nodes_lscc = G_lscc.number_of_nodes()
    edges_lscc = G_lscc.number_of_edges()

    # Print results
    print(f"{name} Graph:")
    print(f"  Number of weakly connected components: {num_wcc}")
    print(f"  Largest WCC: nodes = {nodes_lwcc}, edges = {edges_lwcc}")
    print(f"  Number of strongly connected components: {num_scc}")
    print(f"  Largest SCC: nodes = {nodes_lscc}, edges = {edges_lscc}")
    print()

# ------------------------
# Compute and print for medium and large graphs
# ------------------------
components_stats(G_medium, "Medium")
components_stats(G_large, "Large")
