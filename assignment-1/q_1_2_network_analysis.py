import networkx as nx
import pandas as pd

# Load data
medium_edges = pd.read_csv('snacs2025-student4581733-medium.tsv', sep='\t', header=None, names=['source', 'target'])
large_edges = pd.read_csv('snacs2025-student4581733-large.tsv', sep='\t', header=None, names=['source', 'target'])

# Create directed graphs
G_medium = nx.from_pandas_edgelist(medium_edges, source='source', target='target', create_using=nx.DiGraph())
G_large = nx.from_pandas_edgelist(large_edges, source='source', target='target', create_using=nx.DiGraph())

# Test outputs
print("Medium graph: nodes =", G_medium.number_of_nodes(), ", edges =", G_medium.number_of_edges())
print("Large graph: nodes =", G_large.number_of_nodes(), ", edges =", G_large.number_of_edges())
