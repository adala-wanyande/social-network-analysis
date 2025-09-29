import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Load data
medium_file = 'snacs2025-student4581733-medium.tsv'
edges = pd.read_csv(medium_file, sep='\t', header=None, names=['source', 'target'])

# Create directed graph
G = nx.from_pandas_edgelist(edges, source='source', target='target', create_using=nx.DiGraph())

# Compute centrality measures
degree_centrality = dict(G.degree())          # node size
betweenness_centrality = nx.betweenness_centrality(G)  # node color

# Scale node sizes
node_sizes = [v * 10 for v in degree_centrality.values()]

# Map betweenness to color using a dramatic colormap
bet_values = list(betweenness_centrality.values())
cmap = plt.cm.inferno  # more contrast than plasma
node_colors = [cmap(v/max(bet_values)) if max(bet_values) > 0 else cmap(0.05) for v in bet_values]

# Layout with spring_layout
pos = nx.spring_layout(G, seed=42, k=0.1, iterations=200)

# Draw network with distinct borders
plt.figure(figsize=(12,12))
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    edgecolors='white',    # distinct border
    linewidths=0.8,
    alpha=0.5
)
nx.draw_networkx_edges(G, pos, alpha=0.8, edge_color='gray', connectionstyle='arc3,rad=0.2', width=0.3)  # curved edges
plt.title("Medium Social Network: Node size = Degree, Node color = Betweenness")
plt.axis('off')
plt.tight_layout()
plt.savefig('medium_network_visualization_curved_borders.png', dpi=300)
plt.show()
