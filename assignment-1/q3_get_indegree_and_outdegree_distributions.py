import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import collections
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
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# ------------------------
# Compute degree distributions
# ------------------------
def get_degrees(G):
    in_degrees = [deg for node, deg in G.in_degree()]
    out_degrees = [deg for node, deg in G.out_degree()]
    return in_degrees, out_degrees

in_med, out_med = get_degrees(G_medium)
in_large, out_large = get_degrees(G_large)

# ------------------------
# Plotting functions
# ------------------------
def plot_degree_hist(degrees, title, filename):
    plt.figure(figsize=(6,4))
    plt.hist(degrees, bins=50, color='skyblue', edgecolor='black', log=True)
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes (log scale)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_degree_loglog(degrees, title, filename):
    degree_count = collections.Counter(degrees)
    deg, cnt = zip(*degree_count.items())
    plt.figure(figsize=(6,4))
    plt.scatter(deg, cnt, alpha=0.6)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ------------------------
# Medium graph plots
# ------------------------
plot_degree_hist(in_med, "Medium Graph: In-degree Distribution", os.path.join(output_dir, "medium_in_deg_hist.png"))
plot_degree_hist(out_med, "Medium Graph: Out-degree Distribution", os.path.join(output_dir, "medium_out_deg_hist.png"))
plot_degree_loglog(in_med, "Medium Graph: In-degree Distribution (log-log)", os.path.join(output_dir, "medium_in_deg_loglog.png"))
plot_degree_loglog(out_med, "Medium Graph: Out-degree Distribution (log-log)", os.path.join(output_dir, "medium_out_deg_loglog.png"))

# ------------------------
# Large graph plots
# ------------------------
plot_degree_hist(in_large, "Large Graph: In-degree Distribution", os.path.join(output_dir, "large_in_deg_hist.png"))
plot_degree_hist(out_large, "Large Graph: Out-degree Distribution", os.path.join(output_dir, "large_out_deg_hist.png"))
plot_degree_loglog(in_large, "Large Graph: In-degree Distribution (log-log)", os.path.join(output_dir, "large_in_deg_loglog.png"))
plot_degree_loglog(out_large, "Large Graph: Out-degree Distribution (log-log)", os.path.join(output_dir, "large_out_deg_loglog.png"))

print(f"All plots saved in '{output_dir}' folder.")
