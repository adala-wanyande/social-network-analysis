#!/usr/bin/env python3
"""
Fast, Single-Pass, Sampled Analysis of the Full Twitter Dataset
----------------------------------------------------------------
This script is designed for speed under tight time constraints.
It reads a fraction of the dataset, builds an in-memory graph,
calculates statistics on the sample, and extrapolates counts for the report.
It does NOT write any intermediate files.
"""
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re
import random
import time

# --- KEY PARAMETERS TO CONFIGURE ---
# Path to the massive TSV file
INPUT_TSV = '/vol/share/groups/liacs/scratch/SNACS/2020/twitter.tsv' 
# INPUT_TSV = 'twitter.tsv' # For local testing

# Fraction of the file to sample (0.1 = 10%). Maximum allowed is 0.1 as requested.
SAMPLING_FRACTION = 0.1

# Keep edges with weight > X. (1 means at least 2 mentions).
WEIGHT_THRESHOLD = 1

# Number of nodes to sample for expensive distance calculations.
DISTANCE_SAMPLE_SIZE = 500

# Number of lines per chunk for processing.
CHUNKSIZE = 1_000_000

# --- ANALYSIS SCRIPT ---

def analyze_sampled_graph(input_path, sampling_fraction, weight_threshold, chunksize):
    """
    Performs a single-pass, in-memory analysis of a sample of the graph.
    """
    start_time = time.time()
    print("--- Starting Fast, Sampled Analysis ---")
    print(f"Sampling {sampling_fraction:.1%} of the file: {input_path}")
    print(f"Keeping edges with weight > {weight_threshold}")
    print("This will all be done in-memory without intermediate files.\n")

    # --- 1. Stream and Sample the Data ---
    edge_counts = Counter()
    username_regex = re.compile(r'@([A-Za-z0-9_]{1,15})\b')
    
    # Estimate total chunks to determine when to stop sampling
    # The full dataset has ~257M lines, so at 1M/chunk, that's ~257 chunks.
    # We will stop after processing sampling_fraction * 257 chunks.
    max_chunks_to_process = int(257 * sampling_fraction)
    print(f"Will process approximately {max_chunks_to_process} chunks of {chunksize:,} lines each.")

    reader = pd.read_csv(
        input_path, sep='\t', header=None, names=['timestamp', 'sender', 'content'],
        chunksize=chunksize, on_bad_lines='skip', dtype={'content': str}
    )

    total_lines = 0
    for i, chunk in enumerate(reader):
        if i >= max_chunks_to_process:
            print(f"\nReached sampling limit of {max_chunks_to_process} chunks. Stopping read.")
            break
        
        chunk.dropna(inplace=True)
        for row in chunk.itertuples():
            sender = str(row.sender).lower()
            mentions = username_regex.findall(str(row.content))
            for mentioned_user in mentions:
                mentioned_user_lower = mentioned_user.lower()
                if sender != mentioned_user_lower:
                    edge_counts[(sender, mentioned_user_lower)] += 1
        
        total_lines += len(chunk)
        print(f"  Processed chunk {i+1}/{max_chunks_to_process} ({total_lines:,} lines so far).")

    print(f"\n--- 2. Filtering and Building In-Memory Graph ---")
    filtered_edges = {edge: weight for edge, weight in edge_counts.items() if weight > weight_threshold}
    
    G = nx.DiGraph()
    for (source, target), weight in filtered_edges.items():
        G.add_edge(source, target, weight=weight)

    time_to_build = time.time() - start_time
    print(f"Sampled graph built in {time_to_build:.2f} seconds.")
    print(f"Sampled graph has {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges.")

    # --- 3. Calculate Statistics on the Sampled Graph ---
    print("\n--- 4. Calculating Statistics ---")

    # --- A. Counts from the sample (will be scaled) ---
    nodes_sample = G.number_of_nodes()
    edges_sample = G.number_of_edges()
    
    wccs_sample = list(nx.weakly_connected_components(G))
    wccs_count_sample = len(wccs_sample)
    giant_comp_nodes_sample = max(wccs_sample, key=len) if wccs_sample else set()
    giant_comp_size_sample = len(giant_comp_nodes_sample)

    sccs_sample = list(nx.strongly_connected_components(G))
    sccs_count_sample = len(sccs_sample)
    largest_scc_size_sample = len(max(sccs_sample, key=len)) if sccs_sample else 0

    # --- B. Scale counts up to estimate the full graph ---
    est_nodes = int(nodes_sample / sampling_fraction)
    est_edges = int(edges_sample / sampling_fraction)
    est_wccs = int(wccs_count_sample / sampling_fraction)
    est_sccs = int(sccs_count_sample / sampling_fraction)
    # The size of the giant/largest components themselves are not typically scaled.
    # We report the size of the largest component found in our sample.
    
    # --- C. Structural properties (NOT scaled) ---
    density_sample = nx.density(G) # Density is scale-invariant

    # For expensive metrics, subsample from the giant component
    G_gc_undirected = G.subgraph(giant_comp_nodes_sample).to_undirected()
    if G_gc_undirected.number_of_nodes() > DISTANCE_SAMPLE_SIZE:
        nodes_for_approx = random.sample(list(G_gc_undirected.nodes()), k=DISTANCE_SAMPLE_SIZE)
    else:
        nodes_for_approx = list(G_gc_undirected.nodes())
    
    print(f"Approximating clustering and distance using {len(nodes_for_approx)} nodes...")
    
    # Approximated Average Clustering
    avg_clustering_approx = nx.average_clustering(G, nodes=nodes_for_approx)

    # Approximated Average Distance and Distribution
    all_path_lengths = []
    for node in nodes_for_approx:
        try:
            path_lengths = nx.shortest_path_length(G_gc_undirected, source=node)
            del path_lengths[node] # remove self-distance
            all_path_lengths.extend(path_lengths.values())
        except nx.NetworkXError:
            continue # Skip if node is not in the giant component subgraph for some reason
            
    avg_distance_approx = np.mean(all_path_lengths) if all_path_lengths else 0
    dist_distribution = Counter(all_path_lengths)

    # --- 5. Print Statistics for LaTeX Report ---
    print("\n\n--- Statistics Summary (Copy this into your LaTeX Table) ---")
    print(f"Number of Nodes (Estimated) & {est_nodes:,} \\\\")
    print(f"Number of Edges (Estimated) & {est_edges:,} \\\\")
    print(f"Weakly Connected Components (Est.) & {est_wccs:,} (Giant found in sample: {giant_comp_size_sample:,} nodes) \\\\")
    print(f"Strongly Connected Components (Est.) & {est_sccs:,} (Largest found in sample: {largest_scc_size_sample:,} nodes) \\\\")
    print(f"Density (from sample) & {density_sample:.2e} \\\\")
    print(f"Avg. Clustering Coefficient (Approx.) & {avg_clustering_approx:.4f} \\\\")
    print(f"Avg. Distance (Giant Comp., Approx.) & {avg_distance_approx:.4f} \\\\")
    
    # --- 6. Generate Plots ---
    print("\n--- 6. Generating Plots ---")
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.loglog(list(Counter(in_degrees).keys()), list(Counter(in_degrees).values()), 'bo', ms=2, alpha=0.5)
    ax1.set_title(f'Indegree Distribution ({sampling_fraction:.0%} Sample)')
    ax1.set_xlabel("Indegree (k)")
    ax1.set_ylabel("Count")
    ax2.loglog(list(Counter(out_degrees).keys()), list(Counter(out_degrees).values()), 'ro', ms=2, alpha=0.5)
    ax2.set_title(f'Outdegree Distribution ({sampling_fraction:.0%} Sample)')
    ax2.set_xlabel("Outdegree (k)")
    plt.savefig('twitter-full_degree_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved degree distribution plot.")

    if dist_distribution:
        distances, counts = zip(*sorted(dist_distribution.items()))
        plt.figure(figsize=(8, 6))
        plt.bar(distances, counts, color='skyblue')
        plt.title(f'Distance Distribution (Approx. from {len(nodes_for_approx)} nodes in Giant Comp.)')
        plt.xlabel("Shortest Path Distance")
        plt.ylabel("Count (Log Scale)")
        plt.yscale('log')
        plt.savefig('twitter-full_distance_distribution.png', dpi=300, bbox_inches='tight')
        print("Saved distance distribution plot.")
    else:
        print("Could not generate distance plot (no paths found).")

    total_time = time.time() - start_time
    print(f"\n--- Analysis Complete in {total_time:.2f} seconds ---")


if __name__ == '__main__':
    analyze_sampled_graph(INPUT_TSV, SAMPLING_FRACTION, WEIGHT_THRESHOLD, CHUNKSIZE)