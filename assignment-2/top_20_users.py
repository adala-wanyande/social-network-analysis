import networkx as nx
import pandas as pd
from scipy.stats import kendalltau
import random
import os

def analyze_centrality(dataset_name, directed=True):
    """
    Calculates centrality measures, finds top 20 users,
    and compares the resulting rankings.
    """
    edgelist_path = f'{dataset_name}-edgelist.csv'
    if not os.path.exists(edgelist_path):
        print(f"Error: {edgelist_path} not found. Please run the extraction script first.")
        return

    print(f"\n--- Analyzing Centrality for {dataset_name} ---")

    # Load the directed graph
    df = pd.read_csv(edgelist_path)
    G = nx.from_pandas_edgelist(df, source='Source', target='Target', create_using=nx.DiGraph())

    # --- Centrality Calculations ---

    # 1. Degree Centrality (Indegree)
    # For a mention graph, indegree represents popularity or prestige.
    in_degree_centrality = nx.in_degree_centrality(G)

    # 2. Closeness Centrality
    # Measures how quickly a node can be reached from other nodes.
    # It's only well-defined within a connected component. NetworkX handles this.
    closeness_centrality = nx.closeness_centrality(G)

    # 3. Betweenness Centrality
    # This is computationally expensive. We approximate for the larger graph.
    if G.number_of_nodes() > 5000:
        print("Approximating betweenness centrality by sampling 2000 nodes...")
        k = 2000
        betweenness_centrality = nx.betweenness_centrality(G, k=k, seed=42)
    else:
        betweenness_centrality = nx.betweenness_centrality(G)

    # --- Create Top 20 Rankings ---
    
    def get_top_20(centrality_dict):
        # Sort the dictionary by value and return the top 20 user names
        sorted_users = sorted(centrality_dict.items(), key=lambda item: item[1], reverse=True)
        return [user for user, score in sorted_users[:20]]

    top_20_indegree = get_top_20(in_degree_centrality)
    top_20_closeness = get_top_20(closeness_centrality)
    top_20_betweenness = get_top_20(betweenness_centrality)

    # Create a pandas DataFrame for nice printing
    rankings_df = pd.DataFrame({
        'Rank': range(1, 21),
        'Indegree Centrality': top_20_indegree,
        'Closeness Centrality': top_20_closeness,
        'Betweenness Centrality': top_20_betweenness
    }).set_index('Rank')

    print(f"\nTop 20 Users by Centrality for {dataset_name}:")
    print(rankings_df)
    

    # --- Compare Ranking Similarity using Kendall's Tau ---
    
    # We compare the ordered lists of the top 20 users.
    # Kendall's Tau is robust for comparing two ranked lists.
    # It measures the proportion of concordant pairs minus discordant pairs.
    # Tau = 1: Perfect agreement. Tau = -1: Perfect disagreement. Tau = 0: No correlation.

    tau_degree_betweenness, _ = kendalltau(top_20_indegree, top_20_betweenness)
    tau_degree_closeness, _ = kendalltau(top_20_indegree, top_20_closeness)
    tau_betweenness_closeness, _ = kendalltau(top_20_closeness, top_20_betweenness)

    print("\n--- Ranking Similarity (Kendall's Tau) ---")
    print(f"Indegree vs. Betweenness:   {tau_degree_betweenness:.4f}")
    print(f"Indegree vs. Closeness:     {tau_degree_closeness:.4f}")
    print(f"Closeness vs. Betweenness:  {tau_betweenness_closeness:.4f}")
    print("-" * 45)


if __name__ == '__main__':
    analyze_centrality('twitter-small')
    # analyze_centrality('twitter-larger')