import networkx as nx
import pandas as pd
import community as community_louvain
import os

def detect_and_analyze_communities(dataset_name):
    """
    Detects communities in the giant component of the graph
    and analyzes their composition.
    """
    edgelist_path = f'{dataset_name}-edgelist.csv'
    if not os.path.exists(edgelist_path):
        print(f"Error: {edgelist_path} not found. Please run the extraction script first.")
        return

    print(f"\n--- Community Detection for {dataset_name} ---")

    # 1. Load the directed graph
    df = pd.read_csv(edgelist_path)
    G = nx.from_pandas_edgelist(
        df,
        source='Source',
        target='Target',
        edge_attr='Weight',
        create_using=nx.DiGraph()
    )

    # 2. Extract the Giant Component
    wccs = list(nx.weakly_connected_components(G))
    giant_component_nodes = max(wccs, key=len)
    G_gc = G.subgraph(giant_component_nodes).copy()

    print(f"Extracted Giant Component with {G_gc.number_of_nodes()} nodes and {G_gc.number_of_edges()} edges.")

    # 3. Prepare Graph for Community Detection
    G_gc_undirected = G_gc.to_undirected()

    # 4. Apply the Louvain Algorithm
    print("Running Louvain community detection algorithm...")
    partition = community_louvain.best_partition(G_gc_undirected, weight='Weight')
    modularity = community_louvain.modularity(partition, G_gc_undirected, weight='Weight')
    
    num_communities = len(set(partition.values()))
    print(f"Found {num_communities} communities with a modularity of {modularity:.4f}.")

    # 5. Analyze the Communities
    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)

    sorted_communities = sorted(communities.values(), key=len, reverse=True)

    print("\n--- Top 5 Largest Communities ---")
    
    for i, community_nodes in enumerate(sorted_communities[:5]):
        community_subgraph = G_gc.subgraph(community_nodes)
        indegrees = dict(community_subgraph.in_degree())
        indegrees_in_community = {node: degree for node, degree in indegrees.items() if degree > 0}
        top_members = sorted(indegrees_in_community.items(), key=lambda item: item[1], reverse=True)
        
        print(f"\nCommunity {i+1}:")
        print(f"  - Size: {len(community_nodes)} users")
        print(f"  - Top 5 Members (by indegree): {[user for user, score in top_members[:5]]}")

if __name__ == '__main__':
    # We focus specifically on the small dataset as requested.
    detect_and_analyze_communities('twitter-small')