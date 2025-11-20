import networkx as nx
import os
from pathlib import Path
from collections import defaultdict

# Define the path to the networks folder relative to the script's location
DATA_DIR = Path(__file__).parent / "networks"

# Configuration dictionary to map dataset names to their file info and properties
# This makes it easy to add new datasets in the future.
DATASET_CONFIG = {
    'Wiki-Vote': {
        'filename': 'Wiki-vote.txt',
        'pretty_name': 'Wiki-Vote',
        'is_weighted': False,
        'comment_char': '#'
    },
    'Facebook': {
        'filename': 'facebook_combined.txt',
        'pretty_name': 'Facebook',
        'is_weighted': False,
        'comment_char': '#'
    },
    'Email-EU-Core': {
        'filename': 'email-Eu-core-department-labels.txt',
        'pretty_name': 'Email-EU-Core',
        'is_weighted': False,
        'comment_char': '#'
    },
    'CA-GrQc': {
        'filename': 'CA-GrQc.txt',
        'pretty_name': 'CA-GrQc',
        'is_weighted': False,
        'comment_char': '#'
    }
}


def _load_stackoverflow() -> nx.Graph:
    """
    Custom loader for the Stack Overflow dataset.
    
    The dataset is a temporal edge list (u, v, timestamp). We create a static,
    weighted graph where the edge weight is the inverse of the number of
    interactions between two users. A stronger connection (more interactions)
    results in a smaller weight (shorter path distance).
    
    Returns:
        nx.Graph: The loaded and aggregated weighted graph.
    """
    config = DATASET_CONFIG['stackoverflow']
    filepath = DATA_DIR / config['filename']
    print(f"  -> Applying custom loader for Stack Overflow...")

    # Use a defaultdict to count interactions for each edge pair
    edge_counts = defaultdict(int)
    
    print(f"  -> Reading and aggregating edges from {filepath}...")
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith(config['comment_char']):
                continue
            parts = line.strip().split()
            # The file format can be different, often space or comma separated
            u, v = int(parts[0]), int(parts[1])
            # Ensure canonical edge representation (u, v) where u < v
            edge = tuple(sorted((u, v)))
            edge_counts[edge] += 1
            
    G = nx.Graph()
    print("  -> Creating weighted graph from aggregated interactions...")
    
    weighted_edges = [
        (u, v, 1.0 / count) for (u, v), count in edge_counts.items()
    ]
    
    G.add_weighted_edges_from(weighted_edges)
    return G


def load_and_preprocess_graph(dataset_name: str) -> nx.Graph:
    """
    Loads a graph by its short name, performs standardized pre-processing,
    and returns the final graph object.

    Pre-processing steps:
      1. Loads the graph from the corresponding file in the './networks' folder.
      2. Converts the graph to undirected if needed.
      3. Extracts the Largest Connected Component (LCC).

    Args:
      dataset_name (str): The short name of the dataset (e.g., 'livejournal').

    Returns:
      nx.Graph: The pre-processed graph ready for experiments.
    """
    # dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: '{dataset_name}'. "
                         f"Available datasets are: {list(DATASET_CONFIG.keys())}")

    config = DATASET_CONFIG[dataset_name]
    filepath = DATA_DIR / config['filename']

    print(f"\n--- Loading and Pre-processing Dataset: {config['pretty_name'].upper()} ---")

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: {filepath}\n"
            f"Please download '{config['filename']}' and place it in the '{DATA_DIR}' folder."
        )

    # --- Step 1: Load Graph ---
    print(f"  -> Reading graph from {filepath}...")
    if config['is_weighted']:
        # For weighted graphs
        G = nx.read_weighted_edgelist(
            filepath,
            comments=config['comment_char'],
            nodetype=int
        )
    else:
        G = nx.read_edgelist(
            filepath,
            comments=config['comment_char'],
            nodetype=int
        )

    print(f"  -> Initial graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges.")

    # --- Step 2: Ensure Undirected ---
    if nx.is_directed(G):
        print("  -> Converting directed graph to undirected.")
        G = G.to_undirected()

    # --- Step 3: Extract Largest Connected Component (LCC) ---
    if nx.is_connected(G):
        print("  -> Graph is already connected. Using the full graph.")
        G_lcc = G
    else:
        print("  -> Graph is not connected. Extracting the Largest Connected Component (LCC)...")
        lcc_nodes = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(lcc_nodes).copy()
        print(f"  -> LCC extracted. Graph size reduced:")
        print(f"     Nodes: {G.number_of_nodes():,} -> {G_lcc.number_of_nodes():,}")
        print(f"     Edges: {G.number_of_edges():,} -> {G_lcc.number_of_edges():,}")

    print(f"--- Finished processing {config['pretty_name'].upper()}. Ready for analysis. ---")
    return G_lcc


# This block allows you to run the script directly for testing purposes
if __name__ == '__main__':
    # Build the test list from DATASET_CONFIG to ensure consistency
    datasets_to_test = list(DATASET_CONFIG.keys())

    for name in datasets_to_test:
        pretty = DATASET_CONFIG[name].get('pretty_name', name)
        try:
            graph = load_and_preprocess_graph(name)
            print(f"Successfully loaded and processed '{pretty}'.")
            print(f"Final graph info: Nodes={graph.number_of_nodes()}, Edges={graph.number_of_edges()}\n")
        except (ValueError, FileNotFoundError) as e:
            print(f"Error loading '{pretty}': {e}\n")