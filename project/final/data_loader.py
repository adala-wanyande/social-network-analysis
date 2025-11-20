import networkx as nx
from pathlib import Path

DATA_DIR = Path(__file__).parent / "networks"

# --- DEFINITIVE FINAL 7-DATASET CONFIGURATION ---
DATASET_CONFIG = {
    'wiki-vote': {
        'filename': 'Wiki-Vote.txt',
        'pretty_name': 'Wiki-Vote',
        'is_weighted': False,
        'comment_char': '#'
    },
    'facebook': {
        'filename': 'facebook_combined.txt',
        'pretty_name': 'Facebook',
        'is_weighted': False,
        'comment_char': '#'
    },
    'email-eu': {
        'filename': 'email-Eu-core.txt',
        'pretty_name': 'Email-EU-Core',
        'is_weighted': False,
        'comment_char': '#'
    },
    'ca-grqc': {
        'filename': 'CA-GrQc.txt',
        'pretty_name': 'CA-GrQc',
        'is_weighted': False,
        'comment_char': '#'
    },
    'lesmis': {
        'filename': 'lesmis.gml',
        'pretty_name': 'Les Misérables',
        'is_weighted': True,
        'loader_func': 'gml'
    },
    'norwegian-boards': {
        'filename': 'norwegian-boards.txt',
        'pretty_name': 'Norwegian Boards',
        'is_weighted': True,
        'loader_func': 'edgelist'
    },
    'facebook-forum': {
        'filename': 'facebook-forum.txt',
        'pretty_name': 'Facebook Forum',
        'is_weighted': True,
        'loader_func': 'edgelist'
    }
}

def load_and_preprocess_graph(dataset_name: str) -> nx.Graph:
    """
    Loads a graph by its short name, performs standardized pre-processing,
    and returns the final graph object.
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: '{dataset_name}'.")
    config = DATASET_CONFIG[dataset_name]
    filepath = DATA_DIR / config['filename']
    pretty_name = config['pretty_name']
    print(f"\n--- Loading and Pre-processing Dataset: {pretty_name.upper()} ---")
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    loader = config.get('loader_func')
    if loader == 'gml':
        print(f"  -> Reading GML graph from {filepath}...")
        G = nx.read_gml(filepath, label='id')
    else:
        print(f"  -> Reading edgelist from {filepath}...")
        if config['is_weighted']:
            G = nx.read_weighted_edgelist(filepath, comments=config.get('comment_char', '#'), nodetype=int)
        else:
            G = nx.read_edgelist(filepath, comments=config.get('comment_char', '#'), nodetype=int)
    print(f"  -> Initial graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges.")
    if config['is_weighted']:
        print("  -> Inverting edge weights for shortest path calculation (distance = 1/weight)...")
        for u, v, d in G.edges(data=True):
            raw_weight = d.get('weight', d.get('value', 1.0))
            if raw_weight > 0:
                d['weight'] = 1.0 / raw_weight
            else:
                d['weight'] = float('inf')
    if nx.is_directed(G):
        print("  -> Converting directed graph to undirected.")
        G = G.to_undirected()
    if nx.is_connected(G):
        print("  -> Graph is already connected. Using the full graph.")
        G_lcc = G
    else:
        print("  -> Graph is not connected. Extracting the LCC...")
        lcc_nodes = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(lcc_nodes).copy()
        print(f"  -> LCC extracted: {G_lcc.number_of_nodes():,} nodes, {G_lcc.number_of_edges():,} edges.")
    print(f"--- Finished processing {pretty_name.upper()}. Ready for analysis. ---")
    return G_lcc