#!/usr/bin/env python3
"""
Bounding Diameters Algorithm (Verbose CLI)
-----------------------------------------
This script computes the diameter of an undirected graph using an
iterative bounding technique. It is verbose: all steps, assumptions,
and node selection strategies are explained inline.

The algorithm maintains:
- A lower bound (LB) for each node: minimum eccentricity found so far
- An upper bound (UB) for each node: maximum possible eccentricity
It iteratively selects nodes to BFS based on the strategy:
1. Alternate between node with largest UB and node with smallest LB
2. Break ties by taking the node with the highest degree

We count iterations (BFS runs) and compare with naive all-pairs BFS.
"""

from collections import deque, defaultdict

# ------------------------- Graph Input ----------------------------

def parse_input():
    """
    Reads graph from stdin in adjacency list or edge list format.
    - Adjacency list: "A: B C"
    - Edge list: "A B"
    Returns a dict: node -> set of neighbors
    """
    print("Enter your graph (end with empty line):")
    lines = []
    while True:
        try:
            line = input().strip()
        except EOFError:
            break
        if not line:
            break
        lines.append(line)

    adj = defaultdict(set)
    for line in lines:
        # adjacency list format
        if ":" in line:
            node, neighbors = line.split(":", 1)
            node = node.strip()
            for nb in neighbors.replace(",", " ").split():
                nb = nb.strip()
                if nb:
                    adj[node].add(nb)
                    adj[nb].add(node)
        else:
            # edge list format
            parts = line.replace(",", " ").split()
            if len(parts) == 2:
                u, v = parts
                adj[u].add(v)
                adj[v].add(u)
            elif len(parts) > 2:
                u, *neighbors = parts
                for v in neighbors:
                    adj[u].add(v)
                    adj[v].add(u)
    return adj

# ------------------------- BFS Helper ----------------------------

def bfs(adj, start):
    """
    Standard BFS that computes distances (eccentricities) from start node.
    Returns:
    - dist: node -> distance from start
    - parent: node -> parent in BFS tree (for path reconstruction)
    """
    dist = {start: 0}
    parent = {start: None}
    q = deque([start])
    print(f"\nBFS from node {start}:")
    while q:
        u = q.popleft()
        print(f" Visiting {u}, distance {dist[u]}")
        for v in sorted(adj[u]):  # consistent order
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
                print(f"  Discovered {v} at distance {dist[v]} (via {u})")
    return dist, parent

def reconstruct_path(parent, start, end):
    """
    Reconstruct path from start to end using parent map.
    """
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path

# ------------------------- Bounding Diameters ----------------------------

def bounding_diameters(adj, verbose=True):
    """
    Iterative bounding diameters algorithm:
    - Initializes LB=0, UB=number of nodes for all nodes
    - Tracks iterations and BFS runs
    - Selection strategy: alternate largest UB / smallest LB
    """
    nodes = list(adj.keys())
    N = len(nodes)
    # Initialize lower and upper bounds for eccentricity
    LB = {v: 0 for v in nodes}           # known minimal eccentricity
    UB = {v: N - 1 for v in nodes}       # maximal possible eccentricity (assume worst case)
    iteration = 0
    diameter_edges = 0
    diameter_path = []

    # Track which nodes have been BFSed
    bfs_done = set()
    # Toggle for selection strategy
    select_largest_UB = True

    while True:
        # ------------------ Node Selection Strategy ------------------
        if select_largest_UB:
            # pick node with largest UB, break ties by highest degree
            candidate_nodes = [v for v in nodes if v not in bfs_done]
            if not candidate_nodes:
                break
            max_ub = max(UB[v] for v in candidate_nodes)
            best_nodes = [v for v in candidate_nodes if UB[v] == max_ub]
            # tie-break: pick node with highest degree
            start_node = max(best_nodes, key=lambda x: len(adj[x]))
        else:
            # pick node with smallest LB, break ties by highest degree
            candidate_nodes = [v for v in nodes if v not in bfs_done]
            if not candidate_nodes:
                break
            min_lb = min(LB[v] for v in candidate_nodes)
            best_nodes = [v for v in candidate_nodes if LB[v] == min_lb]
            start_node = max(best_nodes, key=lambda x: len(adj[x]))
        select_largest_UB = not select_largest_UB  # alternate next time

        # ------------------ Run BFS ------------------
        iteration += 1
        print(f"\n=== Iteration {iteration} ===")
        print(f"Selected node {start_node} for BFS (strategy toggle)")
        dist, parent = bfs(adj, start_node)
        bfs_done.add(start_node)

        # Update LB and UB for all nodes
        for v in nodes:
            # LB[v] >= distance from BFS start
            LB[v] = max(LB[v], dist.get(v, 0))
            # UB[v] <= distance to farthest node from start + eccentricity of start node
            ecc_start = max(dist.values())
            UB[v] = min(UB[v], dist.get(v, 0) + ecc_start)

        # Check if we found a new candidate diameter
        far_node, ecc_start = max(dist.items(), key=lambda kv: kv[1])
        if ecc_start > diameter_edges:
            diameter_edges = ecc_start
            diameter_path = reconstruct_path(parent, start_node, far_node)

        # Verbose reporting of bounds
        if verbose:
            print("\nNode bounds after this BFS:")
            for v in nodes:
                print(f" {v}: LB={LB[v]}, UB={UB[v]}, degree={len(adj[v])}")
            print(f"Current diameter (edges): {diameter_edges}")
            print(f"Current diameter path: {' → '.join(diameter_path)}")

        # Termination condition: all nodes' LB == UB (bounds converged)
        if all(LB[v] == UB[v] for v in nodes):
            if verbose:
                print("\nAll bounds converged. Terminating iterations.")
            break

    print(f"\nTotal BFS iterations: {iteration}")
    print("\n==== Final Diameter ====")
    print(f"Diameter (edges): {diameter_edges}")
    print(f"Diameter path ({len(diameter_path)} nodes): {' → '.join(diameter_path)}")
    return diameter_edges, diameter_path, iteration

# ------------------------- Main ----------------------------

def main():
    adj = parse_input()
    if not adj:
        print("No graph input detected.")
        return

    print("\nAdjacency list:")
    for k in sorted(adj.keys()):
        print(f"  {k}: {sorted(adj[k])}")

    bounding_diameters(adj, verbose=True)

if __name__ == "__main__":
    main()
