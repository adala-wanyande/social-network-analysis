#!/usr/bin/env python3
"""
Exact Bounding Diameters Algorithm (CLI)
Implements the algorithm from Boldi & Vigna (2013)
Input: Undirected graph via adjacency list or edge list
Output: Exact diameter of the graph with detailed step-by-step explanation
"""

from collections import defaultdict, deque
import math
import pandas as pd

# ------------------------- Graph Input ------------------------- #
def parse_input():
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
        if ":" in line:
            node, neighbors = line.split(":", 1)
            node = node.strip()
            for nb in neighbors.replace(",", " ").split():
                nb = nb.strip()
                if nb:
                    adj[node].add(nb)
                    adj[nb].add(node)
        else:
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
    return dict(adj)

# ------------------------- BFS & Eccentricity ------------------------- #
def bfs_distances(graph, start):
    dist = {node: math.inf for node in graph}
    dist[start] = 0
    queue = deque([start])
    while queue:
        v = queue.popleft()
        for nbr in graph[v]:
            if dist[nbr] == math.inf:
                dist[nbr] = dist[v] + 1
                queue.append(nbr)
    return dist

def eccentricity(graph, v):
    dist = bfs_distances(graph, v)
    ecc = max(d for d in dist.values() if d < math.inf)
    return ecc, dist

# ------------------------- Node Selection ------------------------- #
def select_from(W, εL, εU, graph, toggle):
    """
    Selection strategy:
    - Alternate between the node with the largest εU and smallest εL.
    - Break ties using degree.
    """
    candidates = list(W)
    if toggle:
        max_ub = max(εU[v] for v in candidates)
        max_nodes = [v for v in candidates if εU[v] == max_ub]
        selected = max(max_nodes, key=lambda x: len(graph[x]))
    else:
        min_lb = min(εL[v] for v in candidates)
        min_nodes = [v for v in candidates if εL[v] == min_lb]
        selected = max(min_nodes, key=lambda x: len(graph[x]))
    return selected

# ------------------------- Bounding Diameters ------------------------- #
def bounding_diameter(graph, verbose=True):
    V = list(graph.keys())
    W = set(V)
    ΔL, ΔU = -math.inf, math.inf
    εL = {v: -math.inf for v in V}
    εU = {v: math.inf for v in V}
    iteration = 0
    toggle = True  # alternates selection strategy

    # --- MODIFICATION START ---
    # Pre-calculate degrees and initialize known eccentricities
    degrees = {v: len(graph[v]) for v in V}
    known_eccentricities = {v: '?' for v in V}
    # --- MODIFICATION END ---

    print("\n===== Bounding Diameters Algorithm (Exact) =====\n")
    print(f"Initial: ΔL = {ΔL}, ΔU = {ΔU}, W = {list(W)}\n")

    while W:
        iteration += 1
        v = select_from(W, εL, εU, graph, toggle)
        toggle = not toggle
        εv, dist_v = eccentricity(graph, v)
        
        # --- MODIFICATION START ---
        # Store the newly calculated eccentricity
        known_eccentricities[v] = εv
        # --- MODIFICATION END ---
        
        if verbose:
            print(f"\n--- Iteration {iteration} ---")
            print(f"Selected node: {v}")
            print(f"Eccentricity ε({v}) = {εv}")
            print(f"Distances from {v}: {dist_v}")

        ΔL = max(ΔL, εv)
        ΔU = min(ΔU, 2 * εv)
        if verbose:
            print(f"Updated ΔL = {ΔL}, ΔU = {ΔU}")

        remove_nodes = []
        for w in list(W):
            d_vw = dist_v[w]
            εL[w] = max(εL[w], max(εv - d_vw, d_vw))
            εU[w] = min(εU[w], εv + d_vw)

            # Removal condition
            if (εU[w] <= ΔL and εL[w] >= ΔU / 2) or (εL[w] == εU[w]):
                remove_nodes.append(w)

        for w in remove_nodes:
            W.remove(w)
        
        # --- MODIFICATION START ---
        # Update the DataFrame creation to include the new columns
        table = pd.DataFrame({
            "Node": V,
            "Degree": [degrees[v] for v in V],
            "Eccentricity": [known_eccentricities[v] for v in V],
            "εL (lower bound)": [εL[v] for v in V],
            "εU (upper bound)": [εU[v] for v in V],
            "In W?": ["Yes" if v in W else "No" for v in V],
        })
        # --- MODIFICATION END ---
        
        print("\nCurrent bounds:")
        print(table.to_string(index=False))
        print(f"\nRemaining W = {list(W)}")
        print(f"Current diameter bounds: {ΔL} ≤ Δ ≤ {ΔU}")

        # Check for convergence
        if ΔL == ΔU:
            print("\n>>> Exact diameter found: Δ =", ΔL)
            return ΔL

    print("\n===== Final Result =====")
    print(f"Exact diameter: Δ = {ΔL}")
    return ΔL

# ------------------------- Main ------------------------- #
if __name__ == "__main__":
    graph = parse_input()
    if not graph:
        print("No input detected.")
    else:
        print("\nAdjacency list:")
        for k in sorted(graph.keys()):
            print(f"  {k}: {sorted(graph[k])}")
        bounding_diameter(graph, verbose=True)