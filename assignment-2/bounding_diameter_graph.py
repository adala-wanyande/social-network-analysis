#!/usr/bin/env python3
"""
Bounding Diameters Algorithm (Double Sweep / Iterative Bounding)
---------------------------------------------------------------
Verbose CLI version: finds the diameter of an undirected graph
with far fewer BFS runs than all-pairs BFS.
"""

from collections import deque, defaultdict

def parse_input():
    print("Enter your graph (end with an empty line):")
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
    return adj

def bfs(adj, start):
    dist = {start: 0}
    parent = {start: None}
    q = deque([start])
    print(f"\nStarting BFS from {start}")
    while q:
        u = q.popleft()
        print(f"Visiting {u}, distance {dist[u]}")
        for v in sorted(adj[u]):
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
                print(f"  Discovered {v} at distance {dist[v]} (via {u})")
    return dist, parent

def farthest_node(dist_map):
    node, d = max(dist_map.items(), key=lambda kv: kv[1])
    return node, d

def reconstruct_path(parent, start, end):
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path

def bounding_diameter(adj, verbose=True):
    # Step 1: pick an arbitrary node
    start_node = next(iter(adj))
    dist, parent = bfs(adj, start_node)
    u1, ecc_u1 = farthest_node(dist)
    if verbose:
        print(f"\nDouble-sweep first farthest: {u1} at distance {ecc_u1}")

    # Step 2: BFS from u1
    dist2, parent2 = bfs(adj, u1)
    u2, ecc_u2 = farthest_node(dist2)
    if verbose:
        print(f"Double-sweep second farthest: {u2} at distance {ecc_u2}")

    diameter_edges = ecc_u2
    diameter_path = reconstruct_path(parent2, u1, u2)

    if verbose:
        print("\n==== Double Sweep Result ====")
        print(f"Diameter (edges): {diameter_edges}")
        print(f"Diameter path ({len(diameter_path)} nodes): {' → '.join(diameter_path)}")

    return diameter_edges, diameter_path

def main():
    adj = parse_input()
    if not adj:
        print("No input detected.")
        return

    print("\nAdjacency list:")
    for k in sorted(adj.keys()):
        print(f"  {k}: {sorted(adj[k])}")

    diameter, path = bounding_diameter(adj, verbose=True)

if __name__ == "__main__":
    main()
