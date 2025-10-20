#!/usr/bin/env python3
"""
Graph Diameter Finder (Verbose CLI Version)
-------------------------------------------
Accepts an undirected graph via edge list or adjacency list.
Computes the diameter using BFS from every node.

Features:
- Detects format (edge list or adjacency list)
- Prints BFS progress from each start node
- Prints the longest shortest path (diameter) and its endpoints
"""

from collections import deque, defaultdict

def parse_input():
    print("Enter your graph (end with an empty line):")
    lines = []
    while True:
        line = input().strip()
        if not line:
            break
        lines.append(line)

    # determine if adjacency or edge list
    edges = []
    adj = defaultdict(set)

    for line in lines:
        if ":" in line:  # adjacency list
            node, neighbors = line.split(":", 1)
            node = node.strip()
            for nb in neighbors.replace(",", " ").split():
                nb = nb.strip()
                if nb:
                    adj[node].add(nb)
                    adj[nb].add(node)
        else:  # edge list
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
    """Run BFS from start node, returning distance map and parent map."""
    dist = {start: 0}
    parent = {start: None}
    q = deque([start])

    print(f"\nStarting BFS from node {start}")
    while q:
        u = q.popleft()
        print(f"Visiting {u}, distance {dist[u]}")
        for v in sorted(adj[u]):  # sorted for consistent order
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
                print(f"  Discovered {v} at distance {dist[v]} (via {u})")

    return dist, parent


def reconstruct_path(parent, a, b):
    """Reconstruct path from a to b given parent map."""
    if b not in parent:
        return []
    path = [b]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def compute_diameter(adj):
    """Compute diameter by running BFS from every node."""
    global_max = -1
    best_pair = (None, None)
    best_path = []

    for node in adj.keys():
        dist, parent = bfs(adj, node)
        far_node, max_d = max(dist.items(), key=lambda kv: kv[1])
        print(f"  -> Farthest from {node} is {far_node} at distance {max_d}")
        if max_d > global_max:
            global_max = max_d
            best_pair = (node, far_node)
            best_path = reconstruct_path(parent, node, far_node)

    return global_max, best_pair, best_path


def main():
    adj = parse_input()

    print("\nAdjacency list representation:")
    for k, v in adj.items():
        print(f"  {k}: {sorted(v)}")

    diameter, (u, v), path = compute_diameter(adj)

    print("\n==== Result ====")
    print(f"Diameter length: {diameter}")
    print(f"Between nodes: {u} and {v}")
    print("Path:", " → ".join(path))


if __name__ == "__main__":
    main()
