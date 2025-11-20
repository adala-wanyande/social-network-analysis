import networkx as nx
import time
import heapq
import pandas as pd

# --- Textbook functions remain the same ---
def textbook_unweighted(G: nx.Graph, k: int) -> dict:
    """Computes top-k closeness using the textbook BFS-from-every-node method."""
    print("  -> Running Textbook Unweighted Algorithm...")
    start_time = time.time()
    nodes = list(G.nodes())
    n = len(nodes)
    centrality_scores = []
    for node in nodes:
        # We assume the graph is the LCC, so all nodes are reachable
        distances = nx.single_source_shortest_path_length(G, node)
        farness_sum = sum(distances.values())
        closeness = (n - 1) / farness_sum if farness_sum > 0 else 0.0
        centrality_scores.append((closeness, node))
    centrality_scores.sort(key=lambda x: x[0], reverse=True)
    runtime = time.time() - start_time
    print(f"     Done in {runtime:.4f} seconds.")
    return {'top_k': centrality_scores[:k], 'runtime': runtime}

def textbook_weighted(G: nx.Graph, k: int) -> dict:
    """Computes top-k closeness using the textbook Dijkstra-from-every-node method."""
    print("  -> Running Textbook Weighted Algorithm...")
    start_time = time.time()
    nodes = list(G.nodes())
    n = len(nodes)
    centrality_scores = []
    for node in nodes:
        distances = nx.single_source_dijkstra_path_length(G, node)
        farness_sum = sum(distances.values())
        closeness = (n - 1) / farness_sum if farness_sum > 0 else 0.0
        centrality_scores.append((closeness, node))
    centrality_scores.sort(key=lambda x: x[0], reverse=True)
    runtime = time.time() - start_time
    print(f"     Done in {runtime:.4f} seconds.")
    return {'top_k': centrality_scores[:k], 'runtime': runtime}


def _update_all_bounds_lb(G: nx.Graph, s: int, distances: dict) -> dict:
    """
    Implements Algorithm 5 from the paper to efficiently compute new lower bounds
    on the farness sum for all other nodes after an SSSP from source `s`.
    """
    n = G.number_of_nodes()
    max_d = 0
    # Group nodes by their distance from s
    levels = {}
    for node, dist in distances.items():
        dist = int(dist) # Distances can be floats in weighted graphs
        if dist not in levels:
            levels[dist] = []
        levels[dist].append(node)
        if dist > max_d:
            max_d = dist

    gamma = [len(levels.get(i, [])) for i in range(max_d + 1)]
    
    prefix_sum_gamma = [0] * (max_d + 1)
    prefix_sum_gamma[0] = gamma[0]
    for i in range(1, max_d + 1):
        prefix_sum_gamma[i] = prefix_sum_gamma[i-1] + gamma[i]

    L_level = [0] * (max_d + 1)
    L_level[0] = sum(i * g for i, g in enumerate(gamma))
    
    for i in range(1, max_d + 1):
        num_closer = prefix_sum_gamma[i-1]
        num_farther_or_equal = n - num_closer
        L_level[i] = L_level[i-1] + num_closer - num_farther_or_equal

    new_farness_sum_bounds = {}
    for i in range(max_d + 1):
        for v in levels.get(i, []):
            # The paper's simpler bound for undirected graphs (Lemma 7.2 + deg)
            # This is a lower bound on the sum of distances, S(v)
            new_farness_sum_bounds[v] = L_level[i] - G.degree(v)
            
    return new_farness_sum_bounds

def _fast_top_k_runner(G: nx.Graph, k: int, is_weighted: bool, log_convergence_data: bool = False) -> dict:
    """
    Fully functional implementation using farness (lower is better) and the
    updateBoundsLB strategy from the paper.
    """
    algo_type = "Weighted" if is_weighted else "Unweighted"
    print(f"  -> Running Full Top-k {algo_type} Algorithm (Sequential)...")
    start_time = time.time()

    nodes = list(G.nodes())
    n = len(nodes)
    
    # Paper uses 'farness', f(v) = S(v) * (n-1) / (r(v)-1)^2. Since r(v)=n for LCC,
    # f(v) is proportional to S(v). We can work directly with S(v), the sum of distances.
    lower_bounds_S = {node: 0.0 for node in nodes} # Lower bound on S(v)
    top_k_list = [] # Stores (exact_S, node)
    sssp_count = 0
    
    pq = [(lower_bounds_S[node], node) for node in nodes]
    heapq.heapify(pq)
    
    while pq:
        current_S_bound, v = heapq.heappop(pq)

        # Stale entry check: if we found a better bound for `v` since this
        # entry was pushed, ignore this old entry.
        if current_S_bound < lower_bounds_S[v]:
            continue

        # --- Stopping Condition ---
        if len(top_k_list) >= k:
            kth_S_exact = top_k_list[k-1][0]
            if current_S_bound > kth_S_exact:
                break # Pruning happens here!

        # --- SSSP Computation ---
        sssp_count += 1
        if is_weighted:
            distances = nx.single_source_dijkstra_path_length(G, v)
        else:
            distances = nx.single_source_shortest_path_length(G, v)
        
        exact_S = sum(distances.values())

        # Update the exact score of the processed node `v`
        lower_bounds_S[v] = exact_S
        heapq.heappush(top_k_list, (exact_S, v))
        top_k_list.sort(key=lambda x: x[0])
        if len(top_k_list) > k:
            top_k_list.pop()
        
        # --- CRITICAL STEP: Update bounds for ALL other nodes ---
        new_S_bounds = _update_all_bounds_lb(G, v, distances)
        
        for node, s_lb in new_S_bounds.items():
            if s_lb > lower_bounds_S[node]:
                 lower_bounds_S[node] = s_lb
                 heapq.heappush(pq, (s_lb, node))

    runtime = time.time() - start_time
    print(f"     Done in {runtime:.4f} seconds. ({sssp_count}/{n} SSSPs performed)")
    
    # Convert final farness sums (S) to closeness scores
    final_top_k = []
    for s_val, node in top_k_list:
        closeness = (n - 1) / s_val if s_val > 0 else 0.0
        final_top_k.append((closeness, node))
    final_top_k.sort(key=lambda x: x[0], reverse=True)

    pruning_power = 1.0 - (sssp_count / n)
    
    result_dict = {'top_k': final_top_k, 'runtime': runtime, 'sssp_count': sssp_count, 'pruning_power': pruning_power}
    # Note: Convergence logging would need to be added back if desired
    return result_dict

def topk_closeness_unweighted(G: nx.Graph, k: int, **kwargs) -> dict:
    return _fast_top_k_runner(G, k, is_weighted=False, **kwargs)

def topk_closeness_weighted(G: nx.Graph, k: int, **kwargs) -> dict:
    return _fast_top_k_runner(G, k, is_weighted=True, **kwargs)