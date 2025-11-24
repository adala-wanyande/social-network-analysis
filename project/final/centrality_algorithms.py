# centrality_algorithms.py
import networkx as nx
import time
import heapq
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque
from typing import Dict, Any

# ------------------------------
# Global graph variable (inherited by forked workers)
# ------------------------------
_GLOBAL_G = None  # Will be set before creating the process pool on Unix (fork)

# ------------------------------
# Textbook functions (unchanged)
# ------------------------------
def textbook_unweighted(G: nx.Graph, k: int) -> dict:
    """Computes top-k closeness using the textbook BFS-from-every-node method."""
    print("  -> Running Textbook Unweighted Algorithm...")
    start_time = time.time()
    nodes = list(G.nodes())
    n = len(nodes)
    centrality_scores = []
    for node in nodes:
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


# ------------------------------
# Helper: update all lower bounds based on SSSP from s (Algorithm 5 style)
# ------------------------------
def _update_all_bounds_lb(G: nx.Graph, s: int, distances: Dict[Any, float]) -> Dict[Any, float]:
    """
    Compute new lower bounds L(v) for all nodes v using distances from s.
    This function follows the updateBoundsLB approach: group nodes by level and compute
    L_level values, then assign bounds to nodes in that level with a degree correction.
    """
    n = G.number_of_nodes()
    max_d = 0
    levels = {}
    for node, dist in distances.items():
        d = int(dist)
        levels.setdefault(d, []).append(node)
        if d > max_d:
            max_d = d

    # gamma[i] = number of nodes at distance i
    gamma = [len(levels.get(i, [])) for i in range(max_d + 1)]

    # prefix sums of gamma
    prefix_sum_gamma = [0] * (max_d + 1)
    if max_d >= 0:
        prefix_sum_gamma[0] = gamma[0]
    for i in range(1, max_d + 1):
        prefix_sum_gamma[i] = prefix_sum_gamma[i - 1] + gamma[i]

    # compute L_level values
    L_level = [0] * (max_d + 1)
    if max_d >= 0:
        L_level[0] = sum(i * g for i, g in enumerate(gamma))
    for i in range(1, max_d + 1):
        num_closer = prefix_sum_gamma[i - 1]
        num_farther_or_equal = n - num_closer
        L_level[i] = L_level[i - 1] + num_closer - num_farther_or_equal

    # assign new lower-bound for each node in its level, minus degree correction
    new_farness_sum_bounds = {}
    for i in range(max_d + 1):
        for v in levels.get(i, []):
            new_farness_sum_bounds[v] = L_level[i] - G.degree(v)

    return new_farness_sum_bounds


# ------------------------------
# Worker helper that runs in child processes: compute farness from node v
# Note: we DO NOT pass the full graph object to the worker; instead the worker
# inherits the global _GLOBAL_G via fork when the pool is created.
# ------------------------------
def _compute_farness_in_worker(v, is_weighted: bool):
    """
    Compute sum of distances (farness) from node v using the inherited global graph.
    Returns (v, farness_sum).
    """
    global _GLOBAL_G
    G = _GLOBAL_G
    if G is None:
        raise RuntimeError("Global graph not initialized in worker process.")
    if is_weighted:
        # Dijkstra (NetworkX) in worker
        distances = nx.single_source_dijkstra_path_length(G, v)
    else:
        distances = nx.single_source_shortest_path_length(G, v)
    return v, sum(distances.values())


# ------------------------------
# Core: fast top-k runner (with fork-based process pool on Ubuntu)
# ------------------------------
def _fast_top_k_runner(G: nx.Graph, k: int, is_weighted: bool,
                       log_convergence_data: bool = False,
                       use_parallel: bool = False,
                       max_workers: int = None) -> dict:
    """
    Fast top-k closeness runner. If use_parallel=True and running on Unix, the function
    will create a ProcessPoolExecutor using fork semantics so child processes inherit
    the graph (avoid pickling the large graph).
    """
    algo_type = "Weighted" if is_weighted else "Unweighted"
    mode = "Parallel (fork)" if use_parallel else "Sequential"
    print(f"  -> Running Fast Top-k {algo_type} Algorithm ({mode})...")
    start_time = time.time()

    nodes = list(G.nodes())
    n = len(nodes)

    # initialize lower bounds and top-k list
    lower_bounds_S = {node: 0.0 for node in nodes}
    top_k_list = []  # stores tuples (farness, node), sorted ascending by farness
    sssp_count = 0
    convergence_log = []

    # priority queue over (lower_bound, node)
    pq = [(lower_bounds_S[node], node) for node in nodes]
    heapq.heapify(pq)

    # Prepare parallel executor if requested and on Unix
    if use_parallel:
        # default max_workers to number of physical cores if not provided
        cpu_count = multiprocessing.cpu_count()
        if max_workers is None:
            max_workers = max(1, cpu_count - 1)  # leave one core free
        # Important: set global graph BEFORE creating the pool so fork inherits it
        global _GLOBAL_G
        _GLOBAL_G = G  # inherited by forked children
        # create a multiprocessing context with 'fork' to ensure inheritance
        ctx = multiprocessing.get_context('fork')
        executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
    else:
        executor = None

    try:
        # Main loop (works both in sequential and parallel modes)
        while pq:
            current_S_bound, v = heapq.heappop(pq)

            # stale entry check (we may have pushed an updated bound earlier)
            if current_S_bound < lower_bounds_S[v]:
                continue

            # logging
            if log_convergence_data:
                kth_S_exact = top_k_list[k - 1][0] if len(top_k_list) >= k else float('inf')
                convergence_log.append({'iteration': sssp_count + 1, 'kth_farness': kth_S_exact, 'lower_bound': current_S_bound})

            # stopping condition
            if len(top_k_list) >= k:
                kth_S_exact_check = top_k_list[k - 1][0]
                if current_S_bound > kth_S_exact_check:
                    break

            # Now compute exact farness for v (either in worker pool or locally)
            sssp_count += 1

            if executor is not None:
                # Build a small batch: include v and up to (max_workers-1) other candidates
                batch = [v]
                batch_size = min(max_workers - 1 if max_workers > 1 else 0, len(pq))
                for _ in range(batch_size):
                    # safe pop (if empty pq then break)
                    if not pq:
                        break
                    batch.append(heapq.heappop(pq)[1])

                # Submit tasks for each node in the batch to the pool
                futures = {executor.submit(_compute_farness_in_worker, node, is_weighted): node for node in batch}

                # As completed, update results
                for fut in as_completed(futures):
                    node, exact_farness = fut.result()
                    lower_bounds_S[node] = exact_farness
                    heapq.heappush(top_k_list, (exact_farness, node))
                    top_k_list.sort(key=lambda x: x[0])
                    if len(top_k_list) > k:
                        top_k_list.pop()
            else:
                # Sequential computation for v
                if is_weighted:
                    distances = nx.single_source_dijkstra_path_length(G, v)
                else:
                    distances = nx.single_source_shortest_path_length(G, v)

                exact_farness = sum(distances.values())
                lower_bounds_S[v] = exact_farness
                heapq.heappush(top_k_list, (exact_farness, v))
                top_k_list.sort(key=lambda x: x[0])
                if len(top_k_list) > k:
                    top_k_list.pop()

            # After computing exact farness for those nodes, update lower bounds using Algorithm 5 style
            # We update using the last processed node(s). To keep behavior similar to sequential version,
            # we iterate over recently computed nodes in top_k_list. For simplicity, use the node v (the pivot).
            # You can enhance by updating using each node that just had distances computed.
            try:
                # Prefer to use distances from 'v' if available locally; otherwise recompute local distances for update.
                # In the parallel branch, we don't have distances objects here; instead we can compute distances locally for v
                # solely for the bound update (costly but small relative to many SSSPs).
                if executor is not None:
                    # compute distances locally for v (used only for bound update)
                    if is_weighted:
                        distances_for_update = nx.single_source_dijkstra_path_length(G, v)
                    else:
                        distances_for_update = nx.single_source_shortest_path_length(G, v)
                else:
                    # we already have distances when sequentially computed - but they are not stored;
                    # recompute to keep code simple and consistent.
                    if is_weighted:
                        distances_for_update = nx.single_source_dijkstra_path_length(G, v)
                    else:
                        distances_for_update = nx.single_source_shortest_path_length(G, v)
            except Exception:
                distances_for_update = {}

            # Compute new lower bounds from this pivot's distances
            if distances_for_update:
                new_S_bounds = _update_all_bounds_lb(G, v, distances_for_update)
                for node, s_lb in new_S_bounds.items():
                    if s_lb > lower_bounds_S.get(node, 0.0):
                        lower_bounds_S[node] = s_lb
                        heapq.heappush(pq, (s_lb, node))

        # main loop finished
    finally:
        # clean up executor
        if executor is not None:
            executor.shutdown(wait=True)
        # clear inherited global graph reference (optional)
        _GLOBAL_G = None

    runtime = time.time() - start_time
    print(f"     Done in {runtime:.4f} seconds. ({sssp_count}/{n} SSSPs performed)")

    # Prepare final outputs (convert farness to closeness)
    final_top_k = []
    for s_val, node in top_k_list:
        closeness = (n - 1) / s_val if s_val > 0 else 0.0
        final_top_k.append((closeness, node))
    final_top_k.sort(key=lambda x: x[0], reverse=True)

    pruning_power = 1.0 - (sssp_count / n) if n > 0 else 0.0

    result_dict = {
        'top_k': final_top_k,
        'runtime': runtime,
        'sssp_count': sssp_count,
        'pruning_power': pruning_power
    }
    if log_convergence_data:
        result_dict['convergence_log'] = convergence_log

    return result_dict


# ------------------------------
# Public wrappers (preserve signatures)
# ------------------------------
def topk_closeness_unweighted(G: nx.Graph, k: int, log_convergence_data: bool = False,
                              use_parallel: bool = False, max_workers: int = None) -> dict:
    return _fast_top_k_runner(G, k, is_weighted=False, log_convergence_data=log_convergence_data,
                              use_parallel=use_parallel, max_workers=max_workers)


def topk_closeness_weighted(G: nx.Graph, k: int, log_convergence_data: bool = False,
                            use_parallel: bool = False, max_workers: int = None) -> dict:
    return _fast_top_k_runner(G, k, is_weighted=True, log_convergence_data=log_convergence_data,
                              use_parallel=use_parallel, max_workers=max_workers)


# ------------------------------
# Quick local test (works when run on Ubuntu)
# ------------------------------
if __name__ == '__main__':
    print("--- Running Small Local Test ---")
    G = nx.star_graph(50)
    print("Unweighted top-k (sequential):")
    print(topk_closeness_unweighted(G, 3, use_parallel=False))
    print("Unweighted top-k (parallel):")
    print(topk_closeness_unweighted(G, 3, use_parallel=True, max_workers=4))
