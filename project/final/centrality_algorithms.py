import networkx as nx
import time
import heapq
import pandas as pd  # Needed for creating the log dataframe
import platform
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


def get_executor():
    """Selects appropriate executor depending on the operating system."""
    if platform.system() == "Windows":
        print("  ⚙️ Using ThreadPoolExecutor (Windows-safe parallelization).")
        return ThreadPoolExecutor
    else:
        print("  ⚙️ Using ProcessPoolExecutor (true multi-core parallelization).")
        return ProcessPoolExecutor


def textbook_unweighted(G: nx.Graph, k: int) -> dict:
    """Computes top-k closeness centrality using the textbook algorithm for unweighted graphs."""
    print("  -> Running Textbook Unweighted Algorithm...")
    start_time = time.time()

    nodes = list(G.nodes())
    n = len(nodes)
    centrality_scores = []

    for node in nodes:
        distances = nx.single_source_shortest_path_length(G, node)
        farness = sum(distances.values())
        closeness = (n - 1) / farness if farness > 0 else 0.0
        centrality_scores.append((closeness, node))

    centrality_scores.sort(key=lambda x: x[0], reverse=True)
    runtime = time.time() - start_time
    print(f"     Done in {runtime:.4f} seconds.")

    return {'top_k': centrality_scores[:k], 'runtime': runtime}


def textbook_weighted(G: nx.Graph, k: int) -> dict:
    """Computes top-k closeness centrality using the textbook algorithm for weighted graphs."""
    print("  -> Running Textbook Weighted Algorithm...")
    start_time = time.time()

    nodes = list(G.nodes())
    n = len(nodes)
    centrality_scores = []

    for node in nodes:
        distances = nx.single_source_dijkstra_path_length(G, node)
        farness = sum(distances.values())
        closeness = (n - 1) / farness if farness > 0 else 0.0
        centrality_scores.append((closeness, node))

    centrality_scores.sort(key=lambda x: x[0], reverse=True)
    runtime = time.time() - start_time
    print(f"     Done in {runtime:.4f} seconds.")

    return {'top_k': centrality_scores[:k], 'runtime': runtime}


# --- Worker Function ---
def _compute_farness(args):
    """Worker function for parallel BFS/Dijkstra computation."""
    G, v, is_weighted = args
    if is_weighted:
        distances = nx.single_source_dijkstra_path_length(G, v)
    else:
        distances = nx.single_source_shortest_path_length(G, v)
    return v, sum(distances.values())


def _fast_top_k_runner(G: nx.Graph, k: int, is_weighted: bool, log_convergence_data: bool = False, use_parallel: bool = True) -> dict:
    """Fast top-k closeness centrality algorithm with optional CPU parallelization."""
    algo_type = "Weighted" if is_weighted else "Unweighted"
    exec_mode = "Parallel" if use_parallel else "Sequential"
    print(f"  -> Running Fast Top-k {algo_type} Algorithm ({exec_mode})...")
    start_time = time.time()

    nodes = list(G.nodes())
    n = len(nodes)

    lower_bounds = {node: 0 for node in nodes}
    top_k_list = []
    sssp_count = 0
    convergence_log = []

    pq = [(lower_bounds[node], node) for node in nodes]
    heapq.heapify(pq)
    iteration = 0

    if use_parallel:
        ExecutorClass = get_executor()
        with ExecutorClass(max_workers=4) as executor:  # Adjust cores if needed
            while pq:
                iteration += 1
                current_lower_bound, v = heapq.heappop(pq)
                if current_lower_bound > lower_bounds[v]:
                    continue

                if log_convergence_data:
                    kth_farness = top_k_list[k - 1][0] if len(top_k_list) >= k else float('inf')
                    convergence_log.append({
                        'iteration': iteration,
                        'kth_farness': kth_farness,
                        'lower_bound': current_lower_bound
                    })

                if len(top_k_list) >= k:
                    kth_farness_check = top_k_list[k - 1][0]
                    if current_lower_bound > kth_farness_check:
                        break

                # Batch execution
                batch_nodes = [heapq.heappop(pq)[1] for _ in range(min(4, len(pq)))]
                futures = [executor.submit(_compute_farness, (G, node, is_weighted)) for node in [v] + batch_nodes]

                for f in as_completed(futures):
                    node, exact_farness = f.result()
                    sssp_count += 1
                    lower_bounds[node] = exact_farness
                    heapq.heappush(top_k_list, (exact_farness, node))
                    top_k_list.sort(key=lambda x: x[0])
                    if len(top_k_list) > k:
                        top_k_list.pop()
    else:
        # Sequential Fallback
        while pq:
            iteration += 1
            current_lower_bound, v = heapq.heappop(pq)
            if current_lower_bound > lower_bounds[v]:
                continue

            if log_convergence_data:
                kth_farness = top_k_list[k - 1][0] if len(top_k_list) >= k else float('inf')
                convergence_log.append({
                    'iteration': iteration,
                    'kth_farness': kth_farness,
                    'lower_bound': current_lower_bound
                })

            if len(top_k_list) >= k:
                kth_farness_check = top_k_list[k - 1][0]
                if current_lower_bound > kth_farness_check:
                    break

            sssp_count += 1
            if is_weighted:
                distances = nx.single_source_dijkstra_path_length(G, v)
            else:
                distances = nx.single_source_shortest_path_length(G, v)

            exact_farness = sum(distances.values())
            lower_bounds[v] = exact_farness

            heapq.heappush(top_k_list, (exact_farness, v))
            top_k_list.sort(key=lambda x: x[0])
            if len(top_k_list) > k:
                top_k_list.pop()

    runtime = time.time() - start_time
    print(f"     Done in {runtime:.4f} seconds. ({sssp_count}/{n} SSSPs performed)")

    final_top_k = []
    for farness, node in top_k_list:
        closeness = (n - 1) / farness if farness > 0 else 0.0
        final_top_k.append((closeness, node))
    final_top_k.sort(key=lambda x: x[0], reverse=True)

    pruning_power = 1.0 - (sssp_count / n)

    result_dict = {
        'top_k': final_top_k,
        'runtime': runtime,
        'sssp_count': sssp_count,
        'pruning_power': pruning_power
    }
    if log_convergence_data:
        result_dict['convergence_log'] = convergence_log

    return result_dict


def topk_closeness_unweighted(G: nx.Graph, k: int, log_convergence_data: bool = False, use_parallel: bool = True) -> dict:
    return _fast_top_k_runner(G, k, is_weighted=False, log_convergence_data=log_convergence_data, use_parallel=use_parallel)


def topk_closeness_weighted(G: nx.Graph, k: int, log_convergence_data: bool = False, use_parallel: bool = True) -> dict:
    return _fast_top_k_runner(G, k, is_weighted=True, log_convergence_data=log_convergence_data, use_parallel=use_parallel)


# --- Standalone Test ---
if __name__ == '__main__':
    print("--- Running Tests on a Small Example Graph ---")

    G_unweighted = nx.star_graph(10)
    print("\n--- Unweighted Test on Star Graph ---")
    k = 3

    textbook_res_u = textbook_unweighted(G_unweighted, k)
    fast_res_u = topk_closeness_unweighted(G_unweighted, k, use_parallel=True)

    print("\nTextbook Unweighted Results:", [node for score, node in textbook_res_u['top_k']])
    print("Fast Unweighted Results:", [node for score, node in fast_res_u['top_k']])
    print(f"Pruning Power: {fast_res_u['pruning_power']:.2%}")

    G_weighted = nx.Graph()
    G_weighted.add_edge(0, 1, weight=0.1)
    G_weighted.add_edge(1, 2, weight=0.1)
    G_weighted.add_edge(1, 3, weight=0.1)
    G_weighted.add_edge(3, 4, weight=1.0)
    G_weighted.add_edge(3, 5, weight=1.0)

    print("\n--- Weighted Test on Custom Graph ---")
    k = 2

    textbook_res_w = textbook_weighted(G_weighted, k)
    fast_res_w = topk_closeness_weighted(G_weighted, k, use_parallel=True)

    print("\nTextbook Weighted Results:", [node for score, node in textbook_res_w['top_k']])
    print("Fast Weighted Results:", [node for score, node in fast_res_w['top_k']])
    print(f"Pruning Power: {fast_res_w['pruning_power']:.2%}")
