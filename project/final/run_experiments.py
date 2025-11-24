import pandas as pd
from datetime import datetime
import data_loader
import centrality_algorithms

# --- DEFINITIVE FINAL DATASET CONFIG ---
DATASETS_CONFIG = {
    'wiki-vote':        {'is_weighted': False, 'run_textbook': True},
    'facebook':         {'is_weighted': False, 'run_textbook': True},
    'email-eu':         {'is_weighted': False, 'run_textbook': True},
    'ca-grqc':          {'is_weighted': False, 'run_textbook': True},
    'lesmis':           {'is_weighted': True,  'run_textbook': True},
    'norwegian-boards': {'is_weighted': True,  'run_textbook': True},
    'facebook-forum':   {'is_weighted': True,  'run_textbook': True}
}

K_VALUES = [1, 10, 100]

# We only use convergence logs for a small subset
CONVERGENCE_LOG_CONFIG = {
    ('wiki-vote', 10),
    ('facebook-forum', 10)
}

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV_FILE = f"experiment_results_{TIMESTAMP}.csv"


def main():
    all_results = []
    print("=" * 50)
    print("      STARTING TOP-K CENTRALITY EXPERIMENTS")
    print("=" * 50)

    for name, config in DATASETS_CONFIG.items():

        print(f"\n===== DATASET: {name.upper()} =====")

        try:
            G = data_loader.load_and_preprocess_graph(name)
            n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
        except Exception as e:
            print(f"!!!!!! ERROR loading '{name}': {e}. Skipping. !!!!!!")
            continue

        for k in K_VALUES:
            print(f"\n--- Running k = {k} ---")

            # ------------------------
            # 1. Textbook (Sequential BFS/Dijkstra)
            # ------------------------
            if config['run_textbook']:
                algo_func = (
                    centrality_algorithms.textbook_weighted
                    if config['is_weighted']
                    else centrality_algorithms.textbook_unweighted
                )
                res = algo_func(G, k)

                all_results.append({
                    'dataset': name,
                    'nodes': n_nodes,
                    'edges': n_edges,
                    'k': k,
                    'algorithm': 'textbook',
                    'runtime': res['runtime'],
                    'sssp_count': n_nodes,
                    'pruning_power': 0.0,
                    'parallel_used': False
                })

            # ------------------------
            # 2. Fast Top-K SEQUENTIAL
            # ------------------------
            log_this_run = (name, k) in CONVERGENCE_LOG_CONFIG
            fast_func = (
                centrality_algorithms.topk_closeness_weighted
                if config['is_weighted']
                else centrality_algorithms.topk_closeness_unweighted
            )

            res_seq = fast_func(G, k, log_convergence_data=log_this_run, use_parallel=False)

            all_results.append({
                'dataset': name,
                'nodes': n_nodes,
                'edges': n_edges,
                'k': k,
                'algorithm': 'fast_topk_sequential',
                'runtime': res_seq['runtime'],
                'sssp_count': res_seq['sssp_count'],
                'pruning_power': res_seq['pruning_power'],
                'parallel_used': False
            })

            # ------------------------
            # 3. Fast Top-K PARALLEL
            # ------------------------
            res_par = fast_func(G, k, log_convergence_data=False, use_parallel=True, max_workers=8)

            all_results.append({
                'dataset': name,
                'nodes': n_nodes,
                'edges': n_edges,
                'k': k,
                'algorithm': 'fast_topk_parallel',
                'runtime': res_par['runtime'],
                'sssp_count': res_par['sssp_count'],
                'pruning_power': res_par['pruning_power'],
                'parallel_used': True
            })

    # -------------------------------
    # Save Final CSV
    # -------------------------------
    if not all_results:
        print("\nNo results generated. Exiting.")
        return

    results_df = pd.DataFrame(all_results)

    # Compute improvement factor relative to sequential fast_topk
    def calculate_improvement(df_group):
        try:
            seq = df_group[df_group['algorithm'] == 'fast_topk_sequential']['runtime'].iloc[0]
            par = df_group[df_group['algorithm'] == 'fast_topk_parallel']['runtime'].iloc[0]
            df_group['speedup_parallel_vs_seq'] = seq / par
        except:
            df_group['speedup_parallel_vs_seq'] = None
        return df_group

    results_df = results_df.groupby(['dataset', 'k']).apply(calculate_improvement).reset_index(drop=True)

    print("\n--- Sample Results ---")
    print(results_df.head())

    results_df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"\nSaved full results → {OUTPUT_CSV_FILE}")


if __name__ == "__main__":
    main()
