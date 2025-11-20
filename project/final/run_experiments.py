import pandas as pd
from datetime import datetime
import data_loader
import centrality_algorithms

# --- DEFINITIVE FINAL 7-DATASET CONFIGURATION ---
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
CONVERGENCE_LOG_CONFIG = {('wiki-vote', 10), ('facebook-forum', 10)}
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV_FILE = f'experiment_results_{TIMESTAMP}.csv'

# These tuples MUST EXACTLY MATCH the keys above
CONVERGENCE_LOG_CONFIG = {
    ('wiki-vote', 10), 
    ('facebook-forum', 10)
}

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV_FILE = f'experiment_results_{TIMESTAMP}.csv'

def main():
    all_results = []
    print("="*50 + "\n      STARTING TOP-K CENTRALITY EXPERIMENTS\n" + "="*50)

    for name, config in DATASETS_CONFIG.items():
        try:
            G = data_loader.load_and_preprocess_graph(name)
            n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
        except Exception as e:
            print(f"!!!!!! ERROR loading '{name}': {e}. Skipping. !!!!!!")
            continue

        for k in K_VALUES:
            print(f"\n--- DATASET: {name.upper()}, K: {k} ---")
            
            if config['run_textbook']:
                algo_func = centrality_algorithms.textbook_weighted if config['is_weighted'] else centrality_algorithms.textbook_unweighted
                res = algo_func(G, k)
                all_results.append({
                    'dataset': name, 'nodes': n_nodes, 'edges': n_edges, 'k': k, 
                    'algorithm': 'textbook', 'runtime': res['runtime'],
                    'sssp_count': n_nodes, 'pruning_power': 0.0
                })

            log_this_run = (name, k) in CONVERGENCE_LOG_CONFIG
            algo_func = centrality_algorithms.topk_closeness_weighted if config['is_weighted'] else centrality_algorithms.topk_closeness_unweighted
            res = algo_func(G, k, log_convergence_data=log_this_run)

            if 'convergence_log' in res:
                log_df = pd.DataFrame.from_records(res['convergence_log'])
                log_df.to_csv(f'convergence_log_{name}_k{k}.csv', index=False)
                print(f"  -> Saved convergence data to 'convergence_log_{name}_k{k}.csv'")

            all_results.append({
                'dataset': name, 'nodes': n_nodes, 'edges': n_edges, 'k': k, 
                'algorithm': 'fast_topk', 'runtime': res['runtime'], 
                'sssp_count': res['sssp_count'], 'pruning_power': res['pruning_power']
            })

    if not all_results:
        print("\nNo results generated. Exiting.")
        return

    print("\n" + "="*50 + "\n         ALL EXPERIMENTS COMPLETED\n" + "="*50)
    results_df = pd.DataFrame(all_results)
    
    def calculate_improvement(df_group):
        try:
            textbook_time = df_group.loc[df_group['algorithm'] == 'textbook', 'runtime'].iloc[0]
            fast_time = df_group.loc[df_group['algorithm'] == 'fast_topk', 'runtime'].iloc[0]
            df_group['improvement_factor'] = textbook_time / fast_time
        except (IndexError, ZeroDivisionError):
            df_group['improvement_factor'] = None
        return df_group

    results_df = results_df.groupby(['dataset', 'k']).apply(calculate_improvement).reset_index(drop=True)
    
    print("\n--- Sample of Results ---")
    print(results_df.head())
    
    results_df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"\nSuccessfully saved all results to '{OUTPUT_CSV_FILE}'")

if __name__ == '__main__':
    main()