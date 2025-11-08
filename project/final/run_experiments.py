import pandas as pd
from datetime import datetime
import os

# Import our custom modules
import data_loader
import centrality_algorithms

# --- Configuration Section ---

# Define the datasets to run experiments on and their properties
# The key is the short name used in data_loader
DATASETS_CONFIG = {
    'livejournal':   {'is_weighted': False, 'run_textbook': False}, # Textbook is too slow
    'orkut':         {'is_weighted': False, 'run_textbook': False}, # Textbook is too slow
    'roadnet-ca':    {'is_weighted': True,  'run_textbook': True},
    'google':        {'is_weighted': False, 'run_textbook': True},
    'dblp':          {'is_weighted': False, 'run_textbook': True},
    'stackoverflow': {'is_weighted': True,  'run_textbook': False}  # Textbook is too slow
}

# Define the k-values to test for top-k
K_VALUES = [1, 10, 100]

# --- NEW CONFIGURATION FOR CONVERGENCE PLOTS ---
# Specify which (dataset, k) combinations should generate detailed convergence logs.
# This avoids creating huge log files for every single run.
CONVERGENCE_LOG_CONFIG = {
    ('roadnet-ca', 10),
    ('orkut', 10)
}
# --- END NEW CONFIGURATION ---

# Define the output file for the results
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV_FILE = f'experiment_results_{TIMESTAMP}.csv'

# --- Main Experiment Runner ---

def main():
    """Main function to orchestrate the experimental pipeline."""
    
    all_results = []

    print("==================================================")
    print("      STARTING TOP-K CENTRALITY EXPERIMENTS      ")
    print("==================================================")

    for name, config in DATASETS_CONFIG.items():
        print(f"\n{'='*20} DATASET: {name.upper()} {'='*20}")
        
        try:
            G = data_loader.load_and_preprocess_graph(name)
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
        except Exception as e:
            print(f"!!!!!! ERROR: Could not load or process dataset '{name}'. Skipping. !!!!!!")
            print(f"Error details: {e}")
            continue

        for k in K_VALUES:
            print(f"\n---------- Running for k = {k} ----------")
            
            # --- Textbook Algorithm Run (No changes here) ---
            if config['run_textbook']:
                if config['is_weighted']:
                    result = centrality_algorithms.textbook_weighted(G, k)
                    algo_name = 'textbook_weighted'
                else:
                    result = centrality_algorithms.textbook_unweighted(G, k)
                    algo_name = 'textbook_unweighted'
                
                res_dict = {
                    'dataset': name, 'nodes': n_nodes, 'edges': n_edges, 'k': k,
                    'algorithm': algo_name, 'runtime': result['runtime'], 'sssp_count': n_nodes,
                    'pruning_power': 0.0, 'top_k_nodes': [node for score, node in result['top_k']]
                }
                all_results.append(res_dict)
            else:
                print(f"  -> Skipping Textbook algorithm for '{name}' as configured (likely too slow).")

            # --- Fast Top-k Algorithm Run ---
            
            # --- MODIFIED SECTION ---
            # Check if this specific (dataset, k) run should generate a convergence log
            log_this_run = (name, k) in CONVERGENCE_LOG_CONFIG
            
            if config['is_weighted']:
                result = centrality_algorithms.topk_closeness_weighted(
                    G, k, log_convergence_data=log_this_run
                )
                algo_name = 'fast_topk_weighted'
            else:
                result = centrality_algorithms.topk_closeness_unweighted(
                    G, k, log_convergence_data=log_this_run
                )
                algo_name = 'fast_topk_unweighted'

            # Save the convergence log to a separate file if it was generated
            if 'convergence_log' in result:
                log_df = pd.DataFrame.from_records(result['convergence_log'])
                log_filename = f'convergence_log_{name}_k{k}.csv'
                log_df.to_csv(log_filename, index=False)
                print(f"  -> Saved convergence data to '{log_filename}'")
            # --- END MODIFIED SECTION ---

            res_dict = {
                'dataset': name, 'nodes': n_nodes, 'edges': n_edges, 'k': k,
                'algorithm': algo_name, 'runtime': result['runtime'],
                'sssp_count': result['sssp_count'], 'pruning_power': result['pruning_power'],
                'top_k_nodes': [node for score, node in result['top_k']]
            }
            all_results.append(res_dict)

    # --- Save Results to CSV (No changes here) ---
    if not all_results:
        print("\nNo results were generated. Exiting.")
        return

    print("\n==================================================")
    print("         ALL EXPERIMENTS COMPLETED               ")
    print("==================================================")

    results_df = pd.DataFrame.from_records(all_results)
    
    def calculate_improvement(df_group):
        try:
            textbook_time = df_group[df_group['algorithm'].str.startswith('textbook')]['runtime'].iloc[0]
            fast_time = df_group[df_group['algorithm'].str.startswith('fast')]['runtime'].iloc[0]
            df_group['improvement_factor'] = textbook_time / fast_time
        except (IndexError, ZeroDivisionError):
            df_group['improvement_factor'] = None
        return df_group

    results_df = results_df.groupby(['dataset', 'k']).apply(calculate_improvement).reset_index(drop=True)

    print("\n--- Sample of Results ---")
    print(results_df.head())
    
    try:
        results_df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"\nSuccessfully saved all results to '{OUTPUT_CSV_FILE}'")
    except Exception as e:
        print(f"\n!!!!!! ERROR: Could not save results to file. !!!!!!")
        print(f"Error details: {e}")


if __name__ == '__main__':
    main()