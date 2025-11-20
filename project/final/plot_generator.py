import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
try:
    RESULTS_DIR = Path('.')
    LATEST_RESULTS_FILE = sorted(RESULTS_DIR.glob("experiment_results_*.csv"))[-1]
except IndexError:
    print("Error: No 'experiment_results_*.csv' file found. Please run 'run_experiments.py' first.")
    exit()

PLOTS_DIR = Path('plots')
PLOTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# --- Plotting Functions ---

def generate_latex_table(df: pd.DataFrame):
    """Generates a LaTeX-formatted table for the paper, focusing on k=10."""
    print("\n--- LaTeX Table for k=10 Results ---")
    
    table_df = df[(df['k'] == 10) & (df['algorithm'] == 'fast_topk')].copy()
    
    textbook_times = df[(df['k'] == 10) & (df['algorithm'] == 'textbook')][['dataset', 'runtime']]
    textbook_times = textbook_times.rename(columns={'runtime': 'textbook_time'})
    
    table_df = pd.merge(table_df, textbook_times, on='dataset')

    table_df['pruning_power'] = (table_df['pruning_power'] * 100).map('{:.1f}\%'.format)
    table_df['runtime'] = table_df['runtime'].map('{:.2f}'.format)
    table_df['textbook_time'] = table_df['textbook_time'].map('{:.2f}'.format)
    table_df['improvement_factor'] = table_df['improvement_factor'].map('{:.2f}x'.format)
    
    display_cols = ['dataset', 'pruning_power', 'textbook_time', 'runtime', 'improvement_factor']
    table_df = table_df[display_cols].rename(columns={
        'dataset': 'Dataset', 'pruning_power': 'Pruning Power', 
        'textbook_time': 'Textbook Time (s)', 'runtime': 'Fast Time (s)',
        'improvement_factor': 'Speedup'
    })
    
    print(table_df.to_latex(index=False))


def plot_pruning_power(df: pd.DataFrame, output_path: Path):
    """Plots the pruning power for the fast algorithm across datasets, faceted by k."""
    plot_df = df[df['algorithm'] == 'fast_topk'].copy()
    plot_df['pruning_power_pct'] = plot_df['pruning_power'] * 100
    
    g = sns.catplot(data=plot_df, kind="bar", x="dataset", y="pruning_power_pct", col="k", height=5, aspect=1.1)
    g.set_axis_labels("Dataset", "Pruning Power (%)").set_titles("k = {col_name}").set_xticklabels(rotation=45, ha='right')
    g.fig.suptitle('Algorithm Pruning Power by k', y=1.03, fontsize=16)

    for ax in g.axes.flat:
        for c in ax.containers:
            ax.bar_label(c, fmt='%.1f%%', fontsize=9)
        ax.set_ylim(0, 100)

    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved pruning power plot to {output_path}")
    plt.close()


def plot_improvement_factors(df: pd.DataFrame, output_path: Path):
    """Plots the improvement factor (speedup) across datasets, faceted by k."""
    plot_df = df[(df['algorithm'] == 'fast_topk') & df['improvement_factor'].notna()].copy()
    
    g = sns.catplot(data=plot_df, kind="bar", x="dataset", y="improvement_factor", col="k", height=5, aspect=1.1)
    g.set_axis_labels("Dataset", "Improvement Factor (Speedup)").set_titles("k = {col_name}").set_xticklabels(rotation=45, ha='right')
    g.fig.suptitle('Speedup over Textbook Algorithm by k', y=1.03, fontsize=16)

    for ax in g.axes.flat:
        ax.axhline(1.0, ls='--', color='red', lw=2)

    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved improvement factor plot to {output_path}")
    plt.close()


def plot_bounds_convergence(dataset_name: str, k: int, output_path: Path):
    """Plots the bounds convergence visualization from a dedicated log file."""
    log_file = Path(f'convergence_log_{dataset_name}_k{k}.csv')
    if not log_file.exists():
        print(f"Warning: Convergence log file not found: {log_file}. Skipping plot.")
        return

    df = pd.read_csv(log_file)
    plt.figure(figsize=(10, 6))
    plt.plot(df['iteration'], df['kth_farness'], label=f'Farness of {k}-th Best Node', color='green', lw=2)
    plt.plot(df['iteration'], df['lower_bound'], label='Lower Bound of Next Candidate', color='red', linestyle='--', lw=2)
    
    intersection_df = df[df['lower_bound'] > df['kth_farness']]
    if not intersection_df.empty:
        stop_iteration = intersection_df.iloc[0]['iteration']
        plt.axvline(stop_iteration, color='black', linestyle=':', label=f'Pruning Stop Point (Iter {stop_iteration})')
    else:
        plt.text(0.95, 0.05, 'No early termination', 
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=plt.gca().transAxes, color='gray', fontsize=12)
    
    plt.title(f'Bounds Convergence for {dataset_name.upper()} (k={k})', fontsize=16)
    plt.xlabel('Algorithm Iteration (SSSP Computations)', fontsize=12)
    plt.ylabel('Farness Score (Sum of Distances)', fontsize=12)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved convergence plot for {dataset_name} to {output_path}")
    plt.close()


def main():
    """Main function to load data and generate all plots."""
    print(f"--- Generating Plots from: {LATEST_RESULTS_FILE} ---")
    
    df = pd.read_csv(LATEST_RESULTS_FILE)

    # --- FINAL FIX: Filter out the invalid k > n experimental run ---
    # This removes the row where dataset is 'lesmis' AND k is 100.
    original_rows = len(df)
    df_filtered = df[~((df['dataset'] == 'lesmis') & (df['k'] == 100))].copy()
    if len(df_filtered) < original_rows:
        print("\nNOTE: Excluded 'lesmis, k=100' run from plots as it is an invalid case (k > n).\n")
    
    # Generate the table for k=10 (unaffected by the filter)
    generate_latex_table(df)

    # Generate plots USING THE FILTERED DATAFRAME
    plot_pruning_power(df_filtered, PLOTS_DIR / 'pruning_power_by_k.png')
    plot_improvement_factors(df_filtered, PLOTS_DIR / 'improvement_factors_by_k.png')
    
    # Generate the special convergence plots for the specified datasets
    for dataset in ['wiki-vote', 'facebook-forum']:
        plot_bounds_convergence(dataset, k=10, output_path=PLOTS_DIR / f'convergence_{dataset}.png')


if __name__ == '__main__':
    main()