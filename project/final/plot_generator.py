import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
# Find the latest results CSV file automatically
try:
    RESULTS_DIR = Path('.')
    LATEST_RESULTS_FILE = sorted(RESULTS_DIR.glob("experiment_results_*.csv"))[-1]
except IndexError:
    print("Error: No 'experiment_results_*.csv' file found. Please run 'run_experiments.py' first.")
    exit()

# Directory to save plots
PLOTS_DIR = Path('plots')
PLOTS_DIR.mkdir(exist_ok=True)

# --- Plotting Functions ---

def generate_results_table(df: pd.DataFrame):
    """Prints a formatted summary table of the key results."""
    print("\n--- Key Performance Metrics Summary ---")
    
    # Filter for the fast algorithms to make the table cleaner
    fast_df = df[df['algorithm'].str.startswith('fast')].copy()
    
    # Make the table more readable
    fast_df['runtime'] = fast_df['runtime'].round(2)
    fast_df['pruning_power'] = (fast_df['pruning_power'] * 100).round(2).astype(str) + '%'
    fast_df['improvement_factor'] = fast_df['improvement_factor'].round(1)

    # Select and reorder columns for display
    display_cols = [
        'dataset', 'k', 'runtime', 'pruning_power', 'improvement_factor'
    ]
    print(fast_df[display_cols].to_string(index=False))
    # For LaTeX table, you can use:
    # print(fast_df[display_cols].to_latex(index=False))


def plot_improvement_factors(df: pd.DataFrame, output_path: Path):
    """Plots the improvement factor of fast algorithms over textbook methods."""
    plot_df = df[df['improvement_factor'].notna()].copy()
    plot_df = plot_df.sort_values(by='nodes')

    plt.figure(figsize=(12, 7))
    sns.barplot(data=plot_df, x='dataset', y='improvement_factor', hue='k')
    
    plt.title('Performance Improvement Factor over Textbook Algorithm', fontsize=16)
    plt.xlabel('Dataset (Ordered by Size)', fontsize=12)
    plt.ylabel('Improvement Factor (Speedup)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yscale('log') # Improvement factors can be huge, log scale is better
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved improvement factor plot to {output_path}")
    plt.close()


def plot_runtime_vs_k(df: pd.DataFrame, output_path: Path):
    """Plots how the runtime of fast algorithms scales with k."""
    plot_df = df[df['algorithm'].str.startswith('fast')].copy()

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=plot_df, x='k', y='runtime', hue='dataset', marker='o')
    
    plt.title('Runtime Scalability with Top-k Parameter', fontsize=16)
    plt.xlabel('k (Number of Top Nodes)', fontsize=12)
    plt.ylabel('Runtime (seconds, log scale)', fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved runtime vs k plot to {output_path}")
    plt.close()


def plot_pruning_power(df: pd.DataFrame, output_path: Path):
    """Plots the pruning power for the fast algorithms across datasets."""
    plot_df = df[df['algorithm'].str.startswith('fast') & (df['k'] == 10)].copy()
    plot_df = plot_df.sort_values(by='pruning_power', ascending=False)
    plot_df['pruning_power_pct'] = plot_df['pruning_power'] * 100

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=plot_df, x='dataset', y='pruning_power_pct')
    
    plt.title('Algorithm Pruning Power (for k=10)', fontsize=16)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Pruning Power (% of nodes skipped)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    ax.bar_label(ax.containers[0], fmt='%.1f%%') # Add labels to bars
    plt.ylim(0, 101)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved pruning power plot to {output_path}")
    plt.close()


def plot_bounds_convergence(dataset_name: str, k: int, output_path: Path):
    """Plots the bounds convergence visualization from a dedicated log file."""
    log_file = Path(f'convergence_log_{dataset_name}_k{k}.csv')
    if not log_file.exists():
        print(f"Warning: Convergence log file not found: {log_file}. Skipping plot.")
        return

    df = pd.read_csv(log_file)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['iteration'], df['kth_farness'], label=f'Farness of {k}-th Best Node', color='green')
    plt.plot(df['iteration'], df['lower_bound'], label='Lower Bound of Next Candidate', color='red', linestyle='--')
    
    plt.title(f'Bounds Convergence for {dataset_name.upper()} (k={k})', fontsize=16)
    plt.xlabel('Algorithm Iteration (SSSP Computations)', fontsize=12)
    plt.ylabel('Farness Score', fontsize=12)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved convergence plot for {dataset_name} to {output_path}")
    plt.close()


def main():
    """Main function to load data and generate all plots."""
    print(f"--- Generating Plots from: {LATEST_RESULTS_FILE} ---")
    
    df = pd.read_csv(LATEST_RESULTS_FILE)

    # 1. Generate the main results table in the console
    generate_results_table(df)

    # 2. Generate and save the plots
    plot_improvement_factors(df, PLOTS_DIR / 'improvement_factors.png')
    plot_runtime_vs_k(df, PLOTS_DIR / 'runtime_vs_k.png')
    plot_pruning_power(df, PLOTS_DIR / 'pruning_power.png')
    
    # 3. Generate the special convergence plots
    for dataset in ['roadnet-ca', 'orkut']:
        plot_bounds_convergence(dataset, k=10, output_path=PLOTS_DIR / f'convergence_{dataset}.png')


if __name__ == '__main__':
    main()