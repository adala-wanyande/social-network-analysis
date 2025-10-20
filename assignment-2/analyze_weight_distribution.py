import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os

def analyze_weights(dataset_name):
    """
    Analyzes and plots the link weight distribution from an edgelist file.
    """
    edgelist_path = f'{dataset_name}-edgelist.csv'
    if not os.path.exists(edgelist_path):
        print(f"Error: {edgelist_path} not found. Please run the extraction script first.")
        return

    print(f"\n--- Analyzing Weight Distribution for {dataset_name} ---")

    # 1. Load the data and extract weights
    df = pd.read_csv(edgelist_path)
    weights = df['Weight']

    # 2. Count the frequency of each weight
    weight_counts = Counter(weights)
    
    # Prepare data for plotting
    # Sort by weight to ensure the plot is drawn correctly
    sorted_weights = sorted(weight_counts.keys())
    counts = [weight_counts[w] for w in sorted_weights]
    
    print("Top 5 most frequent weights:")
    for w in sorted_weights[:5]:
        print(f"  - Weight {w}: Occurs {weight_counts[w]:,} times")

    # 3. Create the plot
    plt.figure(figsize=(10, 7))
    
    # A log-log plot is the standard for this type of distribution
    plt.loglog(sorted_weights, counts, 'bo', markersize=5, alpha=0.7)
    
    plt.title(f'Link Weight Distribution for {dataset_name.replace("-", " ").title()}')
    plt.xlabel('Link Weight (w) - Number of Mentions')
    plt.ylabel('Frequency P(w) - Number of Links with that Weight')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    
    # Save the figure for the report
    output_filename = f'{dataset_name}_weight_distribution.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Saved weight distribution plot to {output_filename}")


if __name__ == '__main__':
    # Use the name of the dataset you have been analyzing
    analyze_weights('twitter-small')