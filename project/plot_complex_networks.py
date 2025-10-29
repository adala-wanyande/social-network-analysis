import matplotlib.pyplot as plt
import numpy as np

# --- 1. Configure Matplotlib for LaTeX-style output ---
# This is the most important step for achieving a professional look.
plt.rcParams.update({
    "text.usetex": False,  # Use LaTeX to render all text
    "font.family": "serif",
     "font.serif": ["Times New Roman"],
    # Tell LaTeX to use the 'times' package for math and text
    "text.latex.preamble": r"\usepackage{mathptmx}",
    # --- END MODIFIED SECTION ---
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": (8, 3.5),
})

# --- 2. Data Preparation ---
# Re-structured for easier plotting by k-value
algorithms = ["Olh", "Ocl", "DegCut", "DegBound", "NBCut", "NBBound"]
graph_types = ['Directed', 'Undirected', 'Both']

# Data for k=1
means_k1 = {
    'Directed':   [21.24, 1.71, 104.20, 3.61, 123.46, 17.95],
    'Undirected': [11.11, 2.71, 171.77, 5.83, 257.81, 56.16],
    'Both':       [15.64, 2.12, 131.94, 4.53, 174.79, 30.76],
}
stds_k1 = {
    'Directed':   [5.68, 1.54, 6.36, 3.50, 7.94, 10.73],
    'Undirected': [2.91, 1.50, 6.17, 8.09, 8.54, 9.39],
    'Both':       [4.46, 1.61, 6.38, 5.57, 8.49, 10.81],
}

# Data for k=10
means_k10 = {
    'Directed':   [21.06, 1.31, 56.47, 2.87, 58.81, 9.28],
    'Undirected': [11.11, 1.47, 60.25, 2.04, 62.93, 10.95],
    'Both':       [15.57, 1.38, 58.22, 2.44, 60.72, 10.03],
}
stds_k10 = {
    'Directed':   [5.65, 1.31, 5.10, 3.45, 5.65, 6.29],
    'Undirected': [2.90, 1.11, 4.88, 1.45, 5.01, 3.76],
    'Both':       [4.44, 1.24, 5.00, 2.59, 5.34, 5.05],
}

# Data for k=100
means_k100 = {
    'Directed':   [20.94, 1.30, 22.88, 2.56, 23.93, 4.87],
    'Undirected': [11.11, 1.46, 15.13, 1.67, 15.98, 4.18],
    'Both':       [15.52, 1.37, 18.82, 2.09, 19.78, 4.53],
}
stds_k100 = {
    'Directed':   [5.63, 1.31, 4.70, 3.44, 4.83, 4.01],
    'Undirected': [2.90, 1.11, 3.74, 1.36, 3.89, 2.46],
    'Both':       [4.43, 1.24, 4.30, 2.57, 4.44, 3.28],
}

plot_data = [
    {'title': r'$k=1$', 'means': means_k1, 'stds': stds_k1},
    {'title': r'$k=10$', 'means': means_k10, 'stds': stds_k10},
    {'title': r'$k=100$', 'means': means_k100, 'stds': stds_k100},
]

# --- 3. Plotting Logic ---
fig, axes = plt.subplots(1, 3, sharey=True)

# Use a colorblind-friendly color palette
colors = plt.cm.Set1(np.linspace(0, 1, 3))

# Bar chart settings
x = np.arange(len(algorithms))  # The label locations
n_groups = len(graph_types)
total_width = 0.8
width = total_width / n_groups

# Loop through each k-value to create the three subplots
for i, (ax, data) in enumerate(zip(axes, plot_data)):
    for j, (graph_type, color) in enumerate(zip(graph_types, colors)):
        offset = (j - (n_groups - 1) / 2) * width
        means = data['means'][graph_type]
        stds = data['stds'][graph_type]
        
        ax.bar(x + offset, means, width, yerr=stds, label=graph_type, color=color,
               capsize=2, ecolor='gray', error_kw={'linewidth': 0.5})

    # --- Formatting for each subplot ---
    ax.set_title(data['title'])
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.6)
    ax.set_axisbelow(True) # Ensure grid is behind bars
    
    # Hide top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- 4. Global Formatting ---
axes[0].set_ylabel('Improvement Factor (Log Scale)')

# Create a single, shared legend below the plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15),
           ncol=len(graph_types), fancybox=True, frameon=False)

# Adjust layout to prevent labels from overlapping and make space for legend
plt.tight_layout(rect=[0, 0.05, 1, 1]) # rect=[left, bottom, right, top]

# --- 5. Save the Figure ---
# Save as PDF for high-quality vector graphics in LaTeX
plt.savefig("complex_networks_chart.pdf", bbox_inches='tight')
print("Graph saved as complex_networks_chart.pdf")

# (Optional) Save as high-DPI PNG
# plt.savefig("complex_networks_chart.png", dpi=300, bbox_inches='tight')

plt.show()