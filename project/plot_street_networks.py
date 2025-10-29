import matplotlib.pyplot as plt
import numpy as np

# --- 1. Configure Matplotlib for a consistent, professional look ---
# Using the same settings as before to match the previous plot.
plt.rcParams.update({
    "text.usetex": False,  # Using standard Matplotlib rendering as requested
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": (8, 3.5), # Keep figure size consistent
})

# --- 2. Data Preparation for Street Networks ---
algorithms = ["Olh", "Ocl", "DegCut", "DegBound", "NBCut", "NBBound"]
graph_types = ['Directed', 'Undirected', 'Both']

# Data for k=1
means_k1 = {
    'Directed':   [4.11, 3.39, 4.14, 187.10, 4.12, 250.66],
    'Undirected': [4.36, 3.23, 4.06, 272.22, 4.00, 382.47],
    'Both':       [4.23, 3.31, 4.10, 225.69, 4.06, 309.63],
}
stds_k1 = {
    'Directed':   [1.83, 1.28, 2.07, 1.65, 2.07, 1.71],
    'Undirected': [2.18, 1.28, 2.06, 1.67, 2.07, 1.63],
    'Both':       [2.01, 1.28, 2.07, 1.72, 2.07, 1.74],
}

# Data for k=10
means_k10 = {
    'Directed':   [4.04, 2.93, 4.09, 172.06, 4.08, 225.26],
    'Undirected': [4.28, 2.81, 4.01, 245.96, 3.96, 336.47],
    'Both':       [4.16, 2.87, 4.05, 205.72, 4.02, 275.31],
}
stds_k10 = {
    'Directed':   [1.83, 1.24, 2.07, 1.65, 2.07, 1.71],
    'Undirected': [2.18, 1.24, 2.06, 1.68, 2.07, 1.68],
    'Both':       [2.01, 1.24, 2.07, 1.72, 2.07, 1.76],
}

# Data for k=100
means_k100 = {
    'Directed':   [4.03, 2.90, 3.91, 123.91, 3.92, 149.02],
    'Undirected': [4.27, 2.79, 3.84, 164.65, 3.80, 201.42],
    'Both':       [4.15, 2.85, 3.87, 142.84, 3.86, 173.25],
}
stds_k100 = {
    'Directed':   [1.82, 1.24, 2.07, 1.56, 2.08, 1.59],
    'Undirected': [2.18, 1.24, 2.07, 1.67, 2.09, 1.69],
    'Both':       [2.01, 1.24, 2.07, 1.65, 2.08, 1.67],
}

plot_data = [
    {'title': r'$k=1$', 'means': means_k1, 'stds': stds_k1},
    {'title': r'$k=10$', 'means': means_k10, 'stds': stds_k10},
    {'title': r'$k=100$', 'means': means_k100, 'stds': stds_k100},
]

# --- 3. Plotting Logic (identical to the previous script) ---
fig, axes = plt.subplots(1, 3, sharey=True)
colors = plt.cm.Set1(np.linspace(0, 1, 3))
x = np.arange(len(algorithms))
n_groups = len(graph_types)
total_width = 0.8
width = total_width / n_groups

for i, (ax, data) in enumerate(zip(axes, plot_data)):
    for j, (graph_type, color) in enumerate(zip(graph_types, colors)):
        offset = (j - (n_groups - 1) / 2) * width
        means = data['means'][graph_type]
        stds = data['stds'][graph_type]
        
        ax.bar(x + offset, means, width, yerr=stds, label=graph_type, color=color,
               capsize=2, ecolor='gray', error_kw={'linewidth': 0.5})

    ax.set_title(data['title'])
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- 4. Global Formatting ---
axes[0].set_ylabel('Improvement Factor (Log Scale)')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15),
           ncol=len(graph_types), fancybox=True, frameon=False)
plt.tight_layout(rect=[0, 0.05, 1, 1])

# --- 5. Save the Figure ---
# **CRITICAL: Changed filename to avoid overwriting the previous plot**
plt.savefig("street_networks_chart.pdf", bbox_inches='tight')
print("Graph saved as street_networks_chart.pdf")

plt.show()