import matplotlib.pyplot as plt
import numpy as np

# --- 1. Configure Matplotlib for a consistent, professional look ---
plt.rcParams.update({
    "text.usetex": False, # Using standard Matplotlib rendering as requested
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": (8, 4), # Adjusted size for a 2-panel plot
})

# --- 2. Data Preparation for Big Networks ---
# X-axis will be k-values, groups will be graph types
k_values_labels = [r'$k=1$', r'$k=10$', r'$k=100$']
graph_types = ['Directed', 'Undirected', 'Both']

# Data for Street Networks (NBBound)
means_street = {
    'Directed':   [742.42, 724.72, 686.32],
    'Undirected': [1681.93, 1673.41, 1566.72],
    'Both':       [1117.46, 1101.25, 1036.95],
}
stds_street = {
    'Directed':   [2.60, 2.67, 2.76],
    'Undirected': [2.88, 2.92, 3.04],
    'Both':       [2.97, 3.03, 3.13],
}

# Data for Complex Networks (DegBound)
means_complex = {
    'Directed':   [247.65, 117.45, 59.96],
    'Undirected': [551.51, 115.30, 49.01],
    'Both':       [339.70, 116.59, 55.37],
}
stds_complex = {
    'Directed':   [11.92, 9.72, 8.13],
    'Undirected': [10.68, 4.87, 2.93],
    'Both':       [11.78, 7.62, 5.86],
}

plot_data = [
    {'title': 'Big Street Networks (NBBound)', 'means': means_street, 'stds': stds_street},
    {'title': 'Big Complex Networks (DegBound)', 'means': means_complex, 'stds': stds_complex}
]

# --- 3. Plotting Logic ---
# Note: 1 row, 2 columns for the two network types
fig, axes = plt.subplots(1, 2, sharey=True)
colors = plt.cm.Set1(np.linspace(0, 1, len(graph_types)))
x = np.arange(len(k_values_labels))
n_groups = len(graph_types)
total_width = 0.8
width = total_width / n_groups

# Loop through the two data sets (Street, Complex) to create the two subplots
for ax, data in zip(axes, plot_data):
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
    ax.set_xticklabels(k_values_labels)
    ax.set_xlabel('$k$ Value')
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- 4. Global Formatting ---
axes[0].set_ylabel('Improvement Factor (Log Scale)')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1),
           ncol=len(graph_types), fancybox=True, frameon=False)
plt.tight_layout(rect=[0, 0.05, 1, 1])

# --- 5. Save the Figure ---
plt.savefig("big_networks_chart.pdf", bbox_inches='tight')
print("Graph saved as big_networks_chart.pdf")

plt.show()