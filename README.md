# Social Network Analysis & Complex Networks

A comprehensive collection of network analysis implementations and experiments, including graph algorithms, community detection, centrality measures, and parallel computing approaches for large-scale network analysis.

## Overview

This repository contains coursework and research projects focused on social network analysis and complex networks. The work spans from fundamental graph metrics and visualizations to advanced implementations of state-of-the-art algorithms for computing centrality measures efficiently.

## Repository Structure

```
.
├── assignment-1/          # Network analysis fundamentals
├── assignment-2/          # Twitter network analysis
├── project/              # Final research project
│   ├── sequential/       # Sequential implementation of top-k closeness centrality
│   └── parallel/         # Parallel/optimized implementation
└── sn_env/               # Python virtual environment
```

### Assignment 1: Network Analysis Fundamentals

Analysis of medium and large-scale directed networks, including:
- Network statistics (nodes, edges, components)
- Degree distributions (in-degree and out-degree)
- Distance distributions and average path lengths
- Network visualization and clustering analysis

**Key Files:**
- [q_1_2_network_analysis.py](assignment-1/q_1_2_network_analysis.py) - Basic network loading and statistics
- [q3_get_indegree_and_outdegree_distributions.py](assignment-1/q3_get_indegree_and_outdegree_distributions.py) - Degree distribution analysis
- [q4.py](assignment-1/q4.py) to [q8.py](assignment-1/q8.py) - Various network analysis tasks

### Assignment 2: Twitter Network Analysis

Comprehensive analysis of Twitter mention networks (small and large datasets):
- Network topology analysis (components, density, clustering)
- Degree and distance distributions
- Community detection using Louvain algorithm
- Weight distribution analysis
- Bounded BFS for efficient diameter estimation
- Top-20 user identification

**Key Files:**
- [network_analysis.py](assignment-2/network_analysis.py) - Main analysis pipeline
- [community_detection.py](assignment-2/community_detection.py) - Community structure analysis
- [bfs_bounded_improved.py](assignment-2/bfs_bounded_improved.py) - Efficient diameter approximation
- [analyze_twitter_fast.py](assignment-2/analyze_twitter_fast.py) - Optimized analysis for large networks

### Final Project: Fast Top-k Closeness Centrality

Implementation and analysis of the fast top-k closeness centrality algorithm by Bergamini et al. The project includes both sequential and parallel implementations with comprehensive experimental validation.

#### Sequential Implementation ([project/final/sequential/](project/final/sequential/))

A complete Python/NetworkX implementation of the fast top-k closeness centrality algorithm with:
- Core `updateBoundsLB` pruning strategy
- Support for weighted and unweighted graphs
- Experimental validation on 7 real-world networks
- Convergence analysis and visualization

**Features:**
- Comparison with textbook algorithm
- Pruning power analysis
- Runtime improvement factors
- Bounds convergence tracking

See the [sequential README](project/final/sequential/README.md) for detailed setup and usage instructions.

#### Parallel Implementation ([project/final/parallel/](project/final/parallel/))

CPU-parallelized version of the algorithm with:
- Multi-core BFS exploration
- Parallel bounds computation
- Performance scaling analysis

See the [parallel README](project/final/parallel/Readme.md) for setup and usage instructions.

## Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd snacs
```

2. Create and activate a virtual environment:
```bash
# macOS/Linux
python3 -m venv sn_env
source sn_env/bin/activate

# Windows
python -m venv sn_env
sn_env\Scripts\activate
```

3. Install dependencies:
```bash
# For assignment work
pip install networkx pandas matplotlib numpy

# For final project (sequential)
cd project/final/sequential
pip install -r requirements.txt

# For final project (parallel)
cd project/final/parallel
pip install -r requirements.txt
```

## Usage

### Running Assignment Scripts

Navigate to the respective assignment directory and run the Python scripts:

```bash
cd assignment-1
python q_1_2_network_analysis.py

cd ../assignment-2
python network_analysis.py
```

### Running Final Project Experiments

**Sequential experiments:**
```bash
cd project/final/sequential
python run_experiments.py
python plot_generator.py
```

**Parallel experiments:**
```bash
cd project/final/parallel
python run_experiments.py
```

## Datasets

The project uses various network datasets:
- **Assignment 1:** Custom student datasets (medium and large TSV files)
- **Assignment 2:** Twitter mention networks (small and large)
- **Final Project:** Multiple real-world networks including:
  - Wiki-Vote
  - Facebook Combined
  - Email-Eu-core
  - CA-GrQc
  - Les Misérables
  - Norwegian Boards
  - Facebook Forum
  - Cit-HepTh

Note: Dataset files are excluded from version control (see [.gitignore](.gitignore)).

## Key Technologies

- **NetworkX**: Graph analysis and algorithms
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Visualization and plotting
- **NumPy**: Numerical computations
- **Python-Louvain**: Community detection

## Results

The project generates various outputs:
- Network statistics and metrics
- Degree and distance distribution plots
- Community structure visualizations
- Runtime comparison charts
- Convergence analysis plots

All generated plots and results are saved in respective `plots/` directories within each project folder.

## License

Academic project - please refer to course guidelines for usage and citation requirements.
