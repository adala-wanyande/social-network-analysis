# Fast Top-k Closeness Centrality (Sequential Experiments)

This project provides a Python implementation and experimental analysis of the fast top-k closeness centrality algorithm proposed by Bergamini et al. in "Computing top-k Closeness Centrality Faster in Unweighted Graphs". The implementation includes the paper's core `updateBoundsLB` pruning strategy and extends the algorithm to support weighted graphs.

The primary goal of this work is to validate the algorithm's logical correctness and analyze its empirical performance in a Python/NetworkX environment, revealing a trade-off between its theoretical efficiency and practical implementation overhead.

## Project Structure

The repository is organized into the following components:

-   `requirements.txt`: A list of all required Python packages.
-   `data_loader.py`: A module for loading and pre-processing all network datasets.
-   `centrality_algorithms.py`: The core module containing implementations of the textbook and fast top-k algorithms.
-   `run_experiments.py`: The main executable script to run the full experimental pipeline.
-   `plot_generator.py`: A script to generate all tables and figures from the experimental results.

-   `/networks/`: A directory containing all the network dataset files.
-   `/plots/`: A directory where all generated figures are saved.

## Setup and Installation

To run this project, you will need Python 3.8+ and the packages listed in `requirements.txt`. It is highly recommended to use a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-folder>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Datasets:**
    All network datasets must be downloaded and placed inside the `/networks` directory. Ensure the filenames match those specified in `data_loader.py`. The required datasets are:
    -   `Wiki-Vote.txt`
    -   `facebook_combined.txt`
    -   `email-Eu-core.txt`
    -   `CA-GrQc.txt`
    -   `lesmis.gml`
    -   `norwegian-boards.txt`
    -   `facebook-forum.txt`

## How to Run the Experiments

The entire experimental pipeline can be executed with two main commands.

### Step 1: Generate the Experimental Data

Run the main experiment script. This will execute both the textbook and fast algorithms on all seven datasets for k-values of 1, 10, and 100. Each run is repeated 3 times to gather statistics. This process can be time-consuming.

```bash
python run_experiments.py
```

This will produce two types of output files in the root directory:
-   `experiment_results_[timestamp].csv`: A CSV file containing the complete results (runtimes, pruning power, etc.) for all runs.
-   `convergence_log_[dataset]_[k].csv`: CSV files containing detailed data for the bounds convergence plots for specified datasets.

### Step 2: Generate Plots and Tables

Once the experiments are complete, run the plotting script. This script will automatically find the latest results CSV and generate all visualizations.

```bash
python plot_generator.py
```

This script will:
1.  Print a LaTeX-formatted summary table to the console.
2.  Create a `/plots` directory (if it doesn't exist).
3.  Save the final figures (`pruning_power_by_k.png`, `improvement_factors_by_k.png`, and convergence plots) into the `/plots` directory.
