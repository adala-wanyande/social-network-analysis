# Fast Top-k Closeness Centrality (With CPU Parallelization)

This repository implements the fast top-k closeness centrality algorithm by Bergamini et al., with a focus on CPU-based parallel execution. The implementation includes the algorithm’s pruning strategy and supports both weighted and unweighted graphs.

## Project Structure

The repository is organized into the following components:

-   `data_loader.py`: A module for loading and pre-processing all network datasets.
-   `centrality_algorithms.py`: The core module containing implementations of the textbook and fast top-k algorithms.
-   `run_experiments.py`: The main executable script to run the full experimental pipeline.
-   ` networks/`: A directory containing all the network dataset files.
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
    -   `Cit-HepTh.txt`

## How to Run the Experiments

The entire experimental pipeline can be executed with one main command for parallelization results.

### Step 1: Generate the Experimental Data

Run the main experiment script. This will execute both the textbook and fast algorithms on all seven datasets for k-values of 1, 10, and 100. Each run is repeated 3 times to gather statistics. This process can be time-consuming.

```bash
python run_experiments.py
```

This will produce the below output file in the root directory:
-   `experiment_results_[timestamp].csv`: A CSV file containing the complete results (runtimes, pruning power, etc.) for all runs.


