# Graph-of-Verification (GoV)

This repository is the official implementation for the paper: **Graph of Verification: Structured Verification of LLM Reasoning with Directed Acyclic Graphs** ([arXiv:2506.12509](https://arxiv.org/abs/2506.12509)).

[](https://arxiv.org/abs/2506.12509)

Both humans and GoV validate reasoning by decomposing it into a directed acyclic graph, allowing for flexible verification granularity adapted to different tasks.

## Abstract

Verifying the complex, multi-step reasoning of Large Language Models (LLMs) is a critical challenge, as holistic methods often overlook localized flaws. To address this, we propose the **Graph of Verification (GoV)**, a novel, training-free framework that enhances the reasoning and error-detection capabilities of LLMs. GoV models the verification process as a Directed Acyclic Graph (DAG) and introduces a flexible node block architecture. This allows GoV to adapt its verification granularity—from atomic steps in formal proofs to entire paragraphs in natural language narratives—to match the native structure of the reasoning process]. Our experiments show that GoV significantly outperforms both holistic baselines and other state-of-the-art methods, establishing a new standard for training-free reasoning verification.

## How GoV Works

GoV operationalizes structured validation through a four-stage pipeline that models reasoning as a Directed Acyclic Graph (DAG).

Figure: The GoV Four-Stage Verification Pipeline.

1.  **DAG Construction**: The raw reasoning process is modeled as a DAG, where nodes represent individual steps (premises, conclusions) and edges represent logical dependencies.
2.  **Topological Sorting**: A topological sort of the graph enforces causal consistency, ensuring that premises are always verified before the conclusions that depend on them.
3.  **Sequential Verification**: An LLM assesses each reasoning unit (an atomic node or a block of nodes) in the sorted order, using previously validated steps as context.
4.  **Verification Outcome**: The process terminates at the first detected error, enabling precise fault localization. A reasoning chain is only considered valid if all its units are verified as correct.

This framework navigates a two-dimensional design space:

  * **Verification Granularity**: The scale of the unit being verified, from fine-grained `Atomic Nodes` (e.g., a single equation) for precision to coarse-grained `Node Blocks` (e.g., a paragraph) for robustness.
  * **Contextual Scope**: The amount of prior information provided, from `Minimal Context` (only direct premises) to `Inclusive Context` (all previously verified steps).

## Repository Structure

```
.
├── Number Triangle Summation/
│   ├── generate_problems.py      # Generates NTS datasets
│   ├── NTS_baseline.py           # Runs holistic baseline verification on NTS
│   ├── NTS_GoV.py                # Runs GoV (atomic-level) verification on NTS
│   └── analyze_results.py        # Compares baseline and GoV results and plots F1 scores
|
├── ProcessBench/
│   ├── data_download.py          # Downloads and prepares the ProcessBench (GSM8K) dataset
│   ├── processbench_baseline.py  # Runs holistic baseline verification on ProcessBench
│   └── processbench_GoV.py       # Runs GoV (block-level) verification on ProcessBench
│
├── .env.example                  # Example environment file for API keys
├── requirements.txt              # Python package dependencies
└── README.md
```

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/Graph-of-Verification.git
    cd Graph-of-Verification
    ```

2.  **Install dependencies:**
    Create a virtual environment and install the required packages.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    Create a `.env` file by copying the example file:

    ```bash
    cp .env.example .env
    ```

    Open the `.env` file and add your API key:

    ```env
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

    # [cite_start]Optional: If using a custom endpoint, specify the base URL [cite: 4, 5]
    OPENAI_API_BASE="YOUR_OPTIONAL_API_BASE_URL_HERE"
    ```

    > **Note**: The LLM model name (e.g., `Qwen2.5-72B-Instruct`) is hardcoded in the Python scripts. To use a different model, you must modify the `MODEL_NAME` variable inside each script.

## Running the Experiments

The experiments are divided into two main tasks that showcase GoV's versatility.

### Experiment 1: Number Triangle Summation (Well-Structured Task)

This task evaluates GoV's precision on a formal arithmetic task with an unambiguous dependency graph. GoV is configured with **Atomic Granularity**, treating each addition as a single verification node.

  * **Step 1: Generate Problems**
    Create the datasets for N = 2, 4, 6, and 8.

    ```bash
    python "Number Triangle Summation/generate_problems.py"
    ```

  * **Step 2: Run GoV & Baseline Evaluation**
    Execute the evaluation scripts. They will process the generated files and produce new CSVs with verification results.

    ```bash
    # Run GoV (step-by-step verification)
    python "Number Triangle Summation/NTS_GoV.py"

    # Run Baseline (holistic verification)
    python "Number Triangle Summation/NTS_baseline.py"
    ```

  * **Step 3: Analyze Results**
    Compare the performance of the two methods and generate a performance plot.

    ```bash
    python "Number Triangle Summation/analyze_results.py"
    ```

    This will save a summary CSV and a PNG plot comparing the F1 scores.

### Experiment 2: ProcessBench (Loosely-Structured Task)

This task evaluates GoV's robustness on mathematical reasoning problems expressed in natural language. GoV is configured with **Block Granularity** and **Inclusive Context**, treating each paragraph as a node block and providing all previously verified paragraphs as context.

  * **Step 1: Download and Prepare Data**
    This script downloads the `gsm8k` split from the `Qwen/ProcessBench` dataset on Hugging Face and prepares it.

    ```bash
    python ProcessBench/data_download.py
    ```

    This will create the `gsm8k.csv` file in the `ProcessBench` directory.

  * **Step 2: Run GoV & Baseline Evaluation**
    Run the scripts to get verification results for the ProcessBench dataset.

    ```bash
    # Run GoV (paragraph-by-paragraph verification)
    python ProcessBench/processbench_GoV.py

    # Run Baseline (holistic verification)
    python ProcessBench/processbench_baseline.py
    ```

    The scripts will generate output CSVs (e.g., `gsm8k_qwen2.5-32b_gov.csv`) containing the detailed verification results.

## Citation

If you find our work useful for your research, please consider citing our paper:

```bibtex
@misc{fang2025graphverificationstructuredverification,
      title={Graph of Verification: Structured Verification of LLM Reasoning with Directed Acyclic Graphs}, 
      author={Jiwei Fang and Bin Zhang and Changwei Wang and Jin Wan and Zhiwei Xu},
      year={2025},
      eprint={2506.12509},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.12509}, 
}
```
