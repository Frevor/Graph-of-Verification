# Graph-of-Verification
# LLM Evaluation Scripts

This repository contains Python scripts for evaluating Large Language Models (LLMs) on two main tasks: Number Triangle Summation and ProcessBench.

## Repository Structure

```

.
├── Number Triangle Summation/
│   ├── generate\_problems.py
│   ├── NTS\_baseline.py
│   └── NTS\_GoV.py
│
├── ProcessBench/
│   ├── data\_download.py
│   ├── processbench\_baseline.py
│   └── processbench\_GoV.py
│
├── .env.example
└── README.md

````

## Running the Scripts

The general workflow for both tasks is to first run the data acquisition/generation script, followed by either the `baseline` or `GoV` (Generation of Verification) evaluation script.

### 1. Number Triangle Summation

This task involves generating "digital triangle" math problems and then evaluating an LLM's ability to find errors in their solutions.

* **Step 1: Generate Problems**
    Run `generate_problems.py` to create the dataset for this task. This will typically output a CSV file (e.g., `number_triangle_problems.csv`).
    ```bash
    python "Number Triangle Summation/generate_problems.py"
    ```

* **Step 2: Run Evaluation**
    After generating the problems, you can run either the baseline evaluation or the GoV evaluation:
    * For baseline evaluation:
        ```bash
        python "Number Triangle Summation/NTS_baseline.py"
        ```
    * For GoV evaluation:
        ```bash
        python "Number Triangle Summation/NTS_GoV.py"
        ```
    These scripts will process the generated data and produce output files with the LLM's analysis.

### 2. ProcessBench

This task evaluates LLM performance on the ProcessBench dataset (likely a version of GSM8K or similar mathematical reasoning problems).

* **Step 1: Download/Prepare Data**
    Run `data_download.py` to obtain or prepare the necessary dataset for ProcessBench. Ensure this script places the data where the evaluation scripts expect it (e.g., `gsm8k.csv`).
    ```bash
    python ProcessBench/data_download.py
    ```

* **Step 2: Run Evaluation**
    Once the data is ready, run either the baseline or GoV evaluation scripts:
    * For baseline evaluation:
        ```bash
        python ProcessBench/processbench_baseline.py
        ```
    * For GoV evaluation:
        ```bash
        python ProcessBench/processbench_GoV.py
        ```
    These scripts will output CSV files containing the results of the LLM's processing.

## Important Notes

* **Model Name**: The specific LLM model used (`MODEL_NAME`) is defined within each script. If you wish to use a different model, you will need to update this variable in the respective Python files.
* **Output Files**: The evaluation scripts will generate output files (usually CSVs) in their respective directories or as defined within the scripts.
