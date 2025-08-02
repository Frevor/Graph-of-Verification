# analyze_results.py (Comparing Baseline vs. GoV)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# List of N values for the experiment
N_VALUES = [2, 4, 6, 8]

# compare 'baseline' and 'GoV'
METHODS = ["baseline_stronger", "GoV"]

METHOD_NAMES = {
    "baseline_stronger": "Baseline (Equations-Only)",
    "GoV": "GoV (Ours)"
}
PALETTE = {
    "Baseline (Equations-Only)": "salmon",
    "GoV (Ours)": "skyblue"
}


def calculate_metrics(df):
    """Calculates Correct Accuracy, Error Accuracy, and F1 Score."""
    # Convert 'label' and 'llm_verified_error_step' to numeric, coercing errors
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df['llm_verified_error_step'] = pd.to_numeric(df['llm_verified_error_step'], errors='coerce')
    
    # Ground truth correct/incorrect
    correct_gt = df[df['label'] == -1]
    error_gt = df[df['label'] != -1]
    
    # Correct Accuracy: From all problems that are actually correct, how many did we identify as correct?
    true_positives_correct = len(correct_gt[correct_gt['llm_verified_error_step'] == -1])
    correct_accuracy = true_positives_correct / len(correct_gt) if len(correct_gt) > 0 else 0
    
    # Error Accuracy: From all problems that are actually incorrect, how many did we identify correctly (at the right step)?
    true_positives_error = len(error_gt[error_gt['label'] == error_gt['llm_verified_error_step']])
    error_accuracy = true_positives_error / len(error_gt) if len(error_gt) > 0 else 0
    
    # F1 Score
    f1 = 2 * (correct_accuracy * error_accuracy) / (correct_accuracy + error_accuracy) if (correct_accuracy + error_accuracy) > 0 else 0
    
    return {
        "Correct Acc (%)": correct_accuracy * 100,
        "Error Acc (%)": error_accuracy * 100,
        "F1 Score (%)": f1 * 100
    }

def main():
    results_data = []

    for n_val in N_VALUES:
        for method in METHODS:
            filename = f"number_triangle_problems_verified_{method}_N{n_val}.csv"
            try:
                df = pd.read_csv(filename)
                metrics = calculate_metrics(df)
                results_data.append({
                    "N": n_val,
                    "Method": METHOD_NAMES[method],
                    **metrics
                })
            except FileNotFoundError:
                print(f"Warning: File not found, skipping: {filename}")
                continue

    if not results_data:
        print("No results found. Please run the verification scripts first.")
        return

    # --- Print and save the summary table ---
    summary_df = pd.DataFrame(results_data)
    print("--- Summary of Results (Stronger Baseline vs. GoV) ---")
    # Format numbers to two decimal places for cleaner output
    pd.options.display.float_format = '{:,.2f}'.format
    print(summary_df.to_string(index=False))
    summary_df.to_csv("summary_results_stronger_vs_gov.csv", index=False)
    print("\nSummary table saved to summary_results_stronger_vs_gov.csv")

    # --- Generate the plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=summary_df,
        x="N",
        y="F1 Score (%)", # Use the correct column name
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        palette=PALETTE,
        ax=ax,
        linewidth=2.5,
        markersize=8
    )

    ax.set_title("GoV vs. Stronger Baseline by Problem Complexity (N)", fontsize=16, pad=20)
    ax.set_xlabel("Number of Initial Values (N)", fontsize=12)
    ax.set_ylabel("F1 Score (%)", fontsize=12)
    ax.set_xticks(N_VALUES)
    ax.legend(title="Verification Method", fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plot_filename = "performance_stronger_vs_gov.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"\nPlot saved to {plot_filename}")
    plt.show()

if __name__ == "__main__":
    main()