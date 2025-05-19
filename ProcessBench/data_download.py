# Import necessary libraries
from datasets import load_dataset # For loading datasets from Hugging Face
import pandas as pd # For data manipulation and CSV handling
import ast # For safely evaluating string representations of Python literals (e.g., lists)

def save_split_to_csv(dataset_name, split_name, csv_filename):
    """
    Loads a specific split of a dataset and saves it to a CSV file.

    Args:
        dataset_name (str): The name of the dataset on Hugging Face Hub (e.g., "Qwen/ProcessBench").
        split_name (str): The name of the split to load (e.g., "gsm8k", "math").
        csv_filename (str): The name of the CSV file to save the data to.
    """
    # Load the specified split of the dataset
    print(f"Loading {split_name} split from {dataset_name}...")
    dataset = load_dataset(dataset_name, split=split_name)
    
    # Convert to pandas DataFrame
    print("Converting to pandas DataFrame...")
    df = pd.DataFrame(dataset)
    
    # Save to CSV file
    df.to_csv(csv_filename, index=False)
    print(f"Saved {split_name} split to {csv_filename}")

def create_model_response_column(csv_file_path):
    """
    Reads a CSV file, creates a new 'model_response' column based on the 'steps' column,
    ensuring the result is a pure string. The 'steps' column is expected to contain
    string representations of lists of strings.

    Args:
        csv_file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: DataFrame with the new 'model_response' column.
    """
    print(f"Reading CSV file: {csv_file_path}...")
    df = pd.read_csv(csv_file_path)

    def parse_steps_list(steps_str):
        """
        Safely parses a string representation of a list.
        """
        try:
            # Check if it is a string and looks like a list
            if isinstance(steps_str, str) and steps_str.strip().startswith('[') and steps_str.strip().endswith(']'):
                # ast.literal_eval is a safer way to parse literals
                return ast.literal_eval(steps_str)
            else:
                # If it's not a string or doesn't look like a list, return the original value
                return steps_str
        except (ValueError, SyntaxError):
            # If parsing fails, it might be due to incorrect format.
            print(f"Warning: Could not parse steps string: {steps_str}. Returning as is.")
            # Return original value or None, depending on desired handling for parse failures.
            return steps_str 

    print("Parsing 'steps' column...")
    # Try to parse the string representation of a list in the 'steps' column into an actual list
    df['steps'] = df['steps'].apply(parse_steps_list)

    print("Creating 'model_response' column...")
    # Create the new 'model_response' column
    # Ensure the join operation is performed only when 'steps' is a list of strings.
    # Otherwise, convert the content to string.
    df['model_response'] = df['steps'].apply(
        lambda x: "\n\n".join(str(item) for item in x) if isinstance(x, list) else (x if isinstance(x, str) else str(x))
    )

    return df

if __name__ == "__main__":
    # Define the dataset name from Hugging Face Hub
    dataset_name = "Qwen/ProcessBench"
    # Define the specific split to be processed
    split_name_to_process = "gsm8k"
    # Define the CSV filename to be used for saving and processing
    csv_file_name = f"{split_name_to_process}.csv" # e.g., "gsm8k.csv"

    # --- Part 1: Download and save the dataset split ---
    # You can choose to save other splits by uncommenting and modifying the lines below
    # save_split_to_csv(dataset_name, "math", "processbench_math.csv")
    
    # Save the specified split (e.g., gsm8k)
    save_split_to_csv(dataset_name, split_name_to_process, csv_file_name)
    
    print(f"\n--- Finished downloading and saving {csv_file_name} ---\n")

    # --- Part 2: Process the CSV to create the 'model_response' column ---
    # Process the CSV file that was just saved
    df_with_response = create_model_response_column(csv_file_name)

    # Print the first few rows to inspect the 'steps' and 'model_response' columns
    print("\nPreview of 'steps' and 'model_response' columns:")
    print(df_with_response[['steps', 'model_response']].head())

    # Check the data type of elements in the 'model_response' column
    print("\nType of 'model_response' column elements:")
    # This will show the count of each data type found in the 'model_response' column (should primarily be str)
    print(df_with_response['model_response'].apply(type).value_counts())

    # Save the DataFrame with the new 'model_response' column back to the same CSV file, overwriting it
    print(f"\nSaving processed data back to {csv_file_name}...")
    df_with_response.to_csv(csv_file_name, index=False)
    print(f"Successfully saved processed data to {csv_file_name}")
