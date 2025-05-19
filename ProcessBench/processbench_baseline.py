import asyncio
import pandas as pd
import json
import os
from openai import AsyncOpenAI, RateLimitError, APIError, APIConnectionError
# Use standard tqdm, not tqdm.asyncio specifically for the loop wrapper
from tqdm import tqdm
from dotenv import load_dotenv
import logging
import time
import re # <<< --- Added import for regular expressions

# --- Configuration ---
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("OPENAI_API_BASE")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

INPUT_CSV_PATH = 'gsm8k.csv'  # <<< --- INPUT CSV FILE PATH
OUTPUT_CSV_PATH = 'gsm8k_qwen2.5-32b_baseline.csv' # <<< --- Adjusted output name for critique task
MAX_CONCURRENT_REQUESTS = 4
# --->>> Using the model name from your log - consider if 'mini' is suitable for error detection, maybe use gpt-4-turbo if needed
API_MODEL = "qwen2.5-32b-instruct"
REQUEST_TIMEOUT = 60  # Timeout for each API request in seconds
RETRY_ATTEMPTS = 5    # Number of retry attempts on specific errors
RETRY_DELAY = 5       # Initial delay between retries in seconds (will increase exponentially)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- System Prompt Removed ---

# Initialize the Async OpenAI Client
if API_BASE_URL:
    logging.info(f"Using custom API base URL: {API_BASE_URL}")
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL, timeout=REQUEST_TIMEOUT)
else:
    logging.info("Using default OpenAI API base URL.")
    client = AsyncOpenAI(api_key=API_KEY, timeout=REQUEST_TIMEOUT)

def extract_boxed_index(text):
    """
    Extracts the integer index from the \boxed{} marker in the text.
    Returns the integer index or None if not found or not a valid integer.
    """
    if not isinstance(text, str):
        return None
    # Regex to find \boxed{ followed by optional minus sign and digits }
    match = re.search(r"\\boxed\{(-?\d+)\}", text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            logging.warning(f"Found \\boxed{{...}} but content is not an integer: {match.group(1)}")
            return None # Found the box but content isn't an int
    return None # Pattern not found

async def process_row(row_data, semaphore):
    """
    Processes a single row: formats input, calls API for critique, extracts index.
    Returns a dictionary containing raw_output, extracted_index, and error (if any).
    """
    async with semaphore: # Acquire semaphore before making API call
        row_id = row_data.get('id', 'N/A') # Use 'id' or fallback
        problem = str(row_data.get('problem', '')) # Get problem from CSV
        model_response = str(row_data.get('model_response', ''))

        # Ensure row_id is hashable
        try:
            hash(row_id)
        except TypeError:
            logging.warning(f"Row ID '{row_id}' is not hashable. Using its string representation.")
            row_id = str(row_id)

        if not model_response:
            logging.warning(f"Skipping row {row_id}: Missing 'model_response'.")
            # Return structure consistent with other returns
            return row_id, {"raw_output": None, "extracted_index": None, "error": "Missing input data (model_response)"}

        if not problem:
            logging.warning(f"Skipping row {row_id}: Missing 'problem'.")
            return row_id, {"raw_output": None, "extracted_index": None, "error": "Missing input data (problem)"}

        # --- Start: Input Formatting as per Request ---
        paragraphs = model_response.strip().split('\n\n')
        formatted_paragraphs = []
        for i, para in enumerate(paragraphs):
            para_content = para.strip()
            if para_content: # Only include non-empty paragraphs
                formatted_paragraphs.append(f"<paragraph_{i}>\n{para_content}\n</paragraph_{i}>")
        solution_block = "\n".join(formatted_paragraphs)

        # --- Construct the new user message ---
        user_message = f"""
The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):
[Math Problem]
{problem}

[Solution]
{solution_block}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").
Please put your final answer (i.e., the index) in \\boxed{{}}.
"""
        # Note: Escaped curly braces \\boxed{{}} for the f-string
        # --- End: Input Formatting and New User Message ---


        # Construct the messages list (System Prompt removed)
        messages = [
            # {"role": "system", "content": SYSTEM_PROMPT}, # Removed
            {"role": "user", "content": user_message}
        ]

        current_attempt = 0
        while current_attempt < RETRY_ATTEMPTS:
            try:
                logging.debug(f"Requesting critique for ID: {row_id} (Attempt {current_attempt + 1})")
                response = await client.chat.completions.create(
                    model=API_MODEL,
                    messages=messages,
                    temperature=0.0, # Keep low temperature for deterministic tasks
                    # No response_format needed as we expect plain text
                )

                content = response.choices[0].message.content
                logging.debug(f"Raw response for ID {row_id}: {content[:200]}...") # Log beginning of response

                # --- Start: Extract Index from Response ---
                extracted_index = extract_boxed_index(content)

                if extracted_index is not None:
                    logging.info(f"Successfully processed ID: {row_id}. Extracted index: {extracted_index}")
                    return row_id, {"raw_output": content, "extracted_index": extracted_index, "error": None}
                else:
                    logging.warning(f"Could not extract index from \\boxed{{}} for ID {row_id}. Raw content: {content}")
                    # Still return the raw content, but flag the extraction failure
                    return row_id, {"raw_output": content, "extracted_index": None, "error": "Failed to extract index from response"}
                # --- End: Extract Index from Response ---

            # --- [Keep the existing exception handling for API errors, retries, etc.] ---
            except RateLimitError as e:
                current_attempt += 1
                wait_time = RETRY_DELAY * (2 ** current_attempt) # Exponential backoff
                logging.warning(f"Rate limit exceeded for ID {row_id}. Retrying in {wait_time} seconds... (Attempt {current_attempt}/{RETRY_ATTEMPTS})")
                if current_attempt >= RETRY_ATTEMPTS:
                    logging.error(f"Rate limit error persisted for ID {row_id} after {RETRY_ATTEMPTS} attempts: {e}")
                    # Return error in the standard dictionary format
                    return row_id, {"raw_output": None, "extracted_index": None, "error": f"Rate limit error: {e}"}
                await asyncio.sleep(wait_time)
            except (APIError, APIConnectionError) as e:
                 current_attempt += 1
                 wait_time = RETRY_DELAY * (2 ** current_attempt) # Exponential backoff
                 logging.warning(f"API Error/Connection Error for ID {row_id}: {e}. Retrying in {wait_time} seconds... (Attempt {current_attempt}/{RETRY_ATTEMPTS})")
                 if current_attempt >= RETRY_ATTEMPTS:
                     logging.error(f"API/Connection error persisted for ID {row_id} after {RETRY_ATTEMPTS} attempts: {e}")
                     return row_id, {"raw_output": None, "extracted_index": None, "error": f"API/Connection error: {e}"}
                 await asyncio.sleep(wait_time)
            except Exception as e:
                # Check for common errors related to model access or configuration
                if "Could not find model" in str(e) or "Invalid API key" in str(e) or "authentication" in str(e).lower():
                     logging.error(f"Configuration Error for ID {row_id}: {e}. Check API Key, Base URL, and Model Name.")
                     # Stop retrying for config errors
                     return row_id, {"raw_output": None, "extracted_index": None, "error": f"Configuration Error: {e}"}
                # Handle other unexpected API errors
                logging.error(f"Unexpected API error for ID {row_id} (Attempt {current_attempt + 1}): {e}")
                current_attempt += 1 # Still retry for potentially transient issues
                wait_time = RETRY_DELAY * (2 ** current_attempt)
                if current_attempt >= RETRY_ATTEMPTS:
                    logging.error(f"Unexpected API error persisted for ID {row_id} after {RETRY_ATTEMPTS} attempts: {e}")
                    return row_id, {"raw_output": None, "extracted_index": None, "error": f"Unexpected API error: {e}"}
                logging.warning(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)


        # Safeguard return if all retries fail without specific errors being caught
        logging.error(f"Failed to process ID {row_id} after all retries.")
        return row_id, {"raw_output": None, "extracted_index": None, "error": "Failed after multiple retries"}


async def main():
    """
    Main function to read CSV, run tasks concurrently, and save results.
    """
    logging.info(f"Starting script. Reading CSV: {INPUT_CSV_PATH}")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        # Required columns for this task
        required_cols = ['problem', 'model_response']
        if not all(col in df.columns for col in required_cols):
             raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")
        if 'id' not in df.columns:
            logging.warning("CSV does not contain an 'id' column. Using DataFrame index as ID.")
            df['id'] = df.index # Add an ID column if missing
        else:
            # Ensure IDs are suitable as dict keys later
            if not df['id'].is_unique:
                logging.warning("IDs in the 'id' column are not unique. Results for duplicate IDs might overwrite each other.")
            # Convert ID to string just in case to ensure hashability consistency
            df['id'] = df['id'].astype(str)

    except FileNotFoundError:
        logging.error(f"Error: Input CSV file not found at {INPUT_CSV_PATH}")
        return
    except ValueError as ve:
        logging.error(f"Error reading CSV or missing columns: {ve}")
        return
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return

    logging.info(f"Read {len(df)} rows from CSV.")
    logging.info(f"Using model: {API_MODEL}")
    logging.info(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    # Pass row dictionary directly
    tasks = [process_row(row, semaphore) for row in df.to_dict('records')]

    results_map = {} # Use a dictionary to store results mapped by row_id
    processed_count = 0
    error_count = 0
    extraction_failed_count = 0 # Count rows where index extraction failed

    logging.info("Starting API calls for critique...")
    try:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Rows for Critique"):
            try:
                row_id, result_data = await future
                # Ensure result_data is always a dict, even if process_row failed unexpectedly before returning one
                if not isinstance(result_data, dict):
                     logging.error(f"Task for row {row_id} returned unexpected type: {type(result_data)}. Recording as error.")
                     results_map[row_id] = {"raw_output": None, "extracted_index": None, "error": "Internal error: Invalid return type from process_row"}
                     error_count += 1
                     continue # Skip further processing for this malformed result

                results_map[row_id] = result_data # Store the result dictionary

                # Check for errors reported by process_row
                if result_data.get("error"):
                    if result_data["error"] == "Failed to extract index from response":
                        extraction_failed_count += 1
                        logging.debug(f"Row {row_id} processed, but index extraction failed.")
                        # It's not a full error in terms of API call, but a parsing issue
                    else:
                        error_count += 1
                        logging.debug(f"Row {row_id} processed with error: {result_data['error']}")
                elif result_data.get("extracted_index") is not None:
                     processed_count += 1
                     logging.debug(f"Row {row_id} processed successfully, index extracted.")
                else:
                     # Should ideally not happen if error is None and index is None, but handle defensively
                     error_count += 1
                     logging.warning(f"Row {row_id} had no error but also no extracted index. Raw: {result_data.get('raw_output')}")
                     results_map[row_id]["error"] = "Inconsistent state: No error but no index found"


            except Exception as e:
                # This catches errors *during* the await future, less likely if process_row handles its own exceptions
                logging.error(f"Error awaiting a task result: {e}. This might indicate a problem in process_row's error handling or asyncio itself.")
                # We don't know the row_id here easily, so just count a general error
                error_count += 1


    except Exception as e:
         logging.error(f"An error occurred during the main task execution loop: {e}")


    logging.info("API calls finished processing.")
    logging.info(f"Attempted processing for {len(tasks)} rows.")
    logging.info(f"Successfully processed (API call OK, index extracted): {processed_count} rows.")
    logging.info(f"Processed (API call OK, but index extraction failed): {extraction_failed_count} rows.")
    logging.info(f"Encountered API errors or other failures: {error_count} rows.")

    # --- Add results to the DataFrame ---
    # Define new column names
    raw_output_col = 'model_critique_output'
    index_col = 'first_error_paragraph_index'
    error_col = 'processing_error' # Optional: Add a column for the error message

    # Helper function to safely get values from the result dictionary
    def get_result_value(row_id, key, default=None):
        return results_map.get(str(row_id), {}).get(key, default) # Ensure row_id is string for lookup

    df[raw_output_col] = df['id'].apply(lambda x: get_result_value(x, 'raw_output'))
    df[index_col] = df['id'].apply(lambda x: get_result_value(x, 'extracted_index'))
    df[error_col] = df['id'].apply(lambda x: get_result_value(x, 'error')) # Add error message column

    # Save the updated DataFrame
    try:
        df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        logging.info(f"Successfully saved results to {OUTPUT_CSV_PATH}")
    except Exception as e:
        logging.error(f"Error saving results to CSV: {e}")

if __name__ == "__main__":
    # Keep the asyncio loop handling as before
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        print("Asyncio loop already running. Adding main() to the loop.")
        tsk = loop.create_task(main())
    else:
        print("Starting new asyncio event loop.")
        asyncio.run(main())
