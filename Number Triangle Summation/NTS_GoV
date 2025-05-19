import os
import csv
import re
import asyncio
import logging
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("OPENAI_API_BASE")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

if API_BASE_URL:
    CLIENT = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)
else:
    CLIENT = AsyncOpenAI(api_key=API_KEY)

MODEL_NAME = "Qwen2.5-32B-Instruct" # You should choose model here

INPUT_CSV_FILE = "number_triangle_problems.csv"
OUTPUT_CSV_FILE = "number_triangle_problems_verified_GoV.csv"

MAX_CONCURRENT_REQUESTS = 5
MAX_API_CALL_RETRIES_PER_STEP = 3 # Tenacity retries for API call issues
MAX_EXTRACTION_ATTEMPTS_PER_STEP = 3 # Outer loop for extraction (total attempts for a step)
INITIAL_TEMPERATURE = 0.0
RETRY_TEMPERATURE = 0.3
RETRY_DELAY_SECONDS_BASE = 5 # Base for tenacity exponential backoff

LLM_SYSTEM_PROMPT_STEPWISE = "Please think step by step, and put your final conclusion (i.e., 'correct' or 'incorrect') in \\boxed{}."
LLM_USER_PROMPT_STEPWISE_TEMPLATE = "Verify whether the {calculation} is correct."

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def parse_solution_steps_with_layer_info(response_text: str) -> List[Tuple[str, int]]:
    """
    Parses the 'response' string to extract individual calculation steps
    and the original "Step X" number (layer calculation step) they belong to.
    Returns a list of tuples: (calculation_string, layer_step_number).
    """
    items: List[Tuple[str, int]] = []
    current_layer_step_number = 0 # Will be 1-indexed from "Step X" line

    # Regex to find lines like "Step 1 (Calculating Layer 1 from Layer 0):"
    # Extracts the "1" from "Step 1"
    step_line_pattern = re.compile(r"^\s*Step\s*(\d+)\s*\(Calculating Layer.*$", re.IGNORECASE)
    
    # Regex for calculation lines like "123 + 456 = 579"
    calculation_pattern = re.compile(r"^\s*(-?\d+)\s*\+\s*(-?\d+)\s*=\s*(-?\d+)\s*$")

    for line in response_text.splitlines():
        # Try to match "Step X" line first (using original line for pattern matching)
        step_match = step_line_pattern.match(line)
        if step_match:
            current_layer_step_number = int(step_match.group(1))
            # logger.debug(f"Detected Layer Step: {current_layer_step_number}")
            continue # This line is a step header, not a calculation

        # If it's not a step header, try to match a calculation
        stripped_line = line.strip()
        if calculation_pattern.match(stripped_line):
            if current_layer_step_number > 0: # Ensure calculation is under a valid "Step X"
                items.append((stripped_line, current_layer_step_number))
                # logger.debug(f"Found calculation '{stripped_line}' for Layer Step {current_layer_step_number}")
            else:
                logger.warning(f"Found calculation '{stripped_line}' but not under a recognized 'Step X' heading. Ignoring.")
                
    return items

def extract_llm_step_verdict(llm_response: Optional[str]) -> Optional[str]:
    """
    Robustly extracts 'correct' or 'incorrect' from the LLM's response.
    Prioritizes 'incorrect' if both keywords are somehow present.
    """
    if not llm_response:
        return None

    text_to_check = ""
    # Try to find \boxed{} content first, case-insensitive for \boxed tag itself
    # Content within box is then lowercased for keyword checking.
    # [\s\S] matches any character including newlines. *? makes it non-greedy.
    match = re.search(r"\\boxed\{([\s\S]*?)\}", llm_response, re.IGNORECASE)
    
    if match:
        text_to_check = match.group(1).lower()
        # logger.debug(f"Boxed content: '{text_to_check}'")
    else:
        # Fallback: if no \boxed{}, check the whole response (less reliable).
        logger.warning(f"LLM response for step did not contain \\boxed{{}}. Checking full response: {llm_response[:100]}...")
        text_to_check = llm_response.lower()

    # Order is important: check for "incorrect" before "correct"
    if "incorrect" in text_to_check:
        return "incorrect"
    elif "correct" in text_to_check:
        return "correct"
    
    logger.warning(f"Could not extract 'correct' or 'incorrect' from text: '{text_to_check[:100]}'")
    return None

# Decorator for retrying API calls with exponential backoff
@retry(
    stop=stop_after_attempt(MAX_API_CALL_RETRIES_PER_STEP),
    wait=wait_exponential(multiplier=1, min=RETRY_DELAY_SECONDS_BASE, max=RETRY_DELAY_SECONDS_BASE * 4),
    retry=retry_if_exception_type(Exception), # Catch broad API/network errors
    before=lambda retry_state: logger.info(
        f"API Call for step verification (Attempt {retry_state.attempt_number}/"
        f"{retry_state.retry_object.stop.max_attempt_number}) "
        f"for Problem ID {retry_state.args[2]}, Step {retry_state.args[3]}"
        + (f" | Prev. error: {retry_state.outcome.exception()}" if retry_state.attempt_number > 1 and retry_state.outcome and retry_state.outcome.failed else "")
    )
)
async def call_llm_for_step_verification(
    calculation_str: str, 
    temperature: float, 
    problem_id_for_log: str, 
    step_num_for_log: int
) -> Optional[str]:
    """
    Calls the LLM API for a single calculation step verification.
    problem_id_for_log and step_num_for_log are for logging purposes only.
    """
    user_prompt = LLM_USER_PROMPT_STEPWISE_TEMPLATE.format(calculation=calculation_str)
    try:
        response = await CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT_STEPWISE},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Problem ID {problem_id_for_log}, Step {step_num_for_log}: API call failed (temp {temperature}): {e}")
        raise # Reraise to be caught by tenacity or propagate if all retries fail

async def process_row_stepwise(row_data: Dict[str, Any], semaphore: asyncio.Semaphore, row_index: int) -> Dict[str, Any]:
    problem_description_text = row_data.get("problem", "")
    solution_response_text = row_data.get("response", "")
    problem_id_for_log = row_data.get("id", f"CSVRow-{row_index+1}")

    if not solution_response_text:
        logger.warning(f"Problem ID {problem_id_for_log}: Missing 'response' data. Skipping.")
        row_data["llm_verified_error_step"] = "NO_RESPONSE_DATA" # Column name corrected
        row_data["llm_stepwise_raw_responses"] = ""
        return row_data

    # Use the new parsing function
    calculations_with_layer = parse_solution_steps_with_layer_info(solution_response_text)
    
    if not calculations_with_layer:
        logger.info(f"Problem ID {problem_id_for_log}: No calculation steps found in response. Marking as correct by default.")
        row_data["llm_verified_error_step"] = -1 # Column name corrected
        row_data["llm_stepwise_raw_responses"] = ""
        return row_data

    final_error_step_to_report = -1 # Default to -1 (all correct)
    all_step_raw_responses_list = []

    # The semaphore here limits how many process_row_stepwise coroutines run concurrently.
    # Each call to call_llm_for_step_verification is an API call.
    # If MAX_CONCURRENT_REQUESTS is 5, then up to 5 rows will be processed in parallel,
    # and each row processes its calculation steps sequentially one-by-one.
    # This means at any given time, there are at most 5 LLM API calls active.
    async with semaphore:
        # Iterate through (calculation_string, layer_step_number_it_belongs_to)
        for i, (calc_str, layer_step_of_calc) in enumerate(calculations_with_layer):
            # overall_calc_idx is the 1-based index of this specific calculation among all calculations
            overall_calc_idx = i + 1 
            
            logger.info(f"Problem ID {problem_id_for_log}: Verifying Calc {overall_calc_idx} (in Layer Step {layer_step_of_calc}): '{calc_str}'")
            
            step_verdict = None
            current_raw_llm_response = None
            current_temp = INITIAL_TEMPERATURE

            for attempt in range(MAX_EXTRACTION_ATTEMPTS_PER_STEP):
                api_call_succeeded = False
                try:
                    current_raw_llm_response = await call_llm_for_step_verification(
                        calc_str, current_temp, problem_id_for_log, overall_calc_idx # Log with overall_calc_idx
                    )
                    api_call_succeeded = True
                except Exception as e:
                    logger.error(f"Problem ID {problem_id_for_log}, Calc {overall_calc_idx} in Layer Step {layer_step_of_calc}: All API call retries failed. Error: {e}")
                    current_raw_llm_response = f"API_CALL_FAILED: {e}"
                
                if current_raw_llm_response:
                     all_step_raw_responses_list.append(f"Calc{overall_calc_idx}(LStep{layer_step_of_calc},Att{attempt+1},T{current_temp}): {current_raw_llm_response[:200]}")
                
                if api_call_succeeded:
                    step_verdict = extract_llm_step_verdict(current_raw_llm_response)
                    if step_verdict == "correct":
                        logger.info(f"Problem ID {problem_id_for_log}, Calc {overall_calc_idx} in Layer Step {layer_step_of_calc}: Verified as CORRECT.")
                        break 
                    elif step_verdict == "incorrect":
                        logger.info(f"Problem ID {problem_id_for_log}, Calc {overall_calc_idx} in Layer Step {layer_step_of_calc}: Verified as INCORRECT.")
                        final_error_step_to_report = layer_step_of_calc # <--- STORE THE LAYER STEP NUMBER
                        break 
                    else: 
                        logger.warning(f"Problem ID {problem_id_for_log}, Calc {overall_calc_idx} in Layer Step {layer_step_of_calc}: Attempt {attempt + 1} to extract verdict failed.")
                        current_temp = RETRY_TEMPERATURE
                        if attempt < MAX_EXTRACTION_ATTEMPTS_PER_STEP - 1:
                             await asyncio.sleep(1)
                else: # API call failed completely for this attempt (e.g. all tenacity retries exhausted)
                    break # Break extraction attempts loop if API call failed

            if step_verdict == "incorrect": # If an error was found for this calc
                break # Stop processing further calculations for this problem
            
            if step_verdict is None: # If after all extraction attempts, still no clear verdict for this calc
                logger.error(f"Problem ID {problem_id_for_log}, Calc {overall_calc_idx} in Layer Step {layer_step_of_calc}: Could not verify. Halting for this problem.")
                # Report failure with layer and specific calculation index for clarity
                final_error_step_to_report = f"VERIF_FAIL_L{layer_step_of_calc}_C{overall_calc_idx}"
                break # Stop processing further calculations for this problem

    # Use the corrected column name "llm_verified_error_step"
    row_data["llm_verified_error_step"] = final_error_step_to_report 
    row_data["llm_stepwise_raw_responses"] = "; ".join(all_step_raw_responses_list)
    return row_data

async def main():
    if not API_KEY:
        logger.error("OpenAI API key not found.")
        return

    try:
        df = pd.read_csv(INPUT_CSV_FILE)
        logger.info(f"Successfully read {len(df)} rows from {INPUT_CSV_FILE}")
    except FileNotFoundError:
        logger.error(f"Input CSV file not found: {INPUT_CSV_FILE}")
        return
    except Exception as e:
        logger.error(f"Error reading CSV '{INPUT_CSV_FILE}': {e}")
        return

    if df.empty:
        logger.info("Input CSV is empty.")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []
    for i, row_dict in enumerate(df.to_dict('records')): # df.to_dict('records') gives list of dicts
        tasks.append(process_row_stepwise(row_dict, semaphore, i))

    logger.info(f"Starting stepwise LLM verification for {len(tasks)} problems...")
    processed_results_list = await asyncio.gather(*tasks, return_exceptions=True) # Capture exceptions from tasks
    logger.info("Stepwise LLM verification completed for all problems.")

    # Reconstruct DataFrame from list of dictionaries
    results_df_list = []
    for i, res in enumerate(processed_results_list):
        if isinstance(res, Exception):
            logger.error(f"Error processing CSV row index {i} (Problem ID may vary): {res}")
            # Get original row and add error info
            original_row = df.iloc[i].to_dict()
            original_row["llm_stepwise_verified_error_step"] = f"TASK_PROCESSING_ERROR: {type(res).__name__}"
            original_row["llm_stepwise_raw_responses"] = str(res)
            results_df_list.append(original_row)
        else:
            results_df_list.append(res)
    
    output_df = pd.DataFrame(results_df_list)

    try:
        output_df.to_csv(OUTPUT_CSV_FILE, index=False, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"Results written to {OUTPUT_CSV_FILE}")
    except Exception as e:
        logger.error(f"Error writing to output CSV '{OUTPUT_CSV_FILE}': {e}")

if __name__ == "__main__":
    asyncio.run(main())
