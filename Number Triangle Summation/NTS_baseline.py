# NTS_baseline.py

import os
import csv
import re
import asyncio
import logging
import pandas as pd
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("OPENAI_API_BASE")
if not API_KEY: raise ValueError("OPENAI_API_KEY not set.")
CLIENT = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL) if API_BASE_URL else AsyncOpenAI(api_key=API_KEY)

MODEL_NAME = "Qwen2.5-72B-Instruct" # Choose model here
MAX_CONCURRENT_REQUESTS = 4
MAX_RETRIES = 4
INITIAL_TEMPERATURE = 0.0
RETRY_TEMPERATURE = 0.3
RETRY_DELAY_SECONDS_BASE = 5
N_VALUES = [2, 4, 6, 8] # The N values for the experiment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_STRONGER = """Your task is to verify the following list of calculations. Review them in order and return the 1-based index of the first calculation that is incorrect. If all calculations are correct, return -1.

[Calculations]
{calculations_list}

Please put your final answer (i.e., the index) in \\boxed{{}}.
"""

# --- Helper Functions ---

def parse_solution_steps_with_layer_info(response_text: str) -> List[Tuple[str, int]]:
    """
    Parses the 'response' string to extract individual calculation steps
    and the original "Step X" number (layer calculation step) they belong to.
    This is crucial for mapping the LLM's output index back to the correct step label.
    (Copied from GoV script for consistency)
    """
    items: List[Tuple[str, int]] = []
    current_layer_step_number = 0
    step_line_pattern = re.compile(r"^\s*Step\s*(\d+)\s*\(Calculating Layer.*$", re.IGNORECASE)
    calculation_pattern = re.compile(r"^\s*(-?\d+)\s*\+\s*(-?\d+)\s*=\s*(-?\d+)\s*$")

    for line in response_text.splitlines():
        step_match = step_line_pattern.match(line)
        if step_match:
            current_layer_step_number = int(step_match.group(1))
            continue
        stripped_line = line.strip()
        if calculation_pattern.match(stripped_line) and current_layer_step_number > 0:
            items.append((stripped_line, current_layer_step_number))
            
    return items

def extract_boxed_answer(text: str) -> Optional[int]:
    """Extracts the integer value from a \\boxed{} tag."""
    match = re.search(r"\\boxed\{(-?\d+)\}", text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            logger.warning(f"Could not convert extracted value to int: {match.group(1)}")
            return None
    return None

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=RETRY_DELAY_SECONDS_BASE, max=RETRY_DELAY_SECONDS_BASE * 4),
    retry=retry_if_exception_type(Exception),
    before=lambda retry_state: logger.info(
        f"{'Attempting' if retry_state.attempt_number == 1 else 'Retrying'} API call "
        f"(attempt {retry_state.attempt_number}/{retry_state.retry_object.stop.max_attempt_number}) "
        f"for Problem ID {retry_state.args[0]}"
    )
)
async def call_llm_api_with_retries(problem_id: str, prompt: str, temperature: float) -> Optional[str]:
    """Calls the LLM API with the provided prompt and temperature."""
    try:
        response = await CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Problem ID {problem_id}: API call failed: {e}")
        raise

async def process_row_stronger_baseline(row_data: dict, semaphore: asyncio.Semaphore) -> dict:
    """
    Processes a single row for the stronger baseline.
    """
    solution_text = row_data.get("response")
    problem_id = row_data.get("id", "N/A")

    if not solution_text:
        row_data["llm_verified_error_step"] = "MISSING_DATA"
        row_data["llm_raw_response"] = ""
        return row_data
        
    # 1. Parse the solution to get calculations and their original step numbers
    parsed_steps = parse_solution_steps_with_layer_info(solution_text)

    # If there are no calculations, the solution is considered correct.
    if not parsed_steps:
        row_data["llm_verified_error_step"] = -1
        row_data["llm_raw_response"] = "No calculations to verify."
        return row_data

    # 2. Format the calculations for the new prompt
    calculation_strings = [calc for calc, layer in parsed_steps]
    formatted_calcs = "\n".join([f"{i+1}. {calc}" for i, calc in enumerate(calculation_strings)])
    
    prompt = PROMPT_TEMPLATE_STRONGER.format(calculations_list=formatted_calcs)
    
    llm_output_index = None
    raw_llm_response_text = ""
    current_temperature = INITIAL_TEMPERATURE

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                raw_llm_response_text = await call_llm_api_with_retries(
                    problem_id, prompt, current_temperature
                )
                if raw_llm_response_text:
                    llm_output_index = extract_boxed_answer(raw_llm_response_text)
                    if llm_output_index is not None:
                        logger.info(f"Problem ID {problem_id}: Successfully extracted index {llm_output_index}.")
                        break
                current_temperature = RETRY_TEMPERATURE
            except Exception as e:
                logger.error(f"Problem ID {problem_id}: All API retries failed. Error: {e}")
                break
    
    # 3. Map the LLM's output index back to the original step number
    final_verified_step = "EXTRACTION_FAILED"
    if llm_output_index is not None:
        if llm_output_index == -1:
            final_verified_step = -1 # All correct
        # Check if the index is valid (1-based and within the list of calculations)
        elif 1 <= llm_output_index <= len(parsed_steps):
            # Get the original layer number from our parsed list
            # llm_output_index is 1-based, list is 0-indexed
            final_verified_step = parsed_steps[llm_output_index - 1][1]
        else:
            final_verified_step = "INVALID_INDEX_RETURNED" # LLM returned an out-of-bounds index

    row_data["llm_verified_error_step"] = final_verified_step
    row_data["llm_raw_response"] = raw_llm_response_text or ""
    return row_data


async def main():
    if not API_KEY:
        logger.error("API key not found.")
        return

    for n_val in N_VALUES:
        input_csv_file = f"number_triangle_problems_N{n_val}.csv"
        output_csv_file = f"number_triangle_problems_verified_baseline_stronger_N{n_val}.csv"
        
        logger.info(f"--- Processing Stronger Baseline for N = {n_val} ---")
        logger.info(f"Input file: {input_csv_file}")
        
        try:
            df = pd.read_csv(input_csv_file)
            if df.empty: continue
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_csv_file}. Skipping.")
            continue
        
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        tasks = [process_row_stronger_baseline(row, semaphore) for row in df.to_dict('records')]

        logger.info(f"Starting stronger baseline verification for {len(tasks)} problems (N={n_val})...")
        processed_results = await asyncio.gather(*tasks)
        logger.info(f"Stronger baseline verification completed for N={n_val}.")

        output_df = pd.DataFrame(processed_results)
        output_df.to_csv(output_csv_file, index=False, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"Results for N={n_val} written to {output_csv_file}\n")

if __name__ == "__main__":
    asyncio.run(main())
