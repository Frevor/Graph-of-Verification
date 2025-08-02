# NTS_GoV.py

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
if not API_KEY: raise ValueError("OPENAI_API_KEY not set.")
CLIENT = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL) if API_BASE_URL else AsyncOpenAI(api_key=API_KEY)

MODEL_NAME = "Qwen2.5-72B-Instruct" # Choose model here
MAX_CONCURRENT_REQUESTS = 4
MAX_API_CALL_RETRIES_PER_STEP = 3
MAX_EXTRACTION_ATTEMPTS_PER_STEP = 3
INITIAL_TEMPERATURE = 0.0
RETRY_TEMPERATURE = 0.3
RETRY_DELAY_SECONDS_BASE = 5
N_VALUES = [2,4,6,8]

LLM_USER_PROMPT_STEPWISE_TEMPLATE = "Please verify whether the calculation {calculation} is correct, and put your final conclusion (i.e., 'correct' or 'incorrect') in \\boxed{{}}."

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def parse_solution_steps_with_layer_info(response_text: str) -> List[Tuple[str, int]]:
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

def extract_llm_step_verdict(llm_response: Optional[str]) -> Optional[str]:
    if not llm_response: return None
    match = re.search(r"\\boxed\{([\s\S]*?)\}", llm_response, re.IGNORECASE)
    text_to_check = match.group(1).lower() if match else llm_response.lower()
    if "incorrect" in text_to_check: return "incorrect"
    if "correct" in text_to_check: return "correct"
    return None

@retry(
    stop=stop_after_attempt(MAX_API_CALL_RETRIES_PER_STEP),
    wait=wait_exponential(multiplier=1, min=RETRY_DELAY_SECONDS_BASE, max=RETRY_DELAY_SECONDS_BASE * 4),
    retry=retry_if_exception_type(Exception),
    before=lambda retry_state: logger.info(
        f"Retrying API call for Problem ID {retry_state.args[2]}, Step {retry_state.args[3]} (Attempt {retry_state.attempt_number})"
    )
)
async def call_llm_for_step_verification(
    calculation_str: str, temperature: float, problem_id_for_log: str, step_num_for_log: int
) -> Optional[str]:
    user_prompt = LLM_USER_PROMPT_STEPWISE_TEMPLATE.format(calculation=calculation_str)
    try:
        response = await CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Problem ID {problem_id_for_log}, Step {step_num_for_log}: API call failed: {e}")
        raise

async def process_row_stepwise(row_data: Dict[str, Any], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    solution_response_text = row_data.get("response", "")
    problem_id_for_log = row_data.get("id", "N/A")

    if not solution_response_text:
        row_data["llm_verified_error_step"] = "NO_RESPONSE_DATA"
        row_data["llm_stepwise_raw_responses"] = ""
        return row_data

    calculations_with_layer = parse_solution_steps_with_layer_info(solution_response_text)
    
    if not calculations_with_layer:
        row_data["llm_verified_error_step"] = -1
        row_data["llm_stepwise_raw_responses"] = "No calculation steps found."
        return row_data

    final_error_step_to_report = -1
    all_step_raw_responses_list = []

    async with semaphore:
        for i, (calc_str, layer_step_of_calc) in enumerate(calculations_with_layer):
            overall_calc_idx = i + 1
            step_verdict = None
            current_raw_llm_response = None
            current_temp = INITIAL_TEMPERATURE

            for attempt in range(MAX_EXTRACTION_ATTEMPTS_PER_STEP):
                try:
                    current_raw_llm_response = await call_llm_for_step_verification(
                        calc_str, current_temp, problem_id_for_log, overall_calc_idx
                    )
                    if current_raw_llm_response:
                        step_verdict = extract_llm_step_verdict(current_raw_llm_response)
                        if step_verdict: break
                    current_temp = RETRY_TEMPERATURE
                except Exception as e:
                    current_raw_llm_response = f"API_CALL_FAILED: {e}"
                    break
            
            if current_raw_llm_response:
                all_step_raw_responses_list.append(f"Calc{overall_calc_idx}(L{layer_step_of_calc}): {current_raw_llm_response}")

            if step_verdict == "incorrect":
                final_error_step_to_report = layer_step_of_calc
                break
            if step_verdict is None:
                final_error_step_to_report = f"VERIF_FAIL_L{layer_step_of_calc}_C{overall_calc_idx}"
                break

    row_data["llm_verified_error_step"] = final_error_step_to_report
    row_data["llm_stepwise_raw_responses"] = "; ".join(all_step_raw_responses_list)
    return row_data

async def main():
    if not API_KEY:
        logger.error("API key not found.")
        return

    for n_val in N_VALUES:
        input_csv_file = f"number_triangle_problems_N{n_val}.csv"
        output_csv_file = f"number_triangle_problems_verified_GoV_N{n_val}.csv"

        logger.info(f"--- Processing GoV for N = {n_val} ---")
        logger.info(f"Input file: {input_csv_file}")
        
        try:
            df = pd.read_csv(input_csv_file)
            if df.empty:
                logger.warning(f"{input_csv_file} is empty. Skipping.")
                continue
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_csv_file}. Skipping.")
            continue
        
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        tasks = [process_row_stepwise(row, semaphore) for row in df.to_dict('records')]

        logger.info(f"Starting GoV LLM verification for {len(tasks)} problems (N={n_val})...")
        processed_results = await asyncio.gather(*tasks)
        logger.info(f"GoV LLM verification completed for N={n_val}.")

        output_df = pd.DataFrame(processed_results)
        output_df.to_csv(output_csv_file, index=False, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"Results for N={n_val} written to {output_csv_file}\n")

if __name__ == "__main__":
    asyncio.run(main())
