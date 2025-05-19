import os
import csv
import asyncio
import re
import logging
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
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
OUTPUT_CSV_FILE = "number_triangle_problems_verified_baseline.csv"

MAX_CONCURRENT_REQUESTS = 5
MAX_RETRIES = 4
INITIAL_TEMPERATURE = 0.0 # For consistent deterministic output for verification
RETRY_TEMPERATURE = 0.3 # If initial extraction fails
RETRY_DELAY_SECONDS_BASE = 5 # Base for exponential backoff

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """The following is a math problem and a solution (split into steps, with the indices starting from 1):
[Math Problem]
{problem}

[Solution]
{solution}

Your task is to review and critique the solution step by step. Once you identify an error in a step, return the index of the step where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").
Please put your final answer (i.e., the index) in \\boxed{{}}.
"""

# --- Helper Functions ---

def extract_boxed_answer(text: str) -> int | None:
    """Extracts the integer value from a \\boxed{} tag."""
    match = re.search(r"\\boxed\{(-?\d+)\}", text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            logger.warning(f"Could not convert extracted value to int: {match.group(1)}")
            return None
    return None

# Decorator for retrying API calls with exponential backoff for specific API errors
@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=RETRY_DELAY_SECONDS_BASE, max=RETRY_DELAY_SECONDS_BASE * 4),
    retry=retry_if_exception_type(
        (
            Exception # Catching a broad range of OpenAI and network errors
        )
    ),
    before=lambda retry_state: logger.info(
        f"{'Attempting' if retry_state.attempt_number == 1 else 'Retrying'} API call (attempt {retry_state.attempt_number}/{retry_state.retry_object.stop.max_attempt_number}) "
        f"for Problem ID {retry_state.args[0] if retry_state.args and len(retry_state.args) > 0 else 'N/A'}" +
        (f" due to: {retry_state.outcome.exception()}" if retry_state.attempt_number > 1 and retry_state.outcome else "")
    )
)
async def call_llm_api_with_retries(problem_id: int, problem_text: str, solution_text: str, temperature: float) -> str | None:
    """
    Calls the LLM API with specified temperature.
    Handles API errors with retries.
    """
    prompt = PROMPT_TEMPLATE.format(problem=problem_text, solution=solution_text)
    try:
        response = await CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Problem ID {problem_id}: API call failed after several retries with temperature {temperature}: {e}")
        raise # Reraise to trigger tenacity's retry or stop

async def process_row(row_data: dict, semaphore: asyncio.Semaphore, row_index: int) -> dict:
    """
    Processes a single row from the CSV: calls LLM, extracts answer, handles retries for extraction.
    """
    problem_text = row_data.get("problem")
    solution_text = row_data.get("response") # Assuming 'response' column contains the solution
    problem_id = row_data.get("id", f"Row-{row_index}") # Use an ID if available, else row index

    if not problem_text or not solution_text:
        logger.warning(f"Problem ID {problem_id}: Missing 'problem' or 'response' data. Skipping.")
        row_data["llm_verified_error_step"] = "MISSING_DATA"
        row_data["llm_raw_response"] = ""
        return row_data

    llm_verified_step = None
    raw_llm_response_text = ""

    async with semaphore: # Limits concurrent API calls
        current_temperature = INITIAL_TEMPERATURE
        for attempt in range(MAX_RETRIES):
            logger.info(f"Problem ID {problem_id}: Attempt {attempt + 1}/{MAX_RETRIES} with temperature {current_temperature}")
            try:
                # This inner call has its own tenacity retries for API/network issues
                raw_llm_response_text = await call_llm_api_with_retries(
                    problem_id, problem_text, solution_text, current_temperature
                )

                if raw_llm_response_text:
                    llm_verified_step = extract_boxed_answer(raw_llm_response_text)
                    if llm_verified_step is not None:
                        logger.info(f"Problem ID {problem_id}: Successfully extracted step {llm_verified_step}.")
                        break # Success
                    else:
                        logger.warning(f"Problem ID {problem_id}: Could not extract answer from LLM response (Temp: {current_temperature}). Full response: {raw_llm_response_text[:200]}...")
                        # If extraction fails, set temperature for next retry, if any
                        current_temperature = RETRY_TEMPERATURE
                else:
                    # This case should ideally be handled by tenacity if call_llm_api_with_retries returns None
                    logger.error(f"Problem ID {problem_id}: Received empty response from LLM (Temp: {current_temperature}).")
                    current_temperature = RETRY_TEMPERATURE


            except Exception as e: # Catch exceptions from call_llm_api_with_retries if all its retries fail
                logger.error(f"Problem ID {problem_id}: All API call retries failed for attempt {attempt + 1}. Error: {e}")
                # No need to change temperature here as the API call itself failed
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY_SECONDS_BASE * (2**attempt) / 2) # Shorter sleep as tenacity already waited
                # Let the loop continue for the next main attempt if any

            if llm_verified_step is not None: # check again if extracted in this attempt
                break
            elif attempt < MAX_RETRIES - 1:
                 logger.info(f"Problem ID {problem_id}: Preparing for retry {attempt + 2} for extraction/API issue.")
                 await asyncio.sleep(RETRY_DELAY_SECONDS_BASE) # Simple delay before next main attempt if extraction failed

    row_data["llm_verified_error_step"] = llm_verified_step if llm_verified_step is not None else "EXTRACTION_FAILED_ALL_RETRIES"
    row_data["llm_raw_response"] = raw_llm_response_text if raw_llm_response_text else ""
    return row_data


async def main():
    """
    Main function to read CSV, process rows concurrently, and write results.
    """
    if not API_KEY:
        logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return

    try:
        with open(INPUT_CSV_FILE, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            rows = list(reader)
            if not rows:
                logger.info("Input CSV is empty.")
                return
            fieldnames = reader.fieldnames
    except FileNotFoundError:
        logger.error(f"Input CSV file not found: {INPUT_CSV_FILE}")
        return
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []
    for i, row in enumerate(rows):
        # You might want to add a unique ID to each row if not present, for better tracking
        # For example: row['id'] = row.get('id', i)
        tasks.append(process_row(row, semaphore, i))

    logger.info(f"Starting LLM verification for {len(tasks)} rows...")
    processed_results = await asyncio.gather(*tasks)
    logger.info("LLM verification completed.")

    # Add new fieldnames if they don't exist
    if "llm_verified_error_step" not in fieldnames:
        fieldnames.append("llm_verified_error_step")
    if "llm_raw_response" not in fieldnames:
        fieldnames.append("llm_raw_response")

    try:
        with open(OUTPUT_CSV_FILE, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(processed_results)
        logger.info(f"Results written to {OUTPUT_CSV_FILE}")
    except Exception as e:
        logger.error(f"Error writing to output CSV: {e}")

if __name__ == "__main__":
    # On Windows, you might need this for asyncio if you encounter issues with the event loop policy
    # if os.name == 'nt':
    # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
