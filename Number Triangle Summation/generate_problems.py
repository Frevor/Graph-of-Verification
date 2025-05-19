import random
import csv

def generate_triangle_problem(N, create_error=False):
    """
    Generates a single number triangle summation problem and its solution process.

    Args:
        N (int): The number of initial numbers.
        create_error (bool): Whether to introduce an error in the solution process.

    Returns:
        tuple: Contains the problem description, N, the list of initial numbers, the solution process description,
               whether the final answer is correct (bool), and the error layer (int, -1 if no error).
    """
    if N < 2:
        raise ValueError("N must be at least 2 to form a triangle.")

    numbers = [random.randint(1000, 9999) for _ in range(N)]
    original_numbers_str = ", ".join(map(str, numbers))

    problem_description = f"Given {N} numbers at layer 0: {original_numbers_str}. " \
                          f"Calculate the sum of adjacent numbers to form the next layer, " \
                          f"repeating this process until only one number remains at layer {N-1}. " \
                          f"What is the final number?"

    solution_steps = []
    current_layer_numbers = list(numbers)
    # all_layers_calculations = [] # Stores tuples of (num1, num2, correct_sum, actual_sum_used) for each step

    error_introduced = False
    error_layer_index = -1 # 0-indexed layer where error occurs (layer 1 is index 0 for sums)
    # error_step_in_layer = -1 # 0-indexed sum within the error layer. Not strictly needed for output but good for debug.
    actual_final_answer_calculated_by_steps = None

    # Determine where to introduce an error if create_error is True
    if create_error and N > 1:
        possible_error_rounds = N - 1
        error_round_for_calculation = random.randint(0, possible_error_rounds - 1)
        num_sums_in_error_round = N - 1 - error_round_for_calculation
        error_sum_index_in_round = random.randint(0, num_sums_in_error_round - 1)
    else:
        error_round_for_calculation = -1
        error_sum_index_in_round = -1

    # --- Simulate the correct calculation first to get the true final answer ---
    correct_layers = [list(numbers)]
    temp_correct_layer = list(numbers)
    for i in range(N - 1):
        next_correct_layer = []
        for j in range(len(temp_correct_layer) - 1):
            next_correct_layer.append(temp_correct_layer[j] + temp_correct_layer[j+1])
        if not next_correct_layer: # Should not happen if N > 1
            if N == 1: # N=1, loop for range(N-1) which is range(0) doesn't run.
                break
            else: # Problem for N > 1 if next_correct_layer is empty
                # This indicates an issue, possibly len(temp_correct_layer) < 2 unexpectedly.
                # For safety, if temp_correct_layer has one element, it's the result.
                if len(temp_correct_layer) == 1:
                    break # The last element is already found.
                else: # This is an unexpected state.
                    # Fallback or raise error, for now let's assume this path isn't hit with N>=2
                    pass
        correct_layers.append(next_correct_layer)
        temp_correct_layer = next_correct_layer

    if N == 1:
        true_final_answer = numbers[0]
    elif not correct_layers[-1]: # If the last layer is empty (shouldn't happen for N > 1)
        # This might occur if N was 0 or 1 and logic proceeded unexpectedly.
        # For N>=2, correct_layers[-1] should have one element.
        # If N=0, numbers is empty. If N=1, numbers has 1.
        # If N=1, the loop range(N-1) = range(0) does not run. correct_layers = [numbers].
        # So correct_layers[-1][0] is numbers[0].
        # This 'elif' is more of a safeguard for unexpected states.
        if numbers: # If there were initial numbers
            true_final_answer = numbers[0] # Default to the first if something went wrong
        else: # No numbers, no answer
            true_final_answer = 0 # Or None, or raise error
    else:
        true_final_answer = correct_layers[-1][0]
    # --- End of correct calculation ---

    current_layer_for_steps = list(numbers)
    for i in range(N - 1): # i is the current layer being processed to generate layer i+1
        next_layer_numbers = []
        step_description = f"Step {i+1} (Calculating Layer {i+1} from Layer {i}):\n"
        # layer_calculations_details = []

        if not current_layer_for_steps or len(current_layer_for_steps) < 2:
            if len(current_layer_for_steps) == 1:
                actual_final_answer_calculated_by_steps = current_layer_for_steps[0]
            break

        for j in range(len(current_layer_for_steps) - 1):
            num1 = current_layer_for_steps[j]
            num2 = current_layer_for_steps[j+1]
            correct_sum = num1 + num2
            actual_sum_used = correct_sum

            if create_error and not error_introduced and i == error_round_for_calculation and j == error_sum_index_in_round:
                error_introduced = True
                error_layer_index = i + 1 # 1-indexed layer for label
                # error_step_in_layer = j # For debugging

                sum_str = str(correct_sum)
                if len(sum_str) > 0:
                    digit_to_change_index = random.randint(0, len(sum_str) - 1)
                    original_digit = int(sum_str[digit_to_change_index])

                    if random.choice([True, False]):
                        new_digit = original_digit + 1
                        if new_digit > 9 :
                            new_digit = original_digit -1
                            if new_digit < 0:
                                new_digit = 1
                    else:
                        new_digit = original_digit - 1
                        if new_digit < 0:
                            new_digit = original_digit + 1
                            if new_digit > 9:
                                new_digit = 8

                    if new_digit < 0: new_digit = 0
                    if new_digit > 9: new_digit = 9 if original_digit == 9 else (original_digit + 1) % 10

                    temp_sum_list = list(sum_str)
                    temp_sum_list[digit_to_change_index] = str(new_digit)
                    actual_sum_used = int("".join(temp_sum_list))

                    if actual_sum_used <= 0 and correct_sum > 0 :
                        actual_sum_used = correct_sum + 1
                    if actual_sum_used == correct_sum:
                        actual_sum_used = correct_sum + (1 if random.random() < 0.5 else -1)
                        if actual_sum_used <=0 and correct_sum > 0 : actual_sum_used = correct_sum + 1
                        elif actual_sum_used <=0 and correct_sum <=0 : actual_sum_used = correct_sum -1


                # MODIFICATION: Removed hint about error from step_description
                step_description += f"  {num1} + {num2} = {actual_sum_used}\n"
            else:
                step_description += f"  {num1} + {num2} = {actual_sum_used}\n"

            next_layer_numbers.append(actual_sum_used)
            # layer_calculations_details.append((num1, num2, correct_sum, actual_sum_used))

        # all_layers_calculations.append(layer_calculations_details)
        solution_steps.append(step_description.strip())
        current_layer_for_steps = next_layer_numbers
        if len(current_layer_for_steps) == 1:
            actual_final_answer_calculated_by_steps = current_layer_for_steps[0]

    if actual_final_answer_calculated_by_steps is None and current_layer_for_steps:
        actual_final_answer_calculated_by_steps = current_layer_for_steps[0]
    elif N==1:
        actual_final_answer_calculated_by_steps = numbers[0]
        # true_final_answer is already numbers[0] for N=1

    full_response_str = "\n\n".join(solution_steps)
    if N > 1 :
        full_response_str += f"\n\nFinal Answer: The number at Layer {N-1} is {actual_final_answer_calculated_by_steps}."
    elif N == 1:
        full_response_str += f"\n\nFinal Answer: The number at Layer 0 is {actual_final_answer_calculated_by_steps}."


    # Determine final_answer (true/false) and label
    # final_answer_flag: True if the process described in 'response' is entirely correct.
    #                     False if 'create_error' was True and an error was successfully injected.

    final_answer_flag = True # Assume correct initially
    label_for_csv = -1

    if create_error and error_introduced: # An error was intended AND introduced
        final_answer_flag = False
        label_for_csv = error_layer_index
    elif create_error and not error_introduced and N > 1: # Error intended but NOT introduced (e.g. due to N=1 or bug)
        # This means the process described is actually correct despite the *intent* to make an error.
        # So final_answer_flag remains True, label remains -1.
        # This scenario implies `should_create_error` was true, but the response is actually correct.
        pass # final_answer_flag is True, label_for_csv is -1 (correct)
    # If not create_error, then final_answer_flag is True and label_for_csv is -1 (correct)


    # Add the final result to the response string
    return problem_description, N, original_numbers_str, full_response_str, final_answer_flag, label_for_csv


# --- Parameters ---
NUM_DATA_ROWS = 500         # How many rows of data to generate
MIN_N = 5                   # Minimum number of initial numbers (must be at least 1)
MAX_N = 5                   # Maximum number of initial numbers
PROBABILITY_OF_ERROR = 0.5  # Probability of generating an incorrect solution process (0.0 to 1.0)
OUTPUT_CSV_FILE = "number_triangle_problems.csv" # Changed output file name

# --- Main script ---
if __name__ == "__main__":
    if MIN_N < 1:
        print("Error: MIN_N must be at least 1.")
        exit()
    # For N=1, no calculation errors can be introduced in solution steps.
    # The generate_triangle_problem handles N=1 by not allowing error_introduction.

    data_to_write = []
    for i in range(NUM_DATA_ROWS):
        current_N = random.randint(MIN_N, MAX_N)

        # Determine if an error should be created for this row
        # For N=1, create_error will effectively be false inside the function for error injection part
        should_create_error_intent = random.random() < PROBABILITY_OF_ERROR
        if current_N == 1:
            # No calculation steps for N=1, so no error can be *made* in steps.
            # The intent to create an error for N=1 will result in a correct process.
            # generate_triangle_problem will handle N=1 gracefully regarding error injection.
            actual_create_error_flag_for_function = False
        else:
            actual_create_error_flag_for_function = should_create_error_intent

        # print(f"Generating row {i+1}: N={current_N}, Intent to Create Error={should_create_error_intent}, Actual Create Error Flag for Func={actual_create_error_flag_for_function}")

        try:
            problem, N_val, nums_str, response, final_ans_process_correct_flag, err_label = \
                generate_triangle_problem(current_N, actual_create_error_flag_for_function)

            # The final_ans_process_correct_flag already reflects if the *described process* is correct.
            # If actual_create_error_flag_for_function was True and an error was introduced,
            # final_ans_process_correct_flag will be False and err_label will be set.
            # If actual_create_error_flag_for_function was False, or True but N=1 (so no error introduced),
            # final_ans_process_correct_flag will be True and err_label will be -1.
            # This logic is now self-contained within generate_triangle_problem.

            data_to_write.append({
                "problem": problem,
                "N": N_val,
                "number": nums_str,
                "response": response,
                "final_answer": final_ans_process_correct_flag,
                "label": err_label
            })
        except ValueError as e:
            print(f"Skipping row generation for N={current_N} due to error: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred during row {i+1} (N={current_N}) generation: {e}")
            # import traceback
            # traceback.print_exc() # For more detailed debugging if needed
            continue


    # Write to CSV
    if data_to_write:
        fieldnames = ["problem", "N", "number", "response", "final_answer", "label"]
        with open(OUTPUT_CSV_FILE, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_to_write)
        print(f"\nSuccessfully generated {len(data_to_write)} rows of data in '{OUTPUT_CSV_FILE}'")
    else:
        print("No data was generated.")
