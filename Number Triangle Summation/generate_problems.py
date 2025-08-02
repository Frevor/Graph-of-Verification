# generate_problems.py

import random
import csv
import os

def generate_triangle_problem(N, create_error=False):
    """
    Generates a single number triangle summation problem and its solution process.
    """
    if N < 1:
        raise ValueError("N must be at least 1.")

    numbers = [random.randint(1000, 9999) for _ in range(N)]
    original_numbers_str = ", ".join(map(str, numbers))

    if N > 1:
        problem_description = f"Given {N} numbers at layer 0: {original_numbers_str}. " \
                              f"Calculate the sum of adjacent numbers to form the next layer, " \
                              f"repeating this process until only one number remains at layer {N-1}. " \
                              f"What is the final number?"
    else:
        problem_description = f"Given 1 number at layer 0: {original_numbers_str}. " \
                              f"What is the final number?"

    solution_steps = []
    current_layer_numbers = list(numbers)
    
    error_introduced = False
    error_layer_index = -1 
    actual_final_answer_calculated_by_steps = None

    if create_error and N > 1:
        possible_error_rounds = N - 1
        error_round_for_calculation = random.randint(0, possible_error_rounds - 1)
        num_sums_in_error_round = N - 1 - error_round_for_calculation
        error_sum_index_in_round = random.randint(0, num_sums_in_error_round - 1)
    else:
        error_round_for_calculation = -1
        error_sum_index_in_round = -1

    current_layer_for_steps = list(numbers)
    for i in range(N - 1):
        next_layer_numbers = []
        step_description = f"Step {i+1} (Calculating Layer {i+1} from Layer {i}):\n"
        
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
                error_layer_index = i + 1
                
                sum_str = str(correct_sum)
                if len(sum_str) > 0:
                    digit_to_change_index = random.randint(0, len(sum_str) - 1)
                    original_digit = int(sum_str[digit_to_change_index])
                    new_digit = (original_digit + random.randint(1, 9)) % 10
                    if new_digit == original_digit: new_digit = (new_digit + 1) % 10

                    temp_sum_list = list(sum_str)
                    temp_sum_list[digit_to_change_index] = str(new_digit)
                    actual_sum_used = int("".join(temp_sum_list))

                    if actual_sum_used == correct_sum:
                        actual_sum_used += 1

                step_description += f"  {num1} + {num2} = {actual_sum_used}\n"
            else:
                step_description += f"  {num1} + {num2} = {actual_sum_used}\n"

            next_layer_numbers.append(actual_sum_used)
        
        solution_steps.append(step_description.strip())
        current_layer_for_steps = next_layer_numbers
        if len(current_layer_for_steps) == 1:
            actual_final_answer_calculated_by_steps = current_layer_for_steps[0]

    if N == 1:
        actual_final_answer_calculated_by_steps = numbers[0]
        full_response_str = f"Final Answer: The number at Layer 0 is {actual_final_answer_calculated_by_steps}."
    else:
        full_response_str = "\n\n".join(solution_steps)
        if actual_final_answer_calculated_by_steps is not None:
            full_response_str += f"\n\nFinal Answer: The number at Layer {N-1} is {actual_final_answer_calculated_by_steps}."

    final_answer_flag = not (create_error and error_introduced)
    label_for_csv = error_layer_index if not final_answer_flag else -1
    
    if N == 1:
        final_answer_flag = True
        label_for_csv = -1

    return problem_description, N, original_numbers_str, full_response_str, final_answer_flag, label_for_csv

# --- Parameters ---
NUM_CORRECT = 250   # Number of correct samples
NUM_INCORRECT = 250 # Number of incorrect samples
N_VALUES = [2, 4, 6, 8] # You can change this

# --- Main script ---
if __name__ == "__main__":
    for n_val in N_VALUES:
        print(f"--- Generating dataset for N = {n_val} ---")
        
        output_csv_file = f"number_triangle_problems_N{n_val}.csv"
        
        all_data_for_n = []
        
        # --- Generate correct samples ---
        print(f"Generating {NUM_CORRECT} correct samples...")
        for i in range(NUM_CORRECT):
            problem, N_val, nums_str, response, final_ans_flag, err_label = \
                generate_triangle_problem(n_val, create_error=False)
            all_data_for_n.append({
                "id": f"N{n_val}-correct-{i}",
                "problem": problem, "N": N_val, "number": nums_str, "response": response,
                "final_answer": final_ans_flag, "label": err_label
            })

        # --- Generate incorrect samples ---
        print(f"Generating {NUM_INCORRECT} incorrect samples...")
        for i in range(NUM_INCORRECT):
            problem, N_val, nums_str, response, final_ans_flag, err_label = \
                generate_triangle_problem(n_val, create_error=True)
            all_data_for_n.append({
                "id": f"N{n_val}-incorrect-{i}",
                "problem": problem, "N": N_val, "number": nums_str, "response": response,
                "final_answer": final_ans_flag, "label": err_label
            })
            
        # --- Shuffle the combined data ---
        random.shuffle(all_data_for_n)
        print("Shuffling data...")

        # --- Write to CSV ---
        if all_data_for_n:
            fieldnames = ["id", "problem", "N", "number", "response", "final_answer", "label"]
            with open(output_csv_file, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_data_for_n)
            print(f"Successfully generated {len(all_data_for_n)} rows ({NUM_CORRECT} correct, {NUM_INCORRECT} incorrect) in '{output_csv_file}'\n")
        else:
            print(f"No data was generated for N = {n_val}.\n")
