import json
import os
import subprocess
import re
import tempfile
from tqdm import tqdm

# --- Configuration ---
INPUT_DIR = "cleaned_verilog_results"
OUTPUT_DIR = "evaluation_results"
DATASET_DIR = "../dataset_spec-to-rtl" # Relative path from where this script is run

def run_iverilog_test(verilog_code, test_file_path, problem_name):
    """
    Test generated Verilog code against the test bench.
    Returns: (bool: success, str: message)
    """
    if not verilog_code or 'module' not in verilog_code:
        return False, "No valid Verilog code provided"

    if not os.path.exists(test_file_path):
        return False, f"Test file not found: {test_file_path}"

    with tempfile.TemporaryDirectory() as temp_dir:
        # Write generated code to file
        code_file = os.path.join(temp_dir, f"{problem_name}.sv")
        with open(code_file, 'w') as f:
            f.write(verilog_code)

        # Copy test file
        test_copy = os.path.join(temp_dir, f"{problem_name}_test.sv")
        with open(test_file_path, 'r') as src, open(test_copy, 'w') as dst:
            dst.write(src.read())

        try:
            # Compile with iverilog
            compile_cmd = ['iverilog', '-g2012', '-o', 'sim_v', test_copy, code_file]
            compile_result = subprocess.run(
                compile_cmd, cwd=temp_dir, capture_output=True, text=True, timeout=30
            )
            if compile_result.returncode != 0:
                return False, f"Compilation failed: {compile_result.stderr}"

            # Run simulation
            sim_cmd = ['vvp', 'sim_v']
            sim_result = subprocess.run(
                sim_cmd, cwd=temp_dir, capture_output=True, text=True, timeout=10
            )
            
            output = sim_result.stdout + sim_result.stderr
            if "Mismatches: 0" in output:
                return True, "Test passed"
            else:
                return False, "Test failed: Mismatches found"

        except subprocess.TimeoutExpired:
            return False, "Test timed out"
        except Exception as e:
            return False, f"Execution error: {str(e)}"

def main():
    print("Starting Verilog evaluation process...")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('_cleaned.json')]
    print(f"Found {len(json_files)} cleaned result files to evaluate in '{INPUT_DIR}'.")
    
    for filename in tqdm(json_files, desc="Evaluating files"):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename.replace('_cleaned.json', '_results.json'))

        if os.path.exists(output_path):
            continue
            
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        evaluation_results = []
        for problem in data:
            problem_name = problem['problem_name']
            
            # Find the testbench file
            test_file_path = os.path.join(DATASET_DIR, f"{problem_name}_test.sv")
            
            # Test without steering
            success_without, msg_without = run_iverilog_test(
                problem['without_steering_implementation'], test_file_path, problem_name
            )
            
            # Test with steering
            success_with, msg_with = run_iverilog_test(
                problem['with_steering_implementation'], test_file_path, problem_name
            )
            
            evaluation_results.append({
                'problem_name': problem_name,
                'pass_without_steering': success_without,
                'pass_with_steering': success_with,
                'message_without_steering': msg_without,
                'message_with_steering': msg_with,
            })
            
        # Save the evaluation results
        with open(output_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)

    print(f"\nEvaluation complete. Results are in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main() 