import json
import os
import subprocess
import shutil
import re
import requests
from typing import Optional, Tuple

# Paths
JSON_PATH = "verilog_eval_codes.json"
DATASET_DIR = "dataset_spec-to-rtl"
TEMP_DIR = "temp_code"

# OpenRouter API configuration
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "anthropic/claude-sonnet-4"


def rename_module(verilog_code, old_name, new_name):
    """Replace the module name in the Verilog code."""
    code = re.sub(rf'\bmodule\s+{old_name}\b', f'module {new_name}', verilog_code)
    return code


def write_both_modules(temp_dir, ref_code, top_code):
    """Write both RefModule.sv and TopModule.sv files."""
    ref_sv = os.path.join(temp_dir, "RefModule.sv")
    with open(ref_sv, "w") as f:
        f.write(ref_code)
    
    top_sv = os.path.join(temp_dir, "TopModule.sv")
    with open(top_sv, "w") as f:
        f.write(top_code)
    
    return ref_sv, top_sv


def parse_results(output):
    """Parse simulation output to extract mismatches and total samples."""
    mismatch_pattern = r'Mismatches: (\d+) in (\d+) samples'
    match = re.search(mismatch_pattern, output)
    
    if match:
        mismatches = int(match.group(1))
        total_samples = int(match.group(2))
        success_rate = ((total_samples - mismatches) / total_samples) * 100 if total_samples > 0 else 0
        return total_samples, mismatches, success_rate
    else:
        if "error:" in output.lower():
            return None, None, 0
        return 0, 0, 0


def run_iverilog(testbench, module_files):
    """Compile and run the testbench with the given module files."""
    sim_output = "simv"
    
    try:
        original_cwd = os.getcwd()
        os.chdir(TEMP_DIR)
        
        module_names = [os.path.basename(f) for f in module_files]
        testbench_name = os.path.basename(testbench)
        
        compile_result = subprocess.run(
            ["iverilog", "-g2012", "-o", sim_output] + module_names + [testbench_name],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        if compile_result.returncode != 0:
            os.chdir(original_cwd)
            return (compile_result.stdout or b"").decode() + (compile_result.stderr or b"").decode()
        
        result = subprocess.run(
            ["vvp", sim_output],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        os.chdir(original_cwd)
        return result.stdout.decode() + result.stderr.decode()
        
    except FileNotFoundError as e:
        if 'original_cwd' in locals():
            os.chdir(original_cwd)
        return f"Error: {e}. Make sure iverilog and vvp are installed and in PATH."
    except Exception as e:
        if 'original_cwd' in locals():
            os.chdir(original_cwd)
        return f"Unexpected error: {e}"


def call_openrouter_api(prompt: str) -> Optional[str]:
    """Call OpenRouter API with GPT-4o to generate Verilog code."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    data = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a Verilog RTL designer. Generate correct, synthesizable Verilog code. Only return the module code, no explanations."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            print(f"No response from API: {result}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse API response: {e}")
        return None


def extract_verilog_code(response: str) -> str:
    """Extract Verilog code from API response."""
    # Try to find code between ```verilog and ``` or ```systemverilog and ```
    patterns = [
        r'```(?:verilog|systemverilog)?\s*(.*?)```',
        r'```\s*(module.*?endmodule)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no code blocks found, check if the entire response looks like Verilog
    if "module" in response and "endmodule" in response:
        return response.strip()
    
    return response.strip()


def generate_llm_implementation(prob: dict) -> Optional[str]:
    """Generate a new implementation using LLM."""
    prompt = f"""
{prob['original_prompt']}

The reference implementation is:
```verilog
{prob['reference_implementation']}
```

The previous slightly different implementation failed. Please generate a new, correct implementation that:
1. Follows the same interface as the reference
2. Implements the same functionality 
3. Is syntactically correct Verilog
4. Uses module name TopModule (not RefModule)

Generate only the Verilog module code:
"""
    
    response = call_openrouter_api(prompt)
    if response:
        return extract_verilog_code(response)
    return None


def identify_failed_problems() -> list:
    """Identify problems that had failures from result.txt or by running evaluation."""
    failed_problems = []
    
    # Try to read from result.txt first
    if os.path.exists("result.txt"):
        with open("result.txt", "r") as f:
            for line in f:
                if "COMPILE ERROR" in line or (
                    "%" in line and "100.0" not in line and 
                    any(char.isdigit() for char in line.split("%")[0].split()[-1])
                ):
                    # Extract problem name (first column)
                    parts = line.strip().split()
                    if parts:
                        problem_name = parts[0]
                        if problem_name not in failed_problems and problem_name not in ["Problem", "-"]:
                            failed_problems.append(problem_name)
    
    return failed_problems


def update_json_file(problems: list, updated_problems: dict):
    """Update the JSON file with new implementations."""
    if not updated_problems:
        return
    
    print(f"\nUpdating {len(updated_problems)} problems in {JSON_PATH}...")
    
    # Update the problems list
    for prob in problems:
        if prob["problem_name"] in updated_problems:
            prob["slightly_different_implementation"] = updated_problems[prob["problem_name"]]
    
    # Write back to file
    with open(JSON_PATH, 'w') as f:
        json.dump(problems, f, indent=2)
    
    print(f"Successfully updated {JSON_PATH}")


def main():
    # Create temp directory
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    with open(JSON_PATH) as f:
        problems = json.load(f)

    # Identify failed problems
    failed_problem_names = identify_failed_problems()
    print(f"Found {len(failed_problem_names)} failed problems: {failed_problem_names}")
    
    # Filter problems to only failed ones
    failed_problems = [prob for prob in problems if prob["problem_name"] in failed_problem_names]
    
    if not failed_problems:
        print("No failed problems found!")
        return

    # Print header
    print(f"\n{'Problem Name':<25} {'Type':<20} {'Total Samples':<15} {'Mismatches':<12} {'Success Rate':<12}")
    print("-" * 90)

    updated_problems = {}  # Track problems that should be updated in JSON

    for prob in failed_problems:
        prob_num = prob["problem_number"]
        prob_name = prob["problem_name"]
        testbench_file = os.path.join(DATASET_DIR, f"Prob{prob_num:03d}_{prob_name}_test.sv")

        if not os.path.exists(testbench_file):
            print(f"{prob_name:<25} {'N/A':<20} {'Testbench not found':<15}")
            continue

        print(f"\nGenerating new implementation for {prob_name}...")
        
        # Generate new implementation using LLM
        new_implementation = generate_llm_implementation(prob)
        if not new_implementation:
            print(f"{prob_name:<25} {'LLM Generated':<20} {'GENERATION FAILED':<15}")
            continue

        # Copy testbench
        tb_sv = os.path.join(TEMP_DIR, "testbench.sv")
        shutil.copy(testbench_file, tb_sv)

        ref_code = prob["reference_implementation"]

        # Test the reference implementation first
        ref_as_top = rename_module(ref_code, "RefModule", "TopModule")
        ref_sv, top_sv = write_both_modules(TEMP_DIR, ref_code, ref_as_top)
        ref_result = run_iverilog(tb_sv, [ref_sv, top_sv])
        ref_samples, ref_mismatches, ref_success = parse_results(ref_result)
        
        if ref_samples is not None:
            print(f"{prob_name:<25} {'Reference':<20} {ref_samples:<15} {ref_mismatches:<12} {ref_success:<12.1f}%")
        else:
            print(f"{prob_name:<25} {'Reference':<20} {'COMPILE ERROR':<15}")

        # Test the new LLM-generated implementation
        if "module RefModule" in new_implementation:
            new_as_top = rename_module(new_implementation, "RefModule", "TopModule")
        else:
            new_as_top = new_implementation
        
        ref_sv, top_sv = write_both_modules(TEMP_DIR, ref_code, new_as_top)
        
        new_result = run_iverilog(tb_sv, [ref_sv, top_sv])
        new_samples, new_mismatches, new_success = parse_results(new_result)

        # Print results for new implementation
        if new_samples is not None:
            print(f"{prob_name:<25} {'LLM Generated':<20} {new_samples:<15} {new_mismatches:<12} {new_success:<12.1f}%")
            
            # Check if both reference and new implementation are 100%
            if (ref_success == 100.0 and new_success == 100.0 and 
                ref_samples is not None and new_samples is not None):
                updated_problems[prob_name] = new_as_top
                print(f"{prob_name:<25} {'STATUS':<20} {'WILL UPDATE JSON':<15}")
        else:
            print(f"{prob_name:<25} {'LLM Generated':<20} {'COMPILE ERROR':<15}")

    # Update the JSON file if we have successful replacements
    if updated_problems:
        update_json_file(problems, updated_problems)
        print(f"\nSummary: Updated {len(updated_problems)} problems in {JSON_PATH}")
        for prob_name in updated_problems.keys():
            print(f"  - {prob_name}")
    else:
        print("\nNo problems met the criteria for JSON update (both ref and generated must be 100%)")

    # Clean up
    shutil.rmtree(TEMP_DIR)


if __name__ == "__main__":
    main() 