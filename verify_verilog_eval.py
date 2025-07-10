import json
import os
import subprocess
import shutil
import re

# Paths
JSON_PATH = "verilog_eval_codes.json"
DATASET_DIR = "dataset_spec-to-rtl"
TEMP_DIR = "temp_code"


def rename_module(verilog_code, old_name, new_name):
    """Replace the module name in the Verilog code."""
    # Replace only the module declaration
    code = re.sub(rf'\bmodule\s+{old_name}\b', f'module {new_name}', verilog_code)
    return code


def write_both_modules(temp_dir, ref_code, top_code):
    """Write both RefModule.sv and TopModule.sv files."""
    # Write RefModule.sv (reference, as-is)
    ref_sv = os.path.join(temp_dir, "RefModule.sv")
    with open(ref_sv, "w") as f:
        f.write(ref_code)
    
    # Write TopModule.sv 
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
        # Check for compilation or other errors
        if "error:" in output.lower():
            return None, None, 0  # Failed compilation
        return 0, 0, 0  # No samples found


def run_iverilog(testbench, module_files):
    """Compile and run the testbench with the given module files."""
    # Use relative paths within the temp directory
    sim_output = "simv"
    
    try:
        # Change to temp directory for compilation and execution
        original_cwd = os.getcwd()
        os.chdir(TEMP_DIR)
        
        # Get just the filenames since we're in the temp directory
        module_names = [os.path.basename(f) for f in module_files]
        testbench_name = os.path.basename(testbench)
        
        # Compile all module files and testbench together
        compile_result = subprocess.run(
            ["iverilog", "-g2012", "-o", sim_output] + module_names + [testbench_name],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Check if compilation was successful
        if compile_result.returncode != 0:
            os.chdir(original_cwd)
            return (compile_result.stdout or b"").decode() + (compile_result.stderr or b"").decode()
        
        # Run using vvp from within the temp directory
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


def main():
    # Create temp directory
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    with open(JSON_PATH) as f:
        problems = json.load(f)

    # Print header
    print(f"{'Problem Name':<25} {'Type':<20} {'Total Samples':<15} {'Mismatches':<12} {'Success Rate':<12}")
    print("-" * 90)

    for prob in problems:
        prob_num = prob["problem_number"]
        prob_name = prob["problem_name"]
        testbench_file = os.path.join(DATASET_DIR, f"Prob{prob_num:03d}_{prob_name}_test.sv")

        if not os.path.exists(testbench_file):
            print(f"{prob_name:<25} {'N/A':<20} {'Testbench not found':<15}")
            continue

        # Copy testbench
        tb_sv = os.path.join(TEMP_DIR, "testbench.sv")
        shutil.copy(testbench_file, tb_sv)

        ref_code = prob["reference_implementation"]
        diff_code = prob["slightly_different_implementation"]

        # For reference run: both modules are the reference (TopModule is renamed RefModule)
        ref_as_top = rename_module(ref_code, "RefModule", "TopModule")
        ref_sv, top_sv = write_both_modules(TEMP_DIR, ref_code, ref_as_top)
        
        ref_result = run_iverilog(tb_sv, [ref_sv, top_sv])
        ref_samples, ref_mismatches, ref_success = parse_results(ref_result)

        # For slightly different run: RefModule is reference, TopModule is slightly different
        # Ensure the slightly different implementation is renamed to TopModule if needed
        if "module RefModule" in diff_code:
            diff_as_top = rename_module(diff_code, "RefModule", "TopModule")
        else:
            diff_as_top = diff_code
        
        ref_sv, top_sv = write_both_modules(TEMP_DIR, ref_code, diff_as_top)
        
        diff_result = run_iverilog(tb_sv, [ref_sv, top_sv])
        diff_samples, diff_mismatches, diff_success = parse_results(diff_result)

        # Print results
        if ref_samples is not None:
            print(f"{prob_name:<25} {'Reference':<20} {ref_samples:<15} {ref_mismatches:<12} {ref_success:<12.1f}%")
        else:
            print(f"{prob_name:<25} {'Reference':<20} {'COMPILE ERROR':<15}")
            
        if diff_samples is not None:
            print(f"{prob_name:<25} {'Slightly Different':<20} {diff_samples:<15} {diff_mismatches:<12} {diff_success:<12.1f}%")
        else:
            print(f"{prob_name:<25} {'Slightly Different':<20} {'COMPILE ERROR':<15}")

    # Clean up
    shutil.rmtree(TEMP_DIR)


if __name__ == "__main__":
    main() 