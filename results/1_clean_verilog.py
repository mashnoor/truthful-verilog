import json
import os
import re
import requests
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# --- Configuration ---
INPUT_DIR = "verilog_iti_results"
OUTPUT_DIR = "cleaned_verilog_results"
# OPENROUTER_API_KEY = "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"
# API_URL = "https://openrouter.ai/api/v1/chat/completions"
# MODEL = "deepseek/deepseek-chat-v3-0324:free" # Using a fast model for a simple extraction task

OLLAMA_API_URL = "http://evc30:11434/api/chat"
MODEL = "qwen2.5-coder:7b"

CLEANUP_PROMPT = """
Please extract the complete SystemVerilog module from the following text. 
The module should start with 'module' and end with 'endmodule'.
Output ONLY the code, with no explanations, comments, or markdown formatting.
If the text does not contain a valid module, output only the word "ERROR".

Text to clean:
---
{code}
---
"""

def clean_verilog_code(code_text):
    """Uses a local Ollama model to extract clean Verilog code from text."""
    if "module" not in code_text.lower():
        return "" # Skip if it's clearly not code

    prompt = CLEANUP_PROMPT.format(code=code_text)
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False, # We want the full response at once
        "options": {
            "temperature": 0.0
        }
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=data, timeout=30)
        response.raise_for_status()
        
        # Ollama's response structure is slightly different
        cleaned_code = response.json()["message"]["content"].strip()
        
        if "ERROR" in cleaned_code or "module" not in cleaned_code:
            return "" # LLM failed to find code
            
        # Final sanity check with regex
        match = re.search(r'module\s+.*endmodule', cleaned_code, re.DOTALL | re.IGNORECASE)
        return match.group(0) if match else ""

    except requests.exceptions.RequestException as e:
        print(f"  API Error: {e}")
        return ""

def process_single_file(filename):
    """Process a single JSON file for cleaning."""
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename.replace(".json", "_cleaned.json"))
    
    # Skip if already processed
    if os.path.exists(output_path):
        return f"Skipped {filename} (already exists)"
        
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)

        with open(output_path, 'w') as f:
            f.write('[\n')
            
            for i, problem in enumerate(data):
                cleaned_problem = problem.copy()
                
                # Clean 'without_steering' implementation
                cleaned_problem['without_steering_implementation'] = clean_verilog_code(
                    problem['without_steering_implementation']
                )
                
                # Clean 'with_steering' implementation
                cleaned_problem['with_steering_implementation'] = clean_verilog_code(
                    problem['with_steering_implementation']
                )
                
                # Ensure the module names match the original implementation
                # Extract the module name from the original implementation
                original_module_match = re.search(r'module\s+(\w+)', problem['original_implementation'])
                if original_module_match:
                    original_module_name = original_module_match.group(1)
                    
                    # Rename the generated implementations to match the original module name
                    if 'module' in cleaned_problem['without_steering_implementation']:
                         cleaned_problem['without_steering_implementation'] = re.sub(
                             r'module\s+\w+', f'module {original_module_name}', cleaned_problem['without_steering_implementation'], count=1
                         )
                    if 'module' in cleaned_problem['with_steering_implementation']:
                         cleaned_problem['with_steering_implementation'] = re.sub(
                             r'module\s+\w+', f'module {original_module_name}', cleaned_problem['with_steering_implementation'], count=1
                         )

                # Write the cleaned problem to the file
                if i > 0:
                    f.write(',\n')
                json.dump(cleaned_problem, f, indent=2)
            
            f.write('\n]')
        
        return f"Processed {filename}"
    except Exception as e:
        return f"Error processing {filename}: {str(e)}"

def main():
    print("Starting Verilog code cleanup process...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Find all result JSON files
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    print(f"Found {len(json_files)} result files to process in '{INPUT_DIR}'.")
    
    # Use multiprocessing to process files in parallel
    num_processes = min(mp.cpu_count(), len(json_files))
    print(f"Using {num_processes} processes for parallel processing.")
    
    # Process files in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, json_files),
            total=len(json_files),
            desc="Cleaning files"
        ))
    
    # Print results
    for result in results:
        print(result)

    print(f"\nCleanup complete. Cleaned files are in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main() 