import requests
import time
import json
import re
import os
import glob
from pathlib import Path

# Requirements: requests
# To install: pip install requests

API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-chat-v3-0324:free"

DATASET_DIR = "dataset_spec-to-rtl"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

PROMPT_TEMPLATE = '''Given the following Verilog problem prompt and reference implementation, generate:

1. A SLIGHTLY different but functionally correct implementation. Make MINIMAL changes (such as code structure, order, or style), but DO NOT change any variable or signal names. DO NOT add or change any comments. DO NOT change the core logic or approach significantly.

2. An implementation with a SUBTLE bug. Introduce only a SMALL error like:
   - Wrong bit width or indexing
   - Missing/incorrect condition in if statement
   - Wrong operator or signal assignment
   - Minor timing issue
   DO NOT make major structural changes. DO NOT add or change any comments.

Also analyze the problem and provide:
3. A difficulty level based on these criteria:
   - EASY: Simple combinational logic (basic gates, simple mux/demux, basic assignments)
   - MEDIUM: More complex combinational logic, basic sequential logic (simple counters, basic FSMs, registers)
   - HARD: Complex sequential logic, advanced FSMs, complex algorithms, timing-critical designs
4. A category from: combinational_logic, sequential_logic, finite_state_machine, arithmetic, memory_storage, circuit_analysis

Original Prompt:
{PROMPT}

Reference Implementation:
{REFERENCE}

Return your answer in the following format:

Slightly Different Implementation:
<verilog code>

Buggy Implementation:
<verilog code>

Difficulty: <easy/medium/hard>

Category: <category>
'''

def call_openrouter(prompt):
    data = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(OPENROUTER_URL, headers=HEADERS, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def parse_response(response):
    diff = bug = difficulty = category = ""
    if response:
        diff_match = re.search(r"Slightly Different Implementation:\s*([\s\S]*?)(?:Buggy Implementation:|$)", response)
        bug_match = re.search(r"Buggy Implementation:\s*([\s\S]*?)(?:Difficulty:|$)", response)
        difficulty_match = re.search(r"Difficulty:\s*(\w+)", response)
        category_match = re.search(r"Category:\s*(\w+)", response)
        
        if diff_match:
            diff = diff_match.group(1).strip()
        if bug_match:
            bug = bug_match.group(1).strip()
        if difficulty_match:
            difficulty = difficulty_match.group(1).strip().lower()
        if category_match:
            category = category_match.group(1).strip().lower()
    
    return diff, bug, difficulty, category

def clean_verilog_code(text):
    if not text:
        return ""
    # Remove markdown code blocks
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = re.sub(r"```", "", text)
    # Remove bold/italic markers
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"__", "", text)
    # Remove leading blank lines
    text = re.sub(r"^\s*\n", "", text)
    # Remove all // comments
    text = re.sub(r"//.*", "", text)
    # Remove all /* ... */ comments (including multiline)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Remove trailing whitespace on each line and remove empty lines
    text = "\n".join([line.rstrip() for line in text.splitlines() if line.strip() != ""])
    text = text.strip()
    return text

def extract_problem_info(filename):
    """Extract problem number and name from filename"""
    match = re.match(r"Prob(\d+)_(.+)_prompt\.txt", filename)
    if match:
        return int(match.group(1)), match.group(2)
    return None, None

# Removed heuristic difficulty assignment - now using LLM exclusively

def main():
    # Find all prompt files
    prompt_files = glob.glob(os.path.join(DATASET_DIR, "*_prompt.txt"))
    prompt_files.sort()
    
    print(f"Found {len(prompt_files)} problems to process")
    
    results = []
    
    # Open the JSON file and write the opening bracket
    with open("verilog_eval_results.json", "w") as f:
        f.write('[\n')
    first_result = True
    
    for idx, prompt_file in enumerate(prompt_files, 1):
        filename = os.path.basename(prompt_file)
        prob_num, prob_name = extract_problem_info(filename)
        
        if prob_num is None:
            print(f"Skipping {filename} - couldn't parse problem info")
            continue
        
        # Find corresponding reference file
        ref_file = prompt_file.replace("_prompt.txt", "_ref.sv")
        if not os.path.exists(ref_file):
            print(f"Skipping {filename} - no reference file found")
            continue
        
        print(f"[{idx}/{len(prompt_files)}] Processing Prob{prob_num:03d}_{prob_name}")
        
        try:
            # Read prompt and reference implementation
            with open(prompt_file, 'r') as f:
                prompt_text = f.read().strip()
            with open(ref_file, 'r') as f:
                ref_code = f.read().strip()
            
            # Generate LLM prompt
            llm_prompt = PROMPT_TEMPLATE.format(PROMPT=prompt_text, REFERENCE=ref_code)
            
            # Call LLM
            response = call_openrouter(llm_prompt)
            if response is None:
                print(f"  ERROR: Failed to get LLM response")
                continue
            
            # Parse response
            diff_impl, bug_impl, difficulty, category = parse_response(response)
            
            # Clean up code
            diff_impl = clean_verilog_code(diff_impl)
            bug_impl = clean_verilog_code(bug_impl)
            
            # Validate LLM-assigned difficulty
            if not difficulty or difficulty not in ['easy', 'medium', 'hard']:
                print(f"  WARNING: Invalid difficulty '{difficulty}' from LLM, skipping this problem")
                continue
            # Validate LLM-assigned category
            if not category:
                print(f"  WARNING: No category from LLM, skipping this problem")
                continue
            print(f"  LLM assigned difficulty: {difficulty}, category: {category}")
            
            # Create result entry
            result = {
                "problem_number": prob_num,
                "problem_name": prob_name,
                "original_prompt": prompt_text,
                "reference_implementation": ref_code,
                "slightly_different_implementation": diff_impl,
                "buggy_implementation": bug_impl,
                "difficulty": difficulty,
                "category": category
            }
            
            # Write result to JSON file immediately
            with open("verilog_eval_results.json", "a") as f:
                if not first_result:
                    f.write(',\n')
                json.dump(result, f, indent=2)
                first_result = False
            
            results.append(result)
            
            # Rate limiting
            time.sleep(2)
            
        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")
            continue
    
    # Close the JSON array
    with open("verilog_eval_results.json", "a") as f:
        f.write('\n]')
    
    print(f"\nProcessed {len(results)} problems successfully")
    print("Results saved to verilog_eval_results.json")
    
    # Print summary statistics
    difficulties = {}
    categories = {}
    for result in results:
        diff = result['difficulty']
        cat = result['category']
        difficulties[diff] = difficulties.get(diff, 0) + 1
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nDifficulty Distribution:")
    for diff, count in sorted(difficulties.items()):
        print(f"  {diff}: {count}")
    
    print("\nCategory Distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

if __name__ == "__main__":
    main()
