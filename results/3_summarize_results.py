import json
import os
import re
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
INPUT_DIR = "evaluation_results"
OUTPUT_CSV = "final_summary_results.csv"

def parse_filename(filename):
    """Extracts K and alpha from the filename."""
    k_match = re.search(r'K(\d+)', filename)
    alpha_match = re.search(r'alpha([\d.]+)', filename)
    
    K = int(k_match.group(1)) if k_match else -1
    alpha = float(alpha_match.group(1)) if alpha_match else -1.0
    
    return K, alpha

def main():
    print("Summarizing all evaluation results...")
    
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        return
        
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('_results.json')]
    print(f"Found {len(json_files)} result files to summarize in '{INPUT_DIR}'.")
    
    all_results = []
    
    for filename in tqdm(json_files, desc="Summarizing files"):
        input_path = os.path.join(INPUT_DIR, filename)
        
        with open(input_path, 'r') as f:
            data = json.load(f)
            
        K, alpha = parse_filename(filename)
        total_problems = len(data)
        
        # --- Without Steering ---
        success_without = sum(1 for r in data if r.get('pass_without_steering'))
        failed_without = total_problems - success_without
        percent_without = (success_without / total_problems * 100) if total_problems > 0 else 0
        
        all_results.append({
            'Steering': 'Without',
            'K': K,
            'Alpha': alpha,
            'Total Problems': total_problems,
            'Success': success_without,
            'Failed': failed_without,
            'Success Rate (%)': f"{percent_without:.2f}"
        })
        
        # --- With Steering ---
        success_with = sum(1 for r in data if r.get('pass_with_steering'))
        failed_with = total_problems - success_with
        percent_with = (success_with / total_problems * 100) if total_problems > 0 else 0
        
        all_results.append({
            'Steering': 'With',
            'K': K,
            'Alpha': alpha,
            'Total Problems': total_problems,
            'Success': success_with,
            'Failed': failed_with,
            'Success Rate (%)': f"{percent_with:.2f}"
        })
        
    # Create and save the final CSV
    if not all_results:
        print("No results to summarize.")
        return
        
    df = pd.DataFrame(all_results)
    df = df.sort_values(by=['K', 'Alpha', 'Steering'], ascending=[True, True, False])
    
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nSummary complete! Results saved to '{OUTPUT_CSV}'.")
    print("\nTop 5 results:")
    print(df.sort_values(by='Success Rate (%)', ascending=False).head(5))

if __name__ == "__main__":
    main() 