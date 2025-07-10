import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Configuration
JSON_PATH = "verilog_eval_codes_verified.json"
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

class SimpleVerilogITI:
    """
    Simple Inference-Time Intervention for Verilog Code Generation
    Following the core ITI paper methodology
    """
    
    def __init__(self):
        print(f"Loading {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model dimensions
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        
        print(f"Model: {self.num_layers} layers, {self.num_heads} heads, {self.head_dim} head_dim")
    
    def get_head_activations(self, text):
        """
        Extract attention head activations at the last token
        Returns: [num_layers, num_heads, head_dim]
        """
        # Hook storage
        head_outputs = {}
        hooks = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                # input[0]: [batch, seq_len, hidden_size] - concatenated head outputs
                concatenated = input[0]
                batch_size, seq_len, hidden_size = concatenated.shape
                # Reshape to separate heads: [batch, seq_len, num_heads, head_dim]
                heads = concatenated.view(batch_size, seq_len, self.num_heads, self.head_dim)
                head_outputs[layer_idx] = heads.detach().cpu()
            return hook
        
        # Register hooks on attention output projections
        for layer_idx in range(self.num_layers):
            hook = self.model.model.layers[layer_idx].self_attn.o_proj.register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)
        
        try:
            # Forward pass
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                self.model(**inputs)
            
            # Extract last token activations
            last_pos = inputs['attention_mask'].sum() - 1
            activations = []
            
            for layer_idx in range(self.num_layers):
                # Shape: [num_heads, head_dim]
                layer_acts = head_outputs[layer_idx][0, last_pos, :, :].numpy()
                activations.append(layer_acts)
            
            # Return: [num_layers, num_heads, head_dim]
            return np.array(activations)
            
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
    
    def collect_activations(self, problems):
        """
        Collect activations from ALL correct/buggy implementation pairs
        """
        print(f"Collecting activations from ALL {len(problems)} problems...")
        
        all_activations = []
        all_labels = []
        problem_info = []  # Store problem information
        successful_pairs = 0
        
        for i, problem in enumerate(problems):
            if i % 25 == 0:
                print(f"Processing {i+1}/{len(problems)}...")
            
            ref_impl = problem.get('reference_implementation', '')
            buggy_impl = problem.get('buggy_implementation', '')
            original_prompt = problem.get('original_prompt', '')
            problem_name = problem.get('problem_name', f'prob_{i}')
            
            if not ref_impl or not buggy_impl:
                continue
            
            # Create prompts with original problem description
            prompt_text = f"Problem: {original_prompt}\n\nImplementation:\n"
            correct_prompt = prompt_text + ref_impl
            buggy_prompt = prompt_text + buggy_impl
            
            try:
                # Get activations
                correct_acts = self.get_head_activations(correct_prompt)
                buggy_acts = self.get_head_activations(buggy_prompt)
                
                all_activations.extend([correct_acts, buggy_acts])
                all_labels.extend([1, 0])  # 1=correct, 0=buggy
                
                # Store problem information for both samples
                problem_info.extend([
                    {
                        'problem_name': problem_name,
                        'original_prompt': original_prompt,
                        'is_correct': True
                    },
                    {
                        'problem_name': problem_name, 
                        'original_prompt': original_prompt,
                        'is_correct': False
                    }
                ])
                
                successful_pairs += 1
                
            except Exception as e:
                print(f"Error processing problem {i}: {e}")
                continue
        
        print(f"Successfully processed {successful_pairs} problem pairs")
        print(f"Total samples: {len(all_activations)}")
        
        # Shape: [num_samples, num_layers, num_heads, head_dim]
        activations_array = np.array(all_activations)
        labels_array = np.array(all_labels)
        
        print(f"Final activations shape: {activations_array.shape}")
        print(f"Final labels shape: {labels_array.shape}")
        
        return activations_array, labels_array, problem_info
    
    def train_head_probes(self, activations, labels):
        """
        Train linear probes for each attention head
        Following ITI paper methodology exactly
        """
        print("Training linear probes for all heads...")
        print(f"Total samples: {len(activations)}")
        print(f"Label distribution: {np.bincount(labels)}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            activations, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        print(f"Train labels: {np.bincount(y_train)}, Val labels: {np.bincount(y_val)}")
        
        head_results = []
        perfect_count = 0
        
        for layer in range(self.num_layers):
            if layer % 5 == 0:
                print(f"Layer {layer+1}/{self.num_layers}")
            
            for head in range(self.num_heads):
                # Extract features for this head: [num_samples, head_dim]
                train_features = X_train[:, layer, head, :]
                val_features = X_val[:, layer, head, :]
                
                # Train linear probe (following paper exactly)
                probe = LogisticRegression(random_state=42, max_iter=1000)
                probe.fit(train_features, y_train)
                
                # Evaluate on validation set
                val_pred = probe.predict(val_features)
                accuracy = accuracy_score(y_val, val_pred)
                
                if accuracy == 1.0:
                    perfect_count += 1
                
                head_results.append({
                    'layer': layer,
                    'head': head,
                    'accuracy': accuracy
                })
        
        print(f"\nDiagnostics:")
        print(f"Heads with perfect accuracy: {perfect_count}/{len(head_results)} ({perfect_count/len(head_results)*100:.1f}%)")
        
        return head_results
    
    def save_results_csv(self, head_results, problems, filename="verilog_qwencoder_head_analysis_with_prompts.csv"):
        """
        Save results to CSV with problem prompts, sorted by accuracy
        """
        df = pd.DataFrame(head_results)
        df = df.sort_values('accuracy', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        df = df[['rank', 'layer', 'head', 'accuracy']]
        
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
        print(f"Top 10 heads for Verilog correctness detection:")
        print(df.head(10))
        
        # Also save a detailed analysis with example problems for top heads
        detailed_filename = "verilog_top_heads_analysis.csv"
        self.save_detailed_analysis(df.head(20), problems, detailed_filename)
        
        return df
    
    def save_detailed_analysis(self, top_heads_df, problems, filename):
        """
        Save detailed analysis showing what types of problems top heads excel at
        """
        print(f"\nCreating detailed analysis for top heads...")
        
        detailed_results = []
        
        for _, row in top_heads_df.iterrows():
            rank = row['rank']
            layer = row['layer'] 
            head = row['head']
            accuracy = row['accuracy']
            
            # Add example problems for context
            example_problems = []
            for i, problem in enumerate(problems[:5]):  # First 5 problems as examples
                example_problems.append({
                    'problem_name': problem.get('problem_name', f'prob_{i}'),
                    'prompt_preview': problem.get('original_prompt', '')[:100] + "..." if len(problem.get('original_prompt', '')) > 100 else problem.get('original_prompt', ''),
                    'category': problem.get('category', 'unknown')
                })
            
            detailed_results.append({
                'rank': rank,
                'layer': layer,
                'head': head, 
                'accuracy': accuracy,
                'example_problem_1': example_problems[0]['problem_name'] if len(example_problems) > 0 else '',
                'example_prompt_1': example_problems[0]['prompt_preview'] if len(example_problems) > 0 else '',
                'example_problem_2': example_problems[1]['problem_name'] if len(example_problems) > 1 else '',
                'example_prompt_2': example_problems[1]['prompt_preview'] if len(example_problems) > 1 else '',
                'example_problem_3': example_problems[2]['problem_name'] if len(example_problems) > 2 else '',
                'example_prompt_3': example_problems[2]['prompt_preview'] if len(example_problems) > 2 else ''
            })
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(filename, index=False)
        print(f"Detailed analysis saved to {filename}")
        print(f"Shows example problems that top heads are analyzing.")

def main():
    """
    Simple ITI implementation for Verilog code correctness using FULL dataset
    """
    print("="*60)
    print("SIMPLE ITI: VERILOG CORRECTNESS HEAD ANALYSIS")
    print("Following core ITI paper methodology")
    print("Using FULL DATASET with original problem prompts")
    print("="*60)
    
    # Load data
    with open(JSON_PATH, 'r') as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} problems")
    
    # Initialize ITI
    iti = SimpleVerilogITI()
    
    # Collect activations from ALL correct/buggy pairs with problem info
    activations, labels, problem_info = iti.collect_activations(problems)
    print(f"Collected activations shape: {activations.shape}")
    
    # Train probes for all heads
    head_results = iti.train_head_probes(activations, labels)
    
    # Save results with problem context
    df = iti.save_results_csv(head_results, problems)
    
    print(f"\nAnalysis complete!")
    print(f"Best head accuracy: {df['accuracy'].max():.3f}")
    print(f"Analyzed {len(head_results)} attention heads total")
    print(f"Results now include analysis of what types of Verilog problems each head excels at")

if __name__ == "__main__":
    main() 