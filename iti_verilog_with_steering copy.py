import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import re
from tqdm import tqdm
import argparse
warnings.filterwarnings('ignore')

# Configuration
JSON_PATH = "verilog_eval_codes_verified.json"
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

class VerilogITI:
    """
    Verilog ITI implementation following the exact methodology from the paper:
    "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"
    """
    
    def __init__(self, debug=False):
        self.debug = debug
        print(f"Loading {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Model dimensions
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads  
        self.hidden_size = self.model.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        print(f"Model loaded: {self.num_layers} layers, {self.num_heads} heads per layer")
        
        # Storage for ITI components
        self.head_probes = {}  # (layer, head) -> probe
        self.head_accuracies = {}  # (layer, head) -> accuracy
        self.truthful_directions = {}  # (layer, head) -> direction vector
        self.activation_stds = {}  # (layer, head) -> std along truthful direction
        self.hooks = []
        
    def get_head_activations(self, prompts, labels):
        """
        Extract attention head activations following ITI paper methodology.
        Activations are taken at the last token position.
        """
        print("Extracting attention head activations...")
        
        # Storage for all activations
        all_activations = {(l, h): [] for l in range(self.num_layers) for h in range(self.num_heads)}
        
        for prompt_idx, prompt in enumerate(tqdm(prompts)):
            # Clear any existing hooks
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
            
            # Storage for this sample
            sample_activations = {}
            
            def make_hook(layer_idx):
                def hook(module, input, output):
                    # Input to o_proj contains concatenated head outputs
                    # Shape: [batch, seq_len, hidden_size]
                    head_outputs = input[0]
                    
                    # Take last token
                    last_token_outputs = head_outputs[:, -1, :]  # [batch, hidden_size]
                    
                    # Reshape to separate heads: [batch, num_heads, head_dim]
                    reshaped = last_token_outputs.view(-1, self.num_heads, self.head_dim)
                    
                    # Store each head's activation
                    for h in range(self.num_heads):
                        sample_activations[(layer_idx, h)] = reshaped[0, h, :].detach().cpu().numpy()
                
                return hook
            
            # Register hooks for all layers
            for layer_idx in range(self.num_layers):
                layer = self.model.model.layers[layer_idx]
                hook = layer.self_attn.o_proj.register_forward_hook(make_hook(layer_idx))
                self.hooks.append(hook)
            
            # Run forward pass
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = self.model(**inputs)
            
            # Store activations
            for key, activation in sample_activations.items():
                all_activations[key].append(activation)
            
            # Clean up hooks
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
        
        # Convert to numpy arrays
        activations_dict = {}
        for key, acts in all_activations.items():
            if acts:  # Only store if we have activations
                activations_dict[key] = np.array(acts)
        
        return activations_dict
    
    def train_probes_and_compute_directions(self, prompts, labels):
        """
        Train probes and compute truthful directions following ITI paper:
        1. Train linear probes for each head
        2. Compute mass mean shift directions (true_mean - false_mean)
        3. Calculate standard deviations along truthful directions
        """
        print("Training probes and computing truthful directions...")
        
        # Get activations
        activations_dict = self.get_head_activations(prompts, labels)
        
        # Train probe for each head
        probe_results = []
        
        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                key = (layer, head)
                
                if key not in activations_dict:
                    continue
                    
                head_activations = activations_dict[key]
                
                # Split data
                X_train, X_val, y_train, y_val = train_test_split(
                    head_activations, labels, test_size=0.2, random_state=42, stratify=labels
                )
                
                # Train probe
                probe = LogisticRegression(random_state=42, max_iter=1000)
                probe.fit(X_train, y_train)
                
                # Evaluate
                y_pred = probe.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                
                # Store results
                self.head_probes[key] = probe
                self.head_accuracies[key] = accuracy
                
                # Compute mass mean shift direction (following ITI paper)
                true_mask = labels == 1
                false_mask = labels == 0
                
                if np.sum(true_mask) > 0 and np.sum(false_mask) > 0:
                    true_mean = head_activations[true_mask].mean(axis=0)
                    false_mean = head_activations[false_mask].mean(axis=0)
                    
                    truthful_direction = true_mean - false_mean
                    
                    # Normalize the direction vector and calculate std of projections
                    direction_norm = np.linalg.norm(truthful_direction)
                    if direction_norm > 0:
                        normalized_direction = truthful_direction / direction_norm
                        self.truthful_directions[key] = normalized_direction  # Store normalized vector
                        
                        projections = head_activations @ normalized_direction
                        std_along_direction = np.std(projections)
                        self.activation_stds[key] = std_along_direction
                    else:
                        # Handle zero-norm case
                        self.truthful_directions[key] = np.zeros_like(truthful_direction)
                        self.activation_stds[key] = 1.0
                
                probe_results.append({
                    'layer': layer,
                    'head': head,
                    'accuracy': accuracy
                })
        
        # Sort by accuracy
        probe_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return probe_results
    
    def select_top_heads(self, K=48):
        """Select top K heads based on probe accuracy"""
        sorted_heads = sorted(self.head_accuracies.items(), key=lambda x: x[1], reverse=True)
        top_heads = [head for head, _ in sorted_heads[:K]]
        
        print(f"Selected top {len(top_heads)} heads for intervention")
        print(f"Accuracy range: {sorted_heads[0][1]:.3f} - {sorted_heads[K-1][1]:.3f}")
        
        return top_heads
    
    def apply_iti_hooks(self, top_heads, alpha=15.0):
        """
        Apply ITI intervention using a forward pre-hook on o_proj.
        This modifies the aggregated head outputs before the final projection.
        """
        print(f"Applying ITI with alpha={alpha} on {len(top_heads)} heads")
        
        def make_iti_pre_hook(layer_idx, head_indices, directions, stds, alpha):
            def pre_hook(module, input_tuple):
                # input_tuple is a tuple, the tensor is the first element
                modified_input = input_tuple[0].clone()
                
                batch_size, seq_len, hidden_size = modified_input.shape
                reshaped = modified_input.view(batch_size, seq_len, self.num_heads, self.head_dim)
                
                if self.debug and not hasattr(self, '_debug_printed'):
                    print(f"\nDEBUG: Intervening on layer {layer_idx} with pre-hook")
                    self._debug_printed = True

                # Apply intervention to selected heads
                for i, head_idx in enumerate(head_indices):
                    if i < len(directions):
                        direction = directions[i]
                        std = stds[i]
                        
                        direction_tensor = torch.tensor(direction, dtype=modified_input.dtype, device=modified_input.device)
                        
                        # Add intervention to the last token's activation for generation
                        intervention = alpha * std * direction_tensor
                        reshaped[:, -1, head_idx, :] += intervention
                
                # Reshape back and return as a tuple for the pre-hook
                modified_input = reshaped.view(batch_size, seq_len, hidden_size)
                return (modified_input,)
            
            return pre_hook
        
        # Group heads by layer
        heads_by_layer = {}
        for layer, head in top_heads:
            if layer not in heads_by_layer:
                heads_by_layer[layer] = []
            heads_by_layer[layer].append(head)
        
        # Apply hooks
        for layer_idx, head_indices in heads_by_layer.items():
            directions = [self.truthful_directions.get((layer_idx, h)) for h in head_indices]
            stds = [self.activation_stds.get((layer_idx, h), 1.0) for h in head_indices]
            
            # Filter out any heads for which we don't have directions
            valid_indices = [i for i, d in enumerate(directions) if d is not None]
            if not valid_indices:
                continue
                
            filtered_head_indices = [head_indices[i] for i in valid_indices]
            filtered_directions = [directions[i] for i in valid_indices]
            filtered_stds = [stds[i] for i in valid_indices]

            layer = self.model.model.layers[layer_idx]
            hook = layer.self_attn.o_proj.register_forward_pre_hook(
                make_iti_pre_hook(layer_idx, filtered_head_indices, filtered_directions, filtered_stds, alpha)
            )
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all intervention hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_code(self, prompt, max_length=1000):
        """Generate Verilog code with current model state"""
        formatted_prompt = f"Generate Verilog code for the following specification:\n\n{prompt}\n\nVerilog code:\n```systemverilog\n"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract code
        code_match = re.search(r'```systemverilog\s*(.*?)\s*```', generated_text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        else:
            # Try to extract any code-like content
            lines = generated_text.strip().split('\n')
            code_lines = []
            for line in lines:
                if any(keyword in line.lower() for keyword in ['module', 'input', 'output', 'wire', 'reg', 'assign', 'always', 'endmodule']):
                    code_lines.append(line)
            return '\n'.join(code_lines) if code_lines else generated_text.strip()
    
    def evaluate_iti(self, num_problems=10, K=48, alpha=15.0):
        """
        Main evaluation function following ITI paper methodology
        """
        print("=" * 60)
        print("VERILOG ITI EVALUATION")
        print("=" * 60)
        
        # Load dataset
        print("Loading dataset...")
        with open(JSON_PATH, 'r') as f:
            problems = json.load(f)
        
        # Use subset for evaluation
        eval_problems = problems[:num_problems]
        
        # Prepare training data for probes
        print("Preparing training data...")
        train_prompts = []
        train_labels = []
        
        # Use remaining problems for training (or could use same problems)
        train_problems = problems[num_problems:num_problems+100] if len(problems) > num_problems else problems
        
        for problem in train_problems:
            prompt = problem.get('original_prompt', '')
            
            # Add correct example
            correct_code = problem.get('reference_implementation', '')
            if correct_code:
                train_prompts.append(f"{prompt}\n\nImplementation:\n{correct_code}")
                train_labels.append(1)
            
            # Add buggy example
            buggy_code = problem.get('buggy_implementation', '')
            if buggy_code:
                train_prompts.append(f"{prompt}\n\nImplementation:\n{buggy_code}")
                train_labels.append(0)
        
        train_labels = np.array(train_labels)
        
        # Train probes and compute directions
        probe_results = self.train_probes_and_compute_directions(train_prompts, train_labels)
        
        # Select top heads
        top_heads = self.select_top_heads(K)
        
        # Evaluate generation
        print("\nGenerating code with and without ITI...")
        results = []
        
        for i, problem in enumerate(tqdm(eval_problems)):
            problem_name = problem.get('problem_name', f'problem_{i}')
            prompt = problem.get('original_prompt', '')
            reference_code = problem.get('reference_implementation', '')
            
            # Generate without ITI
            code_without_iti = self.generate_code(prompt)
            
            # Apply ITI and generate
            # Clear debug flags
            if hasattr(self, '_debug_printed'):
                delattr(self, '_debug_printed')
            if hasattr(self, '_intervention_logged'):
                delattr(self, '_intervention_logged')
            
            self.apply_iti_hooks(top_heads, alpha)
            code_with_iti = self.generate_code(prompt)
            self.remove_hooks()
            
            results.append({
                'problem_name': problem_name,
                'original_prompt': prompt,
                'original_implementation': reference_code,
                'without_steering_implementation': code_without_iti,
                'with_steering_implementation': code_with_iti
            })
            
            print(f"Completed {i+1}/{len(eval_problems)}: {problem_name}")
        
        # Save results
        output_file = f'verilog_iti_results_K{K}_alpha{alpha}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED")
        print("=" * 60)
        print(f"Problems evaluated: {len(eval_problems)}")
        print(f"Top K heads: {K}")
        print(f"Alpha (intervention strength): {alpha}")
        print(f"Results saved to: {output_file}")
        
        # Print some statistics about the probes
        print("\nTop 10 attention heads by probe accuracy:")
        for i, result in enumerate(probe_results[:10]):
            print(f"  {i+1}. Layer {result['layer']}, Head {result['head']}: {result['accuracy']:.3f}")
        
        return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Verilog ITI - Inference-Time Intervention for Verilog Code Generation')
    parser.add_argument('--alpha', type=float, default=1.0, help='Intervention strength (default: 1.0)')
    parser.add_argument('--K', type=int, default=48, help='Number of top heads to use for intervention (default: 48)')
    parser.add_argument('--num_problems', type=int, default=10, help='Number of problems to evaluate (default: 10)')
    parser.add_argument('--test_alphas', action='store_true', help='Test multiple alpha values')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    print(f"Running Verilog ITI with parameters:")
    print(f"  Alpha: {args.alpha}")
    print(f"  K (top heads): {args.K}")
    print(f"  Number of problems: {args.num_problems}")
    
    iti = VerilogITI(debug=args.debug)
    
    # Test different alpha values if requested
    if args.test_alphas:
        alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
        print(f"\nTesting multiple alpha values: {alphas}")
        for alpha in alphas:
            print(f"\n\n{'='*60}")
            print(f"Testing alpha = {alpha}")
            print('='*60)
            results = iti.evaluate_iti(
                num_problems=min(3, args.num_problems),  # Just test on a few problems
                K=args.K,
                alpha=alpha
            )
            # Quick check of results
            for r in results:
                print(f"\nProblem: {r['problem_name']}")
                print(f"Without steering preview: {r['without_steering_implementation'][:100]}...")
                print(f"With steering preview: {r['with_steering_implementation'][:100]}...")
    else:
        # Run evaluation with specified parameters
        results = iti.evaluate_iti(
            num_problems=args.num_problems,
            K=args.K,
            alpha=args.alpha
        )
    
    print("\nEvaluation completed successfully!")
    return results

if __name__ == "__main__":
    main() 