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

class VerilogITIVerifier:
    """
    Uses ITI-trained probes to verify generated Verilog code
    Rather than steering during generation
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
        self.hooks = []
        
    def get_head_activations(self, prompts):
        """Extract attention head activations for verification"""
        all_activations = {(l, h): [] for l in range(self.num_layers) for h in range(self.num_heads)}
        
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Extracting activations")):
            # Clear any existing hooks
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
            
            # Storage for this sample
            sample_activations = {}
            
            def make_hook(layer_idx):
                def hook(module, input, output):
                    head_outputs = input[0]
                    last_token_outputs = head_outputs[:, -1, :]
                    reshaped = last_token_outputs.view(-1, self.num_heads, self.head_dim)
                    
                    for h in range(self.num_heads):
                        sample_activations[(layer_idx, h)] = reshaped[0, h, :].detach().cpu().numpy()
                
                return hook
            
            # Register hooks
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
            if acts:
                activations_dict[key] = np.array(acts)
        
        return activations_dict
    
    def train_probes(self, prompts, labels):
        """Train probes to distinguish correct from buggy code"""
        print("Training verification probes...")
        
        # Get activations
        activations_dict = self.get_head_activations(prompts)
        
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
                
                probe_results.append({
                    'layer': layer,
                    'head': head,
                    'accuracy': accuracy
                })
        
        # Sort by accuracy
        probe_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return probe_results
    
    def verify_code(self, prompt, code, top_k=10):
        """
        Use trained probes to verify if generated code is likely correct
        Returns a score between 0 and 1
        """
        # Get top performing heads
        sorted_heads = sorted(self.head_accuracies.items(), key=lambda x: x[1], reverse=True)
        top_heads = [head for head, _ in sorted_heads[:top_k]]
        
        # Format prompt with code
        full_prompt = f"{prompt}\n\nImplementation:\n{code}"
        
        # Get activations for this code
        activations = self.get_head_activations([full_prompt])
        
        # Get predictions from top heads
        predictions = []
        weights = []
        
        for head_key in top_heads:
            if head_key in activations and head_key in self.head_probes:
                probe = self.head_probes[head_key]
                accuracy = self.head_accuracies[head_key]
                
                # Get activation for this head
                head_activation = activations[head_key][0].reshape(1, -1)
                
                # Get probability of being correct
                prob_correct = probe.predict_proba(head_activation)[0, 1]
                
                predictions.append(prob_correct)
                weights.append(accuracy)
        
        # Weighted average of predictions
        if predictions:
            weights = np.array(weights)
            predictions = np.array(predictions)
            score = np.average(predictions, weights=weights)
        else:
            score = 0.5  # No information
        
        return score
    
    def generate_code(self, prompt, max_length=1000):
        """Generate Verilog code normally without steering"""
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
    
    def generate_and_verify(self, prompt, num_attempts=3, threshold=0.7):
        """
        Generate code and verify it. Regenerate if verification score is low.
        """
        best_code = None
        best_score = 0
        
        for attempt in range(num_attempts):
            # Generate code
            code = self.generate_code(prompt)
            
            # Verify code
            score = self.verify_code(prompt, code)
            
            if self.debug:
                print(f"Attempt {attempt + 1}: Verification score = {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_code = code
            
            # If score is good enough, stop
            if score >= threshold:
                break
        
        return best_code, best_score

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Verilog ITI Verifier - Uses probes to verify generated code')
    parser.add_argument('--num_problems', type=int, default=10, help='Number of problems to evaluate')
    parser.add_argument('--num_attempts', type=int, default=3, help='Number of generation attempts per problem')
    parser.add_argument('--threshold', type=float, default=0.7, help='Verification threshold')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    print(f"Running Verilog ITI Verifier with parameters:")
    print(f"  Number of problems: {args.num_problems}")
    print(f"  Generation attempts: {args.num_attempts}")
    print(f"  Verification threshold: {args.threshold}")
    
    verifier = VerilogITIVerifier(debug=args.debug)
    
    # Load dataset
    print("\nLoading dataset...")
    with open(JSON_PATH, 'r') as f:
        problems = json.load(f)
    
    # Prepare training data
    train_prompts = []
    train_labels = []
    
    # Use problems for training
    train_problems = problems[args.num_problems:args.num_problems+100]
    
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
    
    # Train probes
    probe_results = verifier.train_probes(train_prompts, train_labels)
    
    print("\nTop 10 verification heads:")
    for i, result in enumerate(probe_results[:10]):
        print(f"  {i+1}. Layer {result['layer']}, Head {result['head']}: {result['accuracy']:.3f}")
    
    # Evaluate
    print("\nGenerating and verifying code...")
    results = []
    
    eval_problems = problems[:args.num_problems]
    
    for i, problem in enumerate(tqdm(eval_problems)):
        problem_name = problem.get('problem_name', f'problem_{i}')
        prompt = problem.get('original_prompt', '')
        reference_code = problem.get('reference_implementation', '')
        
        # Generate and verify
        generated_code, verification_score = verifier.generate_and_verify(
            prompt, 
            num_attempts=args.num_attempts,
            threshold=args.threshold
        )
        
        results.append({
            'problem_name': problem_name,
            'original_prompt': prompt,
            'original_implementation': reference_code,
            'generated_implementation': generated_code,
            'verification_score': float(verification_score)
        })
        
        print(f"Problem {problem_name}: Verification score = {verification_score:.3f}")
    
    # Save results
    output_file = f'verilog_verification_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    scores = [r['verification_score'] for r in results]
    print(f"\nAverage verification score: {np.mean(scores):.3f}")
    print(f"Min score: {np.min(scores):.3f}, Max score: {np.max(scores):.3f}")

if __name__ == "__main__":
    main() 