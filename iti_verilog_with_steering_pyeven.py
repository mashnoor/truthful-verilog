import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
import re
from tqdm import tqdm
import pyvene as pv
warnings.filterwarnings('ignore')

# Configuration
JSON_PATH = "verilog_eval_codes_verified.json"
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
DATASET_DIR = "dataset_spec-to-rtl"

class Collector:
    """Collects activations from model layers using pyvene"""
    def __init__(self, multiplier=0, head=-1):
        self.multiplier = multiplier
        self.head = head
        self.collect_state = True
        self.states = []
        
    def __call__(self, base, source):
        # For collection, we just store the activations
        if self.collect_state:
            self.states.append(base.clone().detach())
        return base
        
    def reset(self):
        self.states = []
        
class ITI_Intervener:
    """ITI intervention based on honest_llama implementation"""
    def __init__(self, direction, alpha):
        self.direction = direction
        self.alpha = alpha
        
    def __call__(self, base, source):
        # Apply steering by adding scaled direction to activations
        steering = self.direction.to(base.device) * self.alpha
        return base + steering

def wrapper(intervener):
    """Wrapper for pyvene compatibility"""
    return intervener

class VerilogITIWithPyvene:
    """
    Verilog ITI implementation using pyvene framework
    Based on honest_llama methodology for robust interventions
    """
    
    def __init__(self):
        print(f"Loading {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        
        # Model dimensions
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads  
        self.hidden_size = self.model.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        print(f"Model loaded: {self.num_layers} layers, {self.num_heads} heads per layer")
        
        self.probes = {}
        self.steering_vectors = {}
        self.head_accuracies = {}
        
    def get_activations_pyvene(self, prompts, labels):
        """Extract activations using pyvene collectors"""
        print("Setting up pyvene collectors...")
        
        # Create collectors for each layer
        collectors = []
        pv_config = []
        for layer in range(self.num_layers):
            collector = Collector(multiplier=0, head=-1)
            collectors.append(collector)
            pv_config.append({
                "component": f"model.layers[{layer}].self_attn.o_proj.input",
                "intervention": wrapper(collector),
            })
        
        collected_model = pv.IntervenableModel(pv_config, self.model)
        
        all_activations = []
        print("Extracting activations...")
        
        for prompt in tqdm(prompts):
            with torch.no_grad():
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Run through collected model
                output = collected_model(inputs)
                
                # Collect activations from all layers
                layer_activations = []
                for collector in collectors:
                    if collector.collect_state and len(collector.states) > 0:
                        # Get last token activations and reshape to [num_heads, head_dim]
                        activation = collector.states[-1][:, -1, :].cpu()  # [batch=1, hidden_size]
                        activation = activation.reshape(self.num_heads, self.head_dim)
                        layer_activations.append(activation)
                    collector.reset()
                
                # Stack to [num_layers, num_heads, head_dim]
                if layer_activations:
                    sample_activations = torch.stack(layer_activations, dim=0)
                    all_activations.append(sample_activations.numpy())
        
        return np.array(all_activations)
    
    def train_probes_and_get_steering_vectors(self, activations, labels):
        """Train probes and compute steering vectors for each attention head"""
        print("Training probes and computing steering vectors...")
        
        # Reshape activations: [samples, layers, heads, head_dim] -> [samples*layers*heads, head_dim]
        samples, layers, heads, head_dim = activations.shape
        
        probe_results = []
        
        for layer in range(layers):
            for head in range(heads):
                # Extract activations for this specific head
                head_activations = activations[:, layer, head, :]  # [samples, head_dim]
                
                # Split into train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    head_activations, labels, test_size=0.2, random_state=42, stratify=labels
                )
                
                # Train probe
                probe = LogisticRegression(random_state=42, max_iter=1000)
                probe.fit(X_train, y_train)
                
                # Evaluate
                y_pred = probe.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store probe and accuracy
                head_key = (layer, head)
                self.probes[head_key] = probe
                self.head_accuracies[head_key] = accuracy
                
                # Compute steering vector using "mass mean shift"
                correct_mask = labels == 1
                buggy_mask = labels == 0
                
                correct_mean = head_activations[correct_mask].mean(axis=0)
                buggy_mean = head_activations[buggy_mask].mean(axis=0)
                steering_vector = correct_mean - buggy_mean
                
                self.steering_vectors[head_key] = steering_vector
                
                probe_results.append({
                    'layer': layer,
                    'head': head,
                    'accuracy': accuracy
                })
        
        return probe_results
    
    def get_top_heads(self, num_top_heads=48):
        """Get top performing heads for intervention"""
        sorted_heads = sorted(self.head_accuracies.items(), key=lambda x: x[1], reverse=True)
        top_heads = [head for head, accuracy in sorted_heads[:num_top_heads]]
        print(f"Selected top {len(top_heads)} heads for intervention")
        return top_heads
    
    def create_intervened_model(self, top_heads, alpha=15.0):
        """Create intervened model using pyvene"""
        print(f"Creating intervened model with alpha={alpha}")
        
        # Group heads by layer for efficient intervention
        top_heads_by_layer = {}
        for layer, head in top_heads:
            if layer not in top_heads_by_layer:
                top_heads_by_layer[layer] = []
            top_heads_by_layer[layer].append(head)
        
        # Create interventions
        pv_config = []
        for layer, heads in top_heads_by_layer.items():
            # Create combined direction for all heads in this layer
            direction = torch.zeros(self.head_dim * self.num_heads)
            
            for head in heads:
                head_key = (layer, head)
                if head_key in self.steering_vectors:
                    steering_vec = torch.tensor(self.steering_vectors[head_key], dtype=torch.float32)
                    steering_vec = steering_vec / torch.norm(steering_vec)  # Normalize
                    
                    # Place in correct position for this head
                    start_idx = head * self.head_dim
                    end_idx = (head + 1) * self.head_dim
                    direction[start_idx:end_idx] = steering_vec
            
            # Create intervener for this layer
            intervener = ITI_Intervener(direction, alpha)
            pv_config.append({
                "component": f"model.layers[{layer}].self_attn.o_proj.input",
                "intervention": wrapper(intervener),
            })
        
        return pv.IntervenableModel(pv_config, self.model)
    
    def generate_verilog_code(self, model, prompt, max_length=1000):
        """Generate Verilog code using the model"""
        # Format prompt for code generation
        formatted_prompt = f"Generate Verilog code for the following specification:\n\n{prompt}\n\nVerilog code:\n```systemverilog\n"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Generate using the provided model (could be original or intervened)
            if hasattr(model, 'generate'):
                # Original model
                outputs = model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            else:
                # Intervened model (pyvene IntervenableModel)
                outputs = model.generate(
                    inputs,
                    max_length=inputs['input_ids'].shape[1] + max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                outputs = outputs[1]  # pyvene returns (intervention_outputs, generation_outputs)
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract Verilog code between ```systemverilog and ```
        code_match = re.search(r'```systemverilog\s*(.*?)\s*```', generated_text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        else:
            # If no code block found, return the raw generation
            return generated_text.strip()
    

    
    def evaluate_generation(self, num_problems=10, alpha=15.0, num_top_heads=48):
        """
        Evaluate Verilog code generation with and without steering
        Generate JSON with all implementations for comparison
        """
        print("Loading dataset...")
        with open(JSON_PATH, 'r') as f:
            problems = json.load(f)
        
        # Limit to specified number of problems
        problems = problems[:num_problems]
        
        # First, collect activations for probe training using existing correct/buggy implementations
        print("Collecting activations for probe training...")
        prompts = []
        labels = []
        
        for problem in problems:
            prompt = problem.get('original_prompt', problem.get('prompt', ''))
            
            # Use correct implementation
            correct_code = problem.get('correct_code', problem.get('reference_code', ''))
            correct_input = f"{prompt}\n\nImplementation:\n{correct_code}"
            prompts.append(correct_input)
            labels.append(1)  # Correct
            
            # Use buggy implementation  
            buggy_code = problem.get('buggy_code', problem.get('incorrect_code', ''))
            if buggy_code:
                buggy_input = f"{prompt}\n\nImplementation:\n{buggy_code}"
                prompts.append(buggy_input)
                labels.append(0)  # Buggy
        
        # Get activations
        activations = self.get_activations_pyvene(prompts, labels)
        
        # Train probes and get steering vectors
        probe_results = self.train_probes_and_get_steering_vectors(activations, np.array(labels))
        
        # Get top heads and create intervened model
        top_heads = self.get_top_heads(num_top_heads)
        intervened_model = self.create_intervened_model(top_heads, alpha)
        
        # Now evaluate generation
        print("Generating Verilog code with and without steering...")
        evaluation_results = []
        
        for i, problem in enumerate(tqdm(problems)):
            problem_name = problem.get('problem_name', f'problem_{i}')
            prompt = problem.get('original_prompt', problem.get('prompt', ''))
            original_implementation = problem.get('correct_code', problem.get('reference_code', ''))
            
            # Generate code without steering (original model)
            code_without_steering = self.generate_verilog_code(self.model, prompt)
            
            # Generate code with steering (intervened model)
            code_with_steering = self.generate_verilog_code(intervened_model, prompt)
            
            evaluation_results.append({
                'problem_name': problem_name,
                'original_prompt': prompt,
                'original_implementation': original_implementation,
                'without_steering_implementation': code_without_steering,
                'with_steering_implementation': code_with_steering
            })
            
            print(f"Generated {i+1}/{len(problems)}: {problem_name}")
        
        # Save results as JSON
        output_file = f'verilog_iti_steering_evaluation_alpha_{alpha}_heads_{num_top_heads}.json'
        with open(output_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print("=" * 60)
        print("EVALUATION COMPLETED")
        print("=" * 60)
        print(f"Generated code for {len(problems)} problems")
        print(f"Alpha: {alpha}, Top heads: {num_top_heads}")
        print(f"Results saved to: {output_file}")
        
        return evaluation_results, probe_results

def main():
    """Main evaluation function"""
    iti = VerilogITIWithPyvene()
    
    # Run evaluation
    results_df, probe_results = iti.evaluate_generation(
        num_problems=20,  # Start with smaller number for testing
        alpha=15.0,
        num_top_heads=48
    )
    
    print("Evaluation completed! Results saved to 'verilog_iti_steering_evaluation.csv'")
    
    return results_df, probe_results

if __name__ == "__main__":
    main() 