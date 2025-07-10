import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from transformers import GPT2Tokenizer, GPT2Model
import json

class AttentionHeadVisualizer:
    def __init__(self, model_name="gpt2"):
        """Initialize the attention head visualizer."""
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
        self.model.eval()
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Sample Q/A pairs for demonstration
        self.sample_qa_pairs = [
            {
                "question": "What shape did scholars think the Earth was?",
                "truthful_answer": "Scholars thought the Earth was round.",
                "false_answer": "Scholars thought the Earth was flat.",
            },
            {
                "question": "What is the capital of France?",
                "truthful_answer": "The capital of France is Paris.",
                "false_answer": "The capital of France is London.",
            },
            {
                "question": "Do vaccines cause autism?",
                "truthful_answer": "No, vaccines do not cause autism.",
                "false_answer": "Yes, vaccines cause autism.",
            },
            {
                "question": "What is 2+2?",
                "truthful_answer": "2+2 equals 4.",
                "false_answer": "2+2 equals 5.",
            },
            {
                "question": "Is the Earth flat?",
                "truthful_answer": "No, the Earth is not flat.",
                "false_answer": "Yes, the Earth is flat.",
            }
        ]
    
    def get_model_info(self):
        """Get information about the model architecture."""
        num_layers = self.model.config.n_layer
        num_heads = self.model.config.n_head
        hidden_size = self.model.config.n_embd
        
        print(f"Model Architecture Information:")
        print(f"  - Number of layers: {num_layers}")
        print(f"  - Number of attention heads per layer: {num_heads}")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Total attention heads: {num_layers * num_heads}")
        print(f"  - Head dimension: {hidden_size // num_heads}")
        
        return num_layers, num_heads, hidden_size
    
    def get_attention_weights(self, text, layer_idx=0, head_idx=0):
        """Extract attention weights for a specific layer and head."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        attention_weights = outputs.attentions[layer_idx][0, head_idx]
        return attention_weights.numpy(), inputs['input_ids'][0]
    
    def visualize_single_attention_head(self, text, layer_idx=0, head_idx=0, figsize=(12, 8)):
        """Visualize attention patterns for a single attention head."""
        attention_weights, token_ids = self.get_attention_weights(text, layer_idx, head_idx)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(attention_weights, 
                    xticklabels=tokens, 
                    yticklabels=tokens,
                    cmap='Blues',
                    annot=False,
                    cbar_kws={'label': 'Attention Weight'})
        
        plt.title(f'Attention Head {head_idx} in Layer {layer_idx}\nText: "{text[:50]}{"..." if len(text) > 50 else ""}"')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return plt
    
    def visualize_multiple_heads(self, text, layer_idx=0, num_heads=8, figsize=(20, 15)):
        """Visualize multiple attention heads in a grid."""
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.flatten()
        
        for head_idx in range(min(num_heads, 8)):
            attention_weights, token_ids = self.get_attention_weights(text, layer_idx, head_idx)
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            
            # Create heatmap for this head
            sns.heatmap(attention_weights, 
                        xticklabels=tokens, 
                        yticklabels=tokens,
                        cmap='Blues',
                        annot=False,
                        ax=axes[head_idx],
                        cbar=False)
            
            axes[head_idx].set_title(f'Head {head_idx}')
            axes[head_idx].set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            axes[head_idx].set_yticklabels(tokens, rotation=0, fontsize=8)
        
        plt.suptitle(f'Attention Heads in Layer {layer_idx}\nText: "{text[:50]}{"..." if len(text) > 50 else ""}"')
        plt.tight_layout()
        
        return plt
    
    def visualize_qa_comparison(self, layer_idx=0, head_idx=0, figsize=(16, 12)):
        """Visualize attention patterns for Q/A pairs comparison."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        for i, qa_pair in enumerate(self.sample_qa_pairs[:4]):  # Show first 4 Q/A pairs
            # Truthful answer attention
            truthful_text = qa_pair["question"] + " " + qa_pair["truthful_answer"]
            truthful_weights, token_ids = self.get_attention_weights(truthful_text, layer_idx, head_idx)
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            
            # False answer attention
            false_text = qa_pair["question"] + " " + qa_pair["false_answer"]
            false_weights, _ = self.get_attention_weights(false_text, layer_idx, head_idx)
            
            row = i // 2
            col = i % 2
            
            # Plot truthful attention
            sns.heatmap(truthful_weights, 
                        xticklabels=tokens, 
                        yticklabels=tokens,
                        cmap='Blues',
                        annot=False,
                        ax=axes[row, col],
                        cbar_kws={'label': 'Attention Weight'})
            
            axes[row, col].set_title(f'Q{i+1}: {qa_pair["question"][:30]}...\nTruthful Answer')
            axes[row, col].set_xticklabels(tokens, rotation=45, ha='right', fontsize=6)
            axes[row, col].set_yticklabels(tokens, rotation=0, fontsize=6)
        
        plt.suptitle(f'Q/A Attention Comparison\nLayer {layer_idx}, Head {head_idx}')
        plt.tight_layout()
        
        return plt
    
    def visualize_intervention_effect(self, layer_idx=0, head_idx=0, intervention_strength=1.0, figsize=(16, 8)):
        """Visualize the effect of intervention on attention patterns."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Original attention
        sample_qa = self.sample_qa_pairs[0]
        original_text = sample_qa["question"] + " " + sample_qa["truthful_answer"]
        original_weights, token_ids = self.get_attention_weights(original_text, layer_idx, head_idx)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        # Simulate intervention effect
        intervened_weights = original_weights.copy()
        # Add intervention effect (simplified)
        intervened_weights += intervention_strength * 0.1 * np.random.randn(*intervened_weights.shape)
        intervened_weights = np.clip(intervened_weights, 0, 1)  # Ensure valid attention weights
        
        # Plot original attention
        sns.heatmap(original_weights, 
                    xticklabels=tokens, 
                    yticklabels=tokens,
                    cmap='Blues',
                    annot=False,
                    ax=ax1,
                    cbar_kws={'label': 'Attention Weight'})
        ax1.set_title('Original Attention')
        ax1.set_xticklabels(tokens, rotation=45, ha='right')
        ax1.set_yticklabels(tokens, rotation=0)
        
        # Plot intervened attention
        sns.heatmap(intervened_weights, 
                    xticklabels=tokens, 
                    yticklabels=tokens,
                    cmap='Blues',
                    annot=False,
                    ax=ax2,
                    cbar_kws={'label': 'Attention Weight'})
        ax2.set_title(f'Intervened Attention (Strength: {intervention_strength})')
        ax2.set_xticklabels(tokens, rotation=45, ha='right')
        ax2.set_yticklabels(tokens, rotation=0)
        
        plt.suptitle(f'Intervention Effect on Attention\nLayer {layer_idx}, Head {head_idx}')
        plt.tight_layout()
        
        return plt
    
    def create_comprehensive_visualization(self):
        """Create a comprehensive attention visualization summary."""
        fig = plt.figure(figsize=(20, 12))
        
        # Main title
        fig.suptitle('Attention Head Visualization\nBased on ITI Research', 
                    fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Single attention head visualization
        ax1 = fig.add_subplot(gs[0, :2])
        sample_text = self.sample_qa_pairs[0]["question"] + " " + self.sample_qa_pairs[0]["truthful_answer"]
        attention_weights, token_ids = self.get_attention_weights(sample_text, 0, 0)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        im1 = ax1.imshow(attention_weights, cmap='Blues', aspect='auto')
        ax1.set_xticks(range(len(tokens)))
        ax1.set_yticks(range(len(tokens)))
        ax1.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax1.set_yticklabels(tokens, rotation=0, fontsize=8)
        ax1.set_title('Single Attention Head\n(Layer 0, Head 0)')
        plt.colorbar(im1, ax=ax1, label='Attention Weight')
        
        # 2. Multiple heads comparison
        ax2 = fig.add_subplot(gs[0, 2:])
        # Create a simplified multi-head visualization
        head_weights = np.random.rand(4, 4) * 0.5 + 0.2  # Simulated data
        im2 = ax2.imshow(head_weights, cmap='Blues', aspect='auto')
        ax2.set_title('Multiple Attention Heads\n(Layer 0)')
        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Attention Head')
        plt.colorbar(im2, ax=ax2, label='Average Attention Weight')
        
        # 3. Q/A comparison
        ax3 = fig.add_subplot(gs[1, :2])
        truthful_text = self.sample_qa_pairs[0]["question"] + " " + self.sample_qa_pairs[0]["truthful_answer"]
        truthful_weights, _ = self.get_attention_weights(truthful_text, 0, 0)
        false_text = self.sample_qa_pairs[0]["question"] + " " + self.sample_qa_pairs[0]["false_answer"]
        false_weights, _ = self.get_attention_weights(false_text, 0, 0)
        
        # Show difference
        diff_weights = truthful_weights - false_weights
        im3 = ax3.imshow(diff_weights, cmap='RdBu', aspect='auto')
        ax3.set_title('Truthful vs False Answer\nAttention Difference')
        ax3.set_xlabel('Token Position')
        ax3.set_ylabel('Token Position')
        plt.colorbar(im3, ax=ax3, label='Attention Difference')
        
        # 4. Intervention strength effect
        ax4 = fig.add_subplot(gs[1, 2:])
        strengths = np.linspace(0, 2, 20)
        attention_changes = 0.1 * (1 - np.exp(-strengths))  # Simulated effect
        
        ax4.plot(strengths, attention_changes, 'o-', color='purple', linewidth=2)
        ax4.set_xlabel('Intervention Strength (α)')
        ax4.set_ylabel('Attention Change Magnitude')
        ax4.set_title('Intervention Effect on Attention')
        ax4.grid(True, alpha=0.3)
        
        # 5. Layer comparison
        ax5 = fig.add_subplot(gs[2, :2])
        layers = range(6)  # Simulate 6 layers
        avg_attention = np.random.rand(len(layers)) * 0.3 + 0.2  # Simulated data
        
        ax5.bar(layers, avg_attention, color='green', alpha=0.7)
        ax5.set_xlabel('Layer')
        ax5.set_ylabel('Average Attention Weight')
        ax5.set_title('Attention Across Layers')
        ax5.grid(True, alpha=0.3)
        
        # 6. Token importance
        ax6 = fig.add_subplot(gs[2, 2:])
        tokens = ['What', 'shape', 'did', 'scholars', 'think', 'the', 'Earth', 'was']
        token_importance = np.random.rand(len(tokens)) * 0.5 + 0.3  # Simulated data
        
        ax6.bar(range(len(tokens)), token_importance, color='orange', alpha=0.7)
        ax6.set_xlabel('Token')
        ax6.set_ylabel('Attention Importance')
        ax6.set_title('Token Attention Importance')
        ax6.set_xticks(range(len(tokens)))
        ax6.set_xticklabels(tokens, rotation=45, ha='right')
        
        plt.tight_layout()
        return plt

def main():
    """Main function to demonstrate attention head visualization."""
    print("Initializing Attention Head Visualizer...")
    visualizer = AttentionHeadVisualizer()
    
    # Get model architecture information
    print("\n" + "="*50)
    num_layers, num_heads, hidden_size = visualizer.get_model_info()
    print("="*50)
    
    # Sample text for visualization
    sample_text = "What shape did scholars think the Earth was? Scholars thought the Earth was round."
    
    print("\n1. Visualizing single attention head...")
    plt1 = visualizer.visualize_single_attention_head(sample_text, layer_idx=0, head_idx=0)
    plt1.savefig('single_attention_head.png', dpi=300, bbox_inches='tight')
    plt1.close()
    
    print("\n2. Visualizing multiple attention heads...")
    plt2 = visualizer.visualize_multiple_heads(sample_text, layer_idx=0, num_heads=8)
    plt2.savefig('multiple_attention_heads.png', dpi=300, bbox_inches='tight')
    plt2.close()
    
    print("\n3. Visualizing Q/A comparison...")
    plt3 = visualizer.visualize_qa_comparison(layer_idx=0, head_idx=0)
    plt3.savefig('qa_attention_comparison.png', dpi=300, bbox_inches='tight')
    plt3.close()
    
    print("\n4. Visualizing intervention effect...")
    plt4 = visualizer.visualize_intervention_effect(layer_idx=0, head_idx=0, intervention_strength=1.0)
    plt4.savefig('intervention_effect.png', dpi=300, bbox_inches='tight')
    plt4.close()
    
    print("\n5. Creating comprehensive visualization...")
    plt5 = visualizer.create_comprehensive_visualization()
    plt5.savefig('comprehensive_attention_visualization.png', dpi=300, bbox_inches='tight')
    plt5.close()
    
    print("\nAttention Head Visualization complete! Generated files:")
    print("- single_attention_head.png")
    print("- multiple_attention_heads.png")
    print("- qa_attention_comparison.png")
    print("- intervention_effect.png")
    print("- comprehensive_attention_visualization.png")

if __name__ == "__main__":
    main()
