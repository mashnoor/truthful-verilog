# Attention Head Visualization for Truthfulness Experiments

This project provides a simple visualization tool for analyzing attention patterns in transformer models, specifically designed for research on truthfulness in code generation tasks.

## Features

- **Single Head Visualization**: Visualize attention patterns for individual attention heads
- **Multiple Heads Grid**: Compare multiple attention heads in a single view
- **Verilog-Specific Analysis**: Specialized analysis for Verilog code with highlighted key tokens
- **Implementation Comparison**: Compare attention patterns between accurate and buggy code implementations

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main visualization script:
```bash
python visalize.py
```

This will generate several visualization files:
- `attention_simple_text.png`: Basic attention visualization
- `attention_multiple_heads.png`: Grid of multiple attention heads
- `attention_verilog_analysis.png`: Verilog-specific analysis (if data available)
- `attention_comparison.png`: Comparison between accurate and buggy implementations

### Programmatic Usage

```python
from visalize import AttentionVisualizer

# Initialize the visualizer
visualizer = AttentionVisualizer()

# Visualize a single attention head
plt = visualizer.visualize_attention_head(
    "Design a 2-input AND gate in Verilog.", 
    layer_idx=0, 
    head_idx=0
)
plt.show()

# Visualize multiple heads
plt = visualizer.visualize_multiple_heads(
    "Design a 2-input AND gate in Verilog.", 
    layer_idx=0, 
    num_heads=8
)
plt.show()

# Analyze Verilog code specifically
verilog_code = "module and_gate(input a, b, output y); assign y = a & b; endmodule"
plt = visualizer.analyze_verilog_attention(verilog_code, layer_idx=0, head_idx=0)
plt.show()
```

## Research Context

This visualization tool is designed for analyzing attention patterns in transformer models during code generation tasks, particularly for:

1. **Truthfulness Analysis**: Understanding how models attend to different parts of code when generating accurate vs. buggy implementations
2. **Verilog Code Generation**: Specialized analysis for hardware description language code
3. **Attention Pattern Comparison**: Comparing attention patterns between different types of implementations

## Data Format

The tool expects Verilog implementation data in JSON format with the following structure:
```json
[
  {
    "difficulty": "easy",
    "generated_instruction": "instruction text",
    "accurate_implementation": "verilog code",
    "slightly_different_implementation": "verilog code", 
    "buggy_implementation": "verilog code"
  }
]
```

## Customization

You can customize the visualizations by:
- Changing the model (default: GPT-2)
- Adjusting layer and head indices
- Modifying color schemes and figure sizes
- Adding custom token highlighting for specific domains

## Requirements

- Python 3.7+
- PyTorch
- Transformers library
- Matplotlib
- Seaborn
- NumPy

## License

This project is part of truthfulness experiments research. 