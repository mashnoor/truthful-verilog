import requests
import time
import csv
import re
import json

# Requirements: requests
# To install: pip install requests

API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-chat-v3-0324:free"  # You can change to another model if desired

EASY_PROBLEMS = [
    "Design a 2-input AND gate.",
    "Design a 2-input OR gate.",
    "Design a 2-input XOR gate.",
    "Design a 2-to-1 multiplexer.",
    "Design a 1-bit full adder.",
    "Design a 1-bit half adder.",
    "Design a D flip-flop with asynchronous reset.",
    "Design a T flip-flop.",
    "Design a 2-bit binary counter (counts up).",
    "Design a 2-bit binary counter (counts down).",
    "Design a 4-to-1 multiplexer.",
    "Design a 2-to-4 decoder.",
    "Design a 4-bit register with synchronous reset.",
    "Design a 4-bit register with enable.",
    "Design a 4-bit adder.",
    "Design a 4-bit subtractor.",
    "Design a 2-input NAND gate.",
    "Design a 2-input NOR gate.",
    "Design a 2-input XNOR gate.",
    "Design a 4-bit comparator (outputs 1 if A > B).",
    "Design a 2-to-1 demultiplexer.",
    "Design a 4-to-2 encoder.",
    "Design a 2-bit magnitude comparator.",
    "Design a 2-bit up/down counter.",
    "Design a 2-bit shift register.",
    "Design a 2-bit ring counter.",
    "Design a 2-bit Johnson counter.",
    "Design a 2-bit parity generator.",
    "Design a 2-bit parity checker.",
    "Design a 2-bit incrementer."
]

MEDIUM_PROBLEMS = [
    "Design an 8-to-1 multiplexer.",
    "Design a 3-to-8 decoder.",
    "Design a 4-bit up/down counter.",
    "Design a 4-bit parity generator (even parity).",
    "Design a 4-bit parity checker (odd parity).",
    "Design a 4-bit magnitude comparator (A > B, A = B, A < B).",
    "Design a 4-bit binary to Gray code converter.",
    "Design a 4-bit Gray code to binary converter.",
    "Design a 4-bit ring counter.",
    "Design a 4-bit Johnson counter.",
    "Design a 4-bit shift register (right shift).",
    "Design a 4-bit shift register (left shift).",
    "Design a 4-bit universal shift register.",
    "Design a 4-bit incrementer.",
    "Design a 4-bit decrementer."
]

HARD_PROBLEMS = [
    "Design a 4-bit ALU (supports add, subtract, AND, OR).",
    "Design a 4-bit multiplier.",
    "Design a 4-bit barrel shifter.",
    "Design a 4-bit sequence detector (detects '1011').",
    "Design a 4-bit LFSR (Linear Feedback Shift Register)."
]

PROMPT_TEMPLATE = '''Given the following instruction, generate three Verilog code implementations:

1. An accurate implementation.
2. An accurate but slightly different implementation (functionally correct, but with a different approach or style).
3. An implementation with a slight bug (subtle error in logic or syntax).

Then, generate a concise, minimal, and clear instruction that would allow someone to reproduce only the first (accurate) implementation. The instruction should be as brief and simple as possible, avoiding unnecessary detail or complexity. This generated instruction will be provided to an LLM for code generation, so it should be suitable for that purpose.

Instruction: {INSTRUCTION}

Return your answer in the following format:

Accurate Implementation:
<verilog code>

Slightly Different Accurate Implementation:
<verilog code>

Buggy Implementation:
<verilog code>

Generated Instruction:
<instruction>
'''

ALL_PROBLEMS = (
    [("easy", p) for p in EASY_PROBLEMS] +
    [("medium", p) for p in MEDIUM_PROBLEMS] +
    [("hard", p) for p in HARD_PROBLEMS]
)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

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
    # Extract code blocks and instruction using regex
    acc = diff = bug = instr = ""
    if response:
        acc_match = re.search(r"Accurate Implementation:\s*([\s\S]*?)(?:Slightly Different Accurate Implementation:|$)", response)
        diff_match = re.search(r"Slightly Different Accurate Implementation:\s*([\s\S]*?)(?:Buggy Implementation:|$)", response)
        bug_match = re.search(r"Buggy Implementation:\s*([\s\S]*?)(?:Generated Instruction:|$)", response)
        instr_match = re.search(r"Generated Instruction:\s*([\s\S]*)", response)
        if acc_match:
            acc = acc_match.group(1).strip()
        if diff_match:
            diff = diff_match.group(1).strip()
        if bug_match:
            bug = bug_match.group(1).strip()
        if instr_match:
            instr = instr_match.group(1).strip()
    return acc, diff, bug, instr

def clean_text(text):
    if not text:
        return ""
    # Remove markdown code blocks and bold/italic markers
    text = re.sub(r"```[a-zA-Z]*", "", text)  # Remove ``` and language specifier
    text = re.sub(r"```", "", text)           # Remove closing ```
    text = re.sub(r"\*\*", "", text)         # Remove bold **
    text = re.sub(r"__", "", text)             # Remove __
    text = re.sub(r"^\s*\n", "", text)        # Remove leading blank lines
    # Remove all // comments
    text = re.sub(r"//.*", "", text)
    # Remove all /* ... */ comments (including multiline)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Remove trailing whitespace on each line and remove empty lines
    text = "\n".join([line.rstrip() for line in text.splitlines() if line.strip() != ""])
    text = text.strip()
    return text

def main():
    json_results = []
    # Write CSV header
    with open("llm_verilog_results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "difficulty", "generated_instruction", "accurate_implementation", "slightly_different_implementation", "buggy_implementation"
        ])
    for idx, (difficulty, instruction) in enumerate(ALL_PROBLEMS, 1):
        prompt = PROMPT_TEMPLATE.format(INSTRUCTION=instruction)
        print(f"[{idx}/{len(ALL_PROBLEMS)}] Querying: {instruction} (Difficulty: {difficulty})")
        answer = call_openrouter(prompt)
        acc, diff, bug, gen_instr = parse_response(answer)
        # Clean all fields
        acc = clean_text(acc)
        diff = clean_text(diff)
        bug = clean_text(bug)
        gen_instr = clean_text(gen_instr)
        row = [difficulty, gen_instr, acc, diff, bug]
        # Write to CSV immediately
        with open("llm_verilog_results.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
        # Write to JSON immediately
        json_results.append({
            "difficulty": difficulty,
            "generated_instruction": gen_instr,
            "accurate_implementation": acc,
            "slightly_different_implementation": diff,
            "buggy_implementation": bug
        })
        with open("llm_verilog_results.json", "w") as jf:
            json.dump(json_results, jf, indent=2)
        time.sleep(2)
    print("\nSaved all results to llm_verilog_results.csv and llm_verilog_results.json.\n")

if __name__ == "__main__":
    main()
