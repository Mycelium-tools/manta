"""
Sample Questions
=============================
Loads questions from HuggingFace, parses tags from CSV string repr to lists,
drops the legacy variables column if present, and splits questions into
2-turn and 3-turn groups (in order) for use by the eval pipeline.

Called automatically by dataset/sync_questions_to_hf.py after each sync.
Output: samples.json

Usage:
    python sample_questions.py
"""

from datasets import load_dataset
import json
import ast

# Load MANTA questions from HuggingFace
print("Loading MANTA questions from HuggingFace...")
# revision= should be pinned to a specific commit SHA for reproducibility;
# using "main" here as a minimum — replace with a commit SHA once the dataset is stable.
dataset = load_dataset("mycelium-ai/manta-questions", data_files="manta_questions.csv", revision="main")
train_data = dataset['train']

print(f"\nTotal questions: {len(train_data)}")

def parse_tags(tags_val) -> list[str]:
    """Parse tags from CSV string repr (e.g. "['pressure_robustness']") to a list."""
    if not tags_val:
        return []
    if isinstance(tags_val, list):
        return tags_val
    try:
        result = ast.literal_eval(tags_val)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []

all_questions = []
for i in range(len(train_data)):
    row = dict(train_data[i])
    row['tags'] = parse_tags(row.get('tags'))
    # Drop variables if still present (column removed from sheet)
    row.pop('variables', None)
    all_questions.append(row)

# Split into two roughly equal groups (2-turn, 3-turn) in order
n = len(all_questions)
chunk = n // 2
two_turn_questions = all_questions[:chunk]
three_turn_questions = all_questions[chunk:]

print(f"2-turn: {len(two_turn_questions)} questions")
print(f"3-turn: {len(three_turn_questions)} questions")

# Save to samples.json
samples = {
    "2_turn": two_turn_questions,
    "3_turn": three_turn_questions,
}

with open('samples.json', 'w') as f:
    json.dump(samples, f, indent=2)

print(f"\n✓ Saved samples to samples.json")

# Show one example from each group
print(f"\n{'='*60}")
print("EXAMPLE 2-TURN QUESTION:")
print(f"{'='*60}")
print(two_turn_questions[0]['question'])

print(f"\n{'='*60}")
print("EXAMPLE 3-TURN QUESTION:")
print(f"{'='*60}")
print(three_turn_questions[0]['question'])
