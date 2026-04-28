"""
Sample Questions
=============================
Loads questions from HuggingFace, parses tags from CSV string repr to lists,
drops the legacy variables column if present, and outputs all questions as a
flat list for use by the eval pipeline (all questions run as 3-turn).

Called automatically by sync_questions_to_hf.py after each sync.
Output: samples.json

Usage:
    python sample_questions.py
"""

from datasets import load_dataset
import json
import ast

TAG_ALIASES = {
    "Value Stability": "Animal Welfare Value Stability",
    "Contextual Welfare Salience": "Animal Welfare Moral Sensitivity",
}

KNOWN_TAGS = set(TAG_ALIASES) | set(TAG_ALIASES.values())

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
    if not isinstance(tags_val, str):
        return []
    try:
        result = ast.literal_eval(tags_val)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []

def normalize_tags(tags: list[str]) -> list[str]:
    """Map legacy tag names to the current AWMS/AWVS names."""
    return [TAG_ALIASES.get(tag, tag) for tag in tags]

def extract_tags(row: dict) -> list[str]:
    """Read tags, including the malformed HF column where the tag list became a header."""
    tags = normalize_tags(parse_tags(row.get('tags')))
    if tags:
        return tags

    for key, value in row.items():
        key_tags = normalize_tags(parse_tags(key))
        value_tags = normalize_tags(parse_tags(value))
        if any(tag in KNOWN_TAGS for tag in key_tags):
            return key_tags
        if any(tag in KNOWN_TAGS for tag in value_tags):
            return value_tags
    return []

all_questions = []
for i in range(len(train_data)):
    row = dict(train_data[i])
    row['tags'] = extract_tags(row)
    for key in list(row.keys()):
        if key != 'tags' and any(tag in KNOWN_TAGS for tag in normalize_tags(parse_tags(key))):
            row.pop(key, None)
    # Normalize pressure: parse list repr (e.g. "['pragmatic']") then lowercase
    if row.get('pressure'):
        parsed = parse_tags(row['pressure'])
        row['pressure'] = [p.strip().lower() for p in parsed] if parsed else []
    # Drop variables if still present (column removed from sheet)
    row.pop('variables', None)
    # Parse animals column: comma-separated string → list (empty list if blank)
    raw_animals = row.get('animals', '') or ''
    row['animals'] = [a.strip() for a in raw_animals.split(',') if a.strip()]
    row['reference_answer'] = (row.get('reference_answer') or '').strip()
    all_questions.append(row)

# All questions run as 3-turn; old keys kept for backwards compatibility
print(f"Total: {len(all_questions)} questions (all 3-turn)")

samples = {
    "all": all_questions,
    "2_turn": all_questions,  # deprecated, kept for compat
    "3_turn": all_questions,  # deprecated, kept for compat
}

with open('samples.json', 'w') as f:
    json.dump(samples, f, indent=2)

print("\nSaved samples to samples.json")

# Show one example
print(f"\n{'='*60}")
print("EXAMPLE QUESTION:")
print(f"{'='*60}")
print(all_questions[0]['question'])
