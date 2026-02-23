from datasets import load_dataset
import json

# Load MANTA questions from HuggingFace
print("Loading MANTA questions from HuggingFace...")
dataset = load_dataset("mycelium-ai/manta-questions", data_files="manta_questions.csv")
train_data = dataset['train']

print(f"\nTotal questions: {len(train_data)}")

all_questions = [dict(train_data[i]) for i in range(len(train_data))]

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
