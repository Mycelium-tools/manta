from datasets import load_dataset
import random
import json

# Load MANTA questions from HuggingFace
print("Loading MANTA questions from HuggingFace...")
dataset = load_dataset("mycelium-ai/manta-questions", data_files="manta_questions.csv")
train_data = dataset['train']

print(f"\nTotal questions: {len(train_data)}")

# Shuffle all questions randomly
random.seed(42)
all_questions = [dict(train_data[i]) for i in range(len(train_data))]
random.shuffle(all_questions)

# Split into three roughly equal groups (2-turn, 3-turn, 4-turn)
n = len(all_questions)
chunk = n // 3
two_turn_questions = all_questions[:chunk]
three_turn_questions = all_questions[chunk:chunk*2]
four_turn_questions = all_questions[chunk*2:]

print(f"2-turn: {len(two_turn_questions)} questions")
print(f"3-turn: {len(three_turn_questions)} questions")
print(f"4-turn: {len(four_turn_questions)} questions")

# Save to samples.json
samples = {
    "2_turn": two_turn_questions,
    "3_turn": three_turn_questions,
    "4_turn": four_turn_questions
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

print(f"\n{'='*60}")
print("EXAMPLE 4-TURN QUESTION:")
print(f"{'='*60}")
print(four_turn_questions[0]['question'])
