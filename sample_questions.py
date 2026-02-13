from datasets import load_dataset
import random
import json

# Load AHB
print("Loading AHB datset...")
dataset = load_dataset("sentientfutures/ahb", "questions", verification_mode="no_checks")
train_data = dataset['train']

# Calculate 5% of the total number of questions
numQuestions = len(train_data)
sample_size = int(numQuestions * 0.05)

print(f"\nTotal questions: {numQuestions}")
print(f"5% sample size: {sample_size}")

# Get random samples
random.seed(42)
englishQuestions = 25
# the entire question list
all_indices = list(range(englishQuestions))
# all_indices = list(range(numQuestions))
# randomly reorders all elements in the list in-place
random.shuffle(all_indices)

# Split into three 5% samples (for 2-turn, 3-turn, 4-turn)
two_turn_indices = all_indices[:sample_size] # the first 5
three_turn_indices = all_indices[sample_size:sample_size*2] # 5-10
four_turn_indices = all_indices[sample_size*2:sample_size*3] #10-15

print(train_data)
# Get actual questions - e.g. get the question at index i
two_turn_questions = [dict(train_data[i]) for i in two_turn_indices]
three_turn_questions = [dict(train_data[i]) for i in three_turn_indices]
four_turn_questions = [dict(train_data[i]) for i in four_turn_indices]

# print(two_turn_questions)
# print(f"\n2-turn sample: {len(two_turn_questions)} questions")
# print(f"3-turn sample: {len(three_turn_questions)} questions")
# print(f"4-turn sample: {len(four_turn_questions)} questions")

# save samples into dict
samples = {
    "2_turn": two_turn_questions,
    "3_turn": three_turn_questions,
    "4_turn": four_turn_questions
}

with open('samples.json', 'w') as f:
    json.dump(samples, f, indent=2)

print(f"\n✓ Saved samples to samples.json")

# Show examples from each
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