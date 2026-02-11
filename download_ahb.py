# datasets: library from HuggingFace for loading and using datasets.
from datasets import load_dataset
import pandas as pd

# Download AHB - specify "questions" config
print("Downloading AHB dataset...")
# dataset is dict
# Use verification_mode="no_checks" to ignore cached metadata
dataset = load_dataset("sentientfutures/ahb", "questions", verification_mode="no_checks")
# train_data is an object containing all training questions
train_data = dataset["train"]

# Show what we got
print(f"\n✓ Downloaded {len(train_data)} questions")
print(f"\nAvailable fields: {train_data.column_names}")

# Show first question
print(f"\nFirst question example:")
for key in train_data[0].keys():
    print(f"  {key}: {train_data[0][key]}")
    
# Convert to pandas for easier viewing
df = pd.DataFrame(train_data)

# Save to CSV
df.to_csv('ahb_questions.csv', index=False)
print(f"\n✓ Saved {len(df)} questions to ahb_questions.csv")

# Show summary
# print(f"\nDataset shape: {df.shape}")
# print(f"Columns: {list(df.columns)}")

# Show a few example questions
# print(f"\nFirst 5 questions:")
# for i in range(min(5, len(df))):
#     print(f"\n{i+1}. {df['input'].iloc[i][:100]}...")