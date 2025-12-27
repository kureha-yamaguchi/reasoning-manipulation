from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Convert to pandas and filter
df = dataset.to_pandas()
filtered = df[df['input'].isna() | (df['input'] == '')]

# Randomly sample 1/20 of the rows
sampled = filtered.sample(frac=0.05, random_state=42)

# Remove newlines from instructions
sampled['prompt'] = sampled['instruction'].str.replace('\n', ' ', regex=False)
new_df = sampled[['prompt']]
print(len(new_df))
# Save just the instructions to CSV
new_df.to_csv('all_harmless_prompts.csv', index=False)