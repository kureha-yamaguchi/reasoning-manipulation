import csv
import json
import random
import os

## alpaca_data_cleaned.json from https://github.com/gururise/AlpacaDataCleaned
dir = 'dataset/base/'
os.makedirs(dir, exist_ok=True)

# Step 1: Read the JSON file
with open(os.path.join(dir,'alpaca_data_cleaned.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

# Step 2: Extract all instructions
instructions = [item['instruction'] for item in data]

# Step 3: Randomly sample 100 instructions (or take the first 100 if you prefer)
# For random sample:
sampled_instructions = random.sample(instructions, 600)

# Step 4: Write to CSV

with open(os.path.join(dir, 'alpaca_instructions_600.csv'), 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['goal'])  # Header
    for instruction in sampled_instructions:
        writer.writerow([instruction])

print("CSV file created with 600 instructions from the alpaca dataset")