from datasets import load_dataset

def main():
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    filtered = dataset.filter(lambda x: x["input"] is None or x["input"] == "")
    filtered = filtered.select(range(min(250, len(filtered))))
    filtered = filtered.select_columns(["instruction"])
    filtered = filtered.rename_column("instruction", "prompt")
    
    # Convert to Pandas and save
    df = filtered.to_pandas()
    
    print(f"Number of rows: {len(df)}")
    
    df.to_csv("dataset/test_harmless_prompts.csv", index=False)

if __name__ == "__main__":
    main()