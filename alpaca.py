import pandas as pd

df_raw = pd.read_parquet("hf://datasets/tatsu-lab/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")

df = pd.DataFrame()

df["prompt"] = df_raw["instruction"]
print(df.head())
df.to_csv("dataset/base/alpaca_prompts.csv", index=False)