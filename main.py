from utils.loader import load_jsonl

DATA_PATH = "data/conversations_GPT-GPT.jsonl"

df = load_jsonl(DATA_PATH)

print(df.shape)
print(df.head())
print(df.dtypes)
