from datasets import load_dataset

# 加载你的数据
ds = load_dataset("parquet", data_files="data/Maxwell-Jia/AIME_2024/aime_2024_unified.parquet")
print("Dataset columns:", ds["train"].column_names)
print("\nFirst row:")
print(len(ds["train"]))
