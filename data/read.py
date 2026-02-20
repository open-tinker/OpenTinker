from datasets import load_dataset

# 加载你的数据
ds = load_dataset("parquet", data_files="./data/math/test.parquet")
print("Dataset columns:", ds["train"].column_names)
print("\nFirst row:")
print(ds["train"][0])
