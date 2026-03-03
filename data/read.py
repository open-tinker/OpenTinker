from datasets import load_dataset

# 加载你的数据
# ds = load_dataset("parquet", data_files="data/retool_dapo/retool_dapo_train.dedup.parquet")
ds = load_dataset("parquet", data_files="data/aime2024/train.parquet")
print("Dataset columns:", ds["train"].column_names)
print("\nFirst row:")
print(ds["train"][0])
print(len(ds["train"]))
