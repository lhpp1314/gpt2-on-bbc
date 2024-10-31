from datasets import load_dataset

# 加载 Hugging Face 上的公开数据集
dataset = load_dataset("glue", "mrpc")


# 将数据集保存到本地指定路径
dataset.save_to_disk("glue-mrpc")