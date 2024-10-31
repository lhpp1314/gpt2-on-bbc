from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# 本地保存路径
tokenizer_dir = "local-gpt2-tokenizer"
model_dir = "local-gpt2-model"

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载 GPT2 模型和 tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 指定保存路径
save_directory1 = "./local-gpt2-model"
save_directory2 = "./local-gpt2-tokenizer"

# 将模型和 tokenizer 保存到本地指定路径
model.save_pretrained(save_directory1)
tokenizer.save_pretrained(save_directory2)

print(f"Model and tokenizer saved ")
