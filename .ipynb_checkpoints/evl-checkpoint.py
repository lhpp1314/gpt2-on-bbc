import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from datasets import load_dataset
from datasets import load_from_disk

model_path = 'output/gpt2_train.pth'
model = GPT2LMHeadModel.from_pretrained('local-gpt2-model')

model.load_state_dict(torch.load(model_path))

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
# Set the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('local-gpt2-tokenizer')

# Generate text
prompt = "London is cold today soccer game last night is fantasitic"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
output = model.generate(input_ids, max_length=100, num_return_sequences=1)

for i, generated in enumerate(output):
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"Generated text {i+1}: {text}")