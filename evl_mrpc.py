import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse

# Argument parser to accept prompt from the command line
parser = argparse.ArgumentParser(description="Generate text using a fine-tuned GPT-2 model.")
parser.add_argument("prompt", type=str, help="Prompt text to start the text generation")
args = parser.parse_args()

# Define paths and load model
model_path = 'output/gpt2_train_mrpc.pth'
model = GPT2LMHeadModel.from_pretrained('local-gpt2-model')
model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Set the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('local-gpt2-tokenizer')

# Generate text
input_ids = tokenizer.encode(args.prompt, return_tensors='pt').to(device)
output = model.generate(input_ids, max_length=200, num_return_sequences=1)

# Print generated text
print(f"Prompt: {args.prompt}")
for i, generated in enumerate(output):
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"Generated text {i+1}: {text}")