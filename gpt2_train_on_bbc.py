import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from datasets import load_dataset
from datasets import load_from_disk

# Load and preprocess the dataset
dataset = load_from_disk("glue-mrpc")

# Combine sentence1 and sentence2 into a single field for training
def combine_sentences(example):
    # Assuming example has keys 'sentence1' and 'sentence2'
    sentence1 = example['sentence1'] if 'sentence1' in example else ''
    sentence2 = example['sentence2'] if 'sentence2' in example else ''
    example['combined_text'] = f"{sentence1} {sentence2}".strip()  # Combine and strip whitespace
    return example

# Apply the combining function to the dataset
dataset = dataset.map(combine_sentences)

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('local-gpt2-tokenizer')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('local-gpt2-model')

# Tokenize and encode the dataset
def tokenize_function(example):
    return tokenizer(example["combined_text"], truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True)

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["input_ids"] for item in batch]

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)

    # Pad sequences to the same length
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }

# Prepare the data for training
train_dataset = tokenized_dataset["train"]
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Set up the training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)

for batch in train_dataloader:
    print(batch)
    break

# Training loop
model.train()
num_epochs = 100
for epoch in range(num_epochs):
    print("epoch:{}".format(epoch))
    for step, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["input_ids"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        if step % 100 == 0:
            print("Step-{}, Loss-{}".format(step, loss.item()))

        loss.backward()
        optimizer.step()

# Save the trained model
output_path = 'output/gpt2_train.pth'
torch.save(model.state_dict(), output_path)
