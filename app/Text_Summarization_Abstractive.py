# abstractive_summarizer.py

# Install necessary libraries if not already installed
# !pip install transformers datasets evaluate py7zr rouge_score torch accelerate -U

# Import required libraries
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from rouge_score import rouge_scorer
import torch
import zipfile
import os

# Load SAMSum dataset
dataset = load_dataset('samsum')

# Load T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Define preprocessing function for dataset
def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=150, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing to dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define function to summarize text using T5 model
def summarize_text(text, model, tokenizer, max_length=150, min_length=40, num_beams=4):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, num_beams=num_beams, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Define custom dataset class for summarization
class SummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, dialogues, summaries, tokenizer, max_length=512):
        self.dialogues = dialogues
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        summary = self.summaries[idx]
        inputs = self.tokenizer.encode_plus(
            "summarize: " + dialogue,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        targets = self.tokenizer.encode_plus(
            summary,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten()
        }

# Prepare train and validation datasets
train_dataset = SummarizationDataset(dataset['train']['dialogue'], dataset['train']['summary'], tokenizer)
validation_dataset = SummarizationDataset(dataset['validation']['dialogue'], dataset['validation']['summary'], tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    logging_dir='./logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    tokenizer=tokenizer,
)

# Train the model (uncomment if training is needed)
# trainer.train()

# Function to tokenize inputs
def tokenize_inputs(dialogues, tokenizer, max_length=512, device='cuda'):
    inputs = tokenizer(dialogues, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs

# Function to evaluate model using ROUGE metric
def evaluate_model(model, tokenizer, dataset, device='cuda', num_samples=100):
    dialogues = dataset['validation'][:num_samples]['dialogue']
    references = dataset['validation'][:num_samples]['summary']

    inputs = tokenize_inputs(dialogues, tokenizer, device=device)
    model.to(device)

    summaries = []
    for idx in range(len(inputs['input_ids'])):
        input_ids = inputs['input_ids'][idx].unsqueeze(0)
        attention_mask = inputs['attention_mask'][idx].unsqueeze(0)

        summary_ids = model.generate(input_ids.to(device),
                                     attention_mask=attention_mask.to(device),
                                     max_length=150,
                                     min_length=40,
                                     num_beams=4,
                                     early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    results = rouge_scorer.load_and_evaluate(eval_dataloader=None, predictions=summaries, references=references)
    return results

# Evaluate model and print results
# results_after_tuning = evaluate_model(model, tokenizer, dataset, device='cuda')
# print(results_after_tuning)

# Save model and tokenizer
# torch.save(model.state_dict(), 'model.pkl')
# tokenizer.save_pretrained('./tokenizer_directory')

# Zip model and tokenizer files
# with zipfile.ZipFile('model.zip', 'w') as zipf:
#     zipf.write('model.pkl')
#     for root, dirs, files in os.walk('tokenizer_directory'):
#         for file in files:
#             zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), 'tokenizer_directory'))

# Uncomment and use the following for testing
def abstractive_summarize(text):
    summary = summarize_text(text, model, tokenizer)
    return summary

# if __name__ == "__main__":
#     sample_text = "Your sample text here."
#     print(abstractive_summarize(sample_text))
