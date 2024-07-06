# # Importing necessary modules
# import spacy
# from spacy.lang.en.stop_words import STOP_WORDS
# from string import punctuation
# from heapq import nlargest

# # Function to perform extractive summarization
# def extractive_summarize(text):
#     # Load English language model
#     nlp = spacy.load('en_core_web_sm')

#     # Process the text
#     doc = nlp(text)

#     # Define stop words and punctuation
#     stopwords = list(STOP_WORDS)
#     punctuation = punctuation + '\n'

#     # Calculate word frequencies
#     word_frequencies = {}
#     for word in doc:
#         if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
#             if word.text not in word_frequencies.keys():
#                 word_frequencies[word.text] = 1
#             else:
#                 word_frequencies[word.text] += 1

#     # Normalize word frequencies
#     max_frequency = max(word_frequencies.values())
#     for word in word_frequencies.keys():
#         word_frequencies[word] = word_frequencies[word] / max_frequency

#     # Calculate sentence scores based on word frequencies
#     sentence_scores = {}
#     for sent in doc.sents:
#         for word in sent:
#             if word.text.lower() in word_frequencies.keys():
#                 if sent not in sentence_scores.keys():
#                     sentence_scores[sent] = word_frequencies[word.text.lower()]
#                 else:
#                     sentence_scores[sent] += word_frequencies[word.text.lower()]

#     # Determine length of summary
#     select_length = int(len(list(doc.sents)) * 0.3)  # Selecting 30% of sentences for summary

#     # Generate summary using top scoring sentences
#     summary_sentences = nlargest(select_length, sentence_scores, key=sentence_scores.get)
#     summary = ' '.join([sent.text for sent in summary_sentences])

#     return summary

# # Uncomment and use the following for testing if needed
# # if __name__ == "__main__":
# #     sample_text = "Your sample text here."
# #     print(extractive_summarize(sample_text))
# abstractive_summarizer.py

# Install necessary libraries if not already installed
# !pip install transformers datasets evaluate py7zr rouge_score torch accelerate -U

# Import necessary modules
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
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

# Function to perform extractive summarization using SpaCy
def extractive_summarize(text):
    # Load English language model
    nlp = spacy.load('en_core_web_sm')

    # Process the text
    doc = nlp(text)

    # Define stop words and punctuation
    stopwords = list(STOP_WORDS)
    punctuation = punctuation + '\n'

    # Calculate word frequencies
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

    # Normalize word frequencies
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    # Calculate sentence scores based on word frequencies
    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    # Determine length of summary
    select_length = int(len(list(doc.sents)) * 0.3)  # Selecting 30% of sentences for summary

    # Generate summary using top scoring sentences
    summary_sentences = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    summary = ' '.join([sent.text for sent in summary_sentences])

    return summary

# Uncomment and use the following for testing if needed
# if __name__ == "__main__":
#     sample_text = "Your sample text here."
#     print(extractive_summarize(sample_text))
