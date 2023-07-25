# main2.py
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
import argparse
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import load_metric
from transformers import DataCollatorWithPadding
import os
import torch

# ... the rest of your code ...
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1, metavar='N',
                    help='Local rank for distributed training (-1: not distributed)')
args = parser.parse_args()

# Define the directory to load the model and tokenizer
home_dir = os.path.expanduser("~")
model_dir = os.path.join(home_dir, "my_model_directory")

# Define the paths to load the tokenizer and model
tokenizer_path = os.path.join(model_dir, "my_tokenizer")
model_path = os.path.join(model_dir, "my_model")

# Load the tokenizer, model, and tokenized datasets from disk
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

# Here is where you would add the new code
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = torch.nn.DataParallel(model)

# Load the tokenized datasets from disk
tokenized_datasets = load_from_disk("./tokenized_datasets")

# Create TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,  # increased from 16
    per_device_eval_batch_size=128,  # increased from 64
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    # Add these lines for distributed training
    local_rank=args.local_rank,
    fp16=True,
)

# The rest of your code...

# Define a compute metrics function
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],  # use the train split of the tokenized dataset
    eval_dataset=tokenized_datasets["test"],  # use the test split of the tokenized dataset
    # Add this line for distributed training
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save model using Trainer
trainer.save_model("./my_model")

# Save the tokenizer
tokenizer.save_pretrained("./my_tokenizer")
