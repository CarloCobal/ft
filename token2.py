# tokenize_datasets.py
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Load the dataset
data = pd.read_csv('processed_data.csv')

# Create a new 'question' column
data['question'] = 'What is the yield?'

# Convert the 'Yield' column to string
data['Yield'] = data['Yield'].astype(str)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2)

# Convert the dataframes to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# Load the tokenizer from disk
home_dir = os.path.expanduser("~")
tokenizer_path = os.path.join(home_dir, "my_model_directory/my_tokenizer")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Set padding token if it's not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Replace None values with a default string
    questions = ['' if v is None else v for v in examples['question']]
    return tokenizer(questions, padding='max_length', truncation=True)

# Tokenize the datasets
tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
tokenized_test_datasets = test_dataset.map(tokenize_function, batched=True)

# Save the tokenized datasets to disk
tokenized_datasets.save_to_disk("./tokenized_datasets")
tokenized_test_datasets.save_to_disk("./tokenized_test_datasets")
