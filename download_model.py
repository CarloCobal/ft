# download_model.py
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define the directory to save the model and tokenizer
home_dir = os.path.expanduser("~")
model_dir = os.path.join(home_dir, "my_model_directory")

# Create the directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-7b')
model = AutoModelForSequenceClassification.from_pretrained('tiiuae/falcon-7b', num_labels=1, trust_remote_code=True)

# Define the paths to save the tokenizer and model
tokenizer_path = os.path.join(model_dir, "my_tokenizer")
model_path = os.path.join(model_dir, "my_model")


print(f'saving model to {model_path=}')
# Save the tokenizer and model
tokenizer.save_pretrained(tokenizer_path)
model.save_pretrained(model_path)
print("done!")
