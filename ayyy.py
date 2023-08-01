from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainerCallback
from tqdm import tqdm
import pandas as pd


class TqdmCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.epoch_pbar = tqdm(total=state.num_training_epochs, desc="Epochs", position=0)
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.step_pbar = tqdm(total=state.max_steps, desc=f"Epoch {state.epoch}", position=1)

    def on_step_end(self, args, state, control, **kwargs):
        self.step_pbar.update(1)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.step_pbar.close()
        self.epoch_pbar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        self.epoch_pbar.close()


# Load data
data_path = 'APY.csv'
dataset = load_dataset('csv', data_files=data_path)

# Load model and tokenizer
model_name = 'ybelkada/falcon-7b-sharded-bf16'
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, trust_remote_code=True)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


# Maximum length of tokenized sequences
max_seq_length = 512

def tokenize_function(examples):
    # Convert examples to string and replace any NaN or empty values
    examples = [str(e) if pd.notnull(e) else 'unknown' for e in examples['Crop']]
    return tokenizer(examples, padding="max_length", truncation=True, max_length=max_seq_length)

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainerCallback
from tqdm import tqdm
import pandas as pd


class TqdmCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.epoch_pbar = tqdm(total=state.num_training_epochs, desc="Epochs", position=0)
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.step_pbar = tqdm(total=state.max_steps, desc=f"Epoch {state.epoch}", position=1)

    def on_step_end(self, args, state, control, **kwargs):
        self.step_pbar.update(1)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.step_pbar.close()
        self.epoch_pbar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        self.epoch_pbar.close()


# Load data
data_path = 'APY.csv'
dataset = load_dataset('csv', data_files=data_path)

# Load model and tokenizer
model_name = 'ybelkada/falcon-7b-sharded-bf16'
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, trust_remote_code=True)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


# Maximum length of tokenized sequences
max_seq_length = 512

def tokenize_function(examples):
    # Convert examples to string and replace any NaN or empty values
    examples = [str(e) if pd.notnull(e) else 'unknown' for e in examples['Crop']]
    return tokenizer(examples, padding="max_length", truncation=True, max_length=max_seq_length)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=list(dataset['train'].column_names))

# LoRA configuration
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64
peft_config = LoraConfig(lora_alpha=lora_alpha, lora_dropout=lora_dropout, r=lora_r, bias='none',
                         task_type='CAUSAL_LM', target_modules=['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'])

# Training configuration
output_dir = './results'
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim='paged_adamw_32bit',
    save_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=500,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type='constant'
)

# Initialize trainer
max_seq_length = 512
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field='text',  # replace 'text' with the column that contains your text data
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments
)

# Convert all normalization layers to float32
for name, module in trainer.model.named_modules():
    if 'norm' in name:
        module = module.to(torch.float32)

# Train model
trainer.train()

# Save the model
model_path = '/'
trainer.save_model(model_path)

# Load the saved model for inference
model = AutoModelForCausalLM.from_pretrained(model_path)

# Prompt the model with a question
prompt = 'What is the average crop yield in 2020?'
input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids, max_length=100)
decoded_output = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
print(decoded_output)

# LoRA configuration
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64
peft_config = LoraConfig(lora_alpha=lora_alpha, lora_dropout=lora_dropout, r=lora_r, bias='none',
                         task_type='CAUSAL_LM', target_modules=['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'])

# Training configuration
output_dir = './results'
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim='paged_adamw_32bit',
    save_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=500,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type='constant'
)

# Initialize trainer
max_seq_length = 512
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field='text',  # replace 'text' with the column that contains your text data
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments
)

# Convert all normalization layers to float32
for name, module in trainer.model.named_modules():
    if 'norm' in name:
        module = module.to(torch.float32)

# Train model
trainer.train()

# Save the model
model_path = '/'
trainer.save_model(model_path)

# Load the saved model for inference
model = AutoModelForCausalLM.from_pretrained(model_path)

# Prompt the model with a question
prompt = 'What is the average crop yield in 2020?'
input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids, max_length=100)
decoded_output = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
print(decoded_output)
