import os
import torch
from transformers import Trainer, TrainingArguments, TrainerCallback, BertTokenizer
from model import get_model
from data import get_dataloaders, load_fullnames
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import pandas as pd
import ipdb

os.chdir(os.path.dirname(__file__))

# File paths
ea_file = "./fullnames/ea.txt"
non_ea_file = "./fullnames/non_ea.txt"

# Load data
data = load_fullnames(ea_file, non_ea_file)

# Get data loaders
train_loader, test_loader = get_dataloaders(data, max_length=20)

# 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").to(device)  
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Define metrics computation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Define paths and directories
root_dir = './save_files'
output_dir = os.path.join(root_dir, 'save_files')
log_dir = os.path.join(root_dir, 'logs')
print(log_dir)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    learning_rate=2e-6,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    num_train_epochs=11,  # Number of training epochs
    logging_dir=log_dir,  # Log directory
    logging_steps=5000,  # Log every 5000 steps
    eval_steps=5000,  # Evaluate every 5000 steps
    save_steps=5000,  # Save model every 5000 steps
    save_total_limit=2,  # Only keep the last two models
)

# Logging configuration
logging.basicConfig(filename=os.path.join(log_dir, 'training.log'),
                    level=logging.INFO,
                    format='%(asctime)s %(message)s')

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % 500 == 0:
            logging.info(f"Step: {state.global_step}, Accuracy: {logs.get('eval_accuracy', 'N/A')}")

# Initialize model
model = get_model(num_labels=2).to(device)

# Create custom Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_loader.dataset,
    eval_dataset=test_loader.dataset,
    compute_metrics=compute_metrics,
    callbacks=[LoggingCallback()]
)

# Start training
trainer.train()

# Save model
save_directory = os.path.join(output_dir, 'final_checkpoint')
trainer.save_model(save_directory)

# Evaluate model
results = trainer.evaluate()
print(results)
logging.info(results)
