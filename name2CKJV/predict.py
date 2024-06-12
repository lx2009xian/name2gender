import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Model and tokenizer path
model_path = './final_checkpoint'

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# List of new full names
fullnames = [""]

# Tokenize and encode the new full names
new_encodings = tokenizer(fullnames, max_length=15, padding='max_length', truncation=True, return_tensors='pt')
new_encodings = {key: tensor.to(device) for key, tensor in new_encodings.items()}

# Set the model to evaluation mode
model.eval()
with torch.no_grad():
    outputs = model(**new_encodings)
    predictions = outputs.logits.argmax(-1)

# Print the prediction results
for fullname, prediction in zip(fullnames, predictions):
    if prediction == 1:
        print(f"{fullname}: ea")
    else:
        print(f"{fullname}: non_ea")