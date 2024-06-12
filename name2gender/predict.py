import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import pickle
from tqdm import tqdm
import unicodedata
import string
import pandas as pd

# Paths for model and tokenizer
model_path = './final_checkpoint'
tokenizer_path = './bert-base-uncased/bert-base-uncased'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model.to(device)
model.eval()

# Function to convert full names to ASCII
all_letters = string.ascii_letters + " .,;'-"

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def process_file(file_path, root_dir, output_base_path):
    # Load the file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Extract all unique full names and clean them
    all_fullnames = list(set(unicodeToAscii(name) for names in data.values() for name in names))
    predictions = {}

    # Process full names in batches to avoid memory overflow
    batch_size = 5000
    for i in tqdm(range(0, len(all_fullnames), batch_size), desc=f"Processing {os.path.basename(file_path)}"):
        batch_fullnames = all_fullnames[i:i+batch_size]
        new_encodings = tokenizer(batch_fullnames, max_length=15, padding='max_length', truncation=True, return_tensors='pt')
        new_encodings = {key: tensor.to(device) for key, tensor in new_encodings.items()}

        with torch.no_grad():
            outputs = model(**new_encodings)
            batch_predictions = outputs.logits.argmax(-1)

        for fullname, prediction in zip(batch_fullnames, batch_predictions):
            if prediction == 1:
                predictions[fullname] = 'Male'
            elif prediction == 0:
                predictions[fullname] = 'Female'
            else:
                predictions[fullname] = 'Andy'

    # Create a new dictionary to save predictions
    new_data = {article_id: [{'fullname': fullname, 'prediction': predictions[unicodeToAscii(fullname)]} for fullname in fullnames] for article_id, fullnames in data.items()}

    # Create output path
    subdir = os.path.relpath(os.path.dirname(file_path), root_dir)
    output_dir = os.path.join(output_base_path, subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the new pickle file
    output_pickle_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_with_predictions.pkl")
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(new_data, f)
    print(f"Predictions saved to: {output_pickle_path}")

    # Save different categories to CSV files
    categories = ['Male', 'Female', 'Andy']
    for category in categories:
        category_data = []
        for article_id, authors in new_data.items():
            for author in authors:
                if author['prediction'] == category:
                    category_data.append({'id': article_id, 'fullname': author['fullname'], 'prediction': author['prediction']})

        df = pd.DataFrame(category_data)

        # Name CSV files based on the original file name
        csv_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_{category}_authors.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f'{category} category data saved to: {csv_path}')

# List of directories to process
directories = [
    './Biology',
    './Computer science',
    './Economics',
]

output_base_path = './result'

if not os.path.exists(output_base_path):
    os.makedirs(output_base_path)

for directory in directories:
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pickle'):
                file_path = os.path.join(root, file)
                process_file(file_path, directory, output_base_path)

print("All files have been processed.")
