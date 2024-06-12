import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
import os
from tqdm import tqdm
import unicodedata
import string

# Define the path to the BERT model
model_path = '/root/lanyun-tmp/duhaixing/bert-base-uncased/bert-base-uncased'
print(f"Current working directory: {os.getcwd()}")

class FullnameDataset(Dataset):
    def __init__(self, fullnames, labels, tokenizer, max_length):
        self.fullnames = fullnames
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.fullnames)

    def __getitem__(self, idx):
        fullname = self.fullnames[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            fullname,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def unicodeToAscii(s):
    all_letters = string.ascii_letters + " .,;'-"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) 
        if unicodedata.category(c) != 'Mn'  
        and c in all_letters 
    )

def load_fullnames(male_file, female_file, andy_file, chunksize=100000):
    data = {'fullname': [], 'label': []}
    total_processed = 0

    # Read female_file fullnames file
    for chunk in pd.read_csv(female_file, header=None, chunksize=chunksize):
        chunk[0] = chunk[0].apply(unicodeToAscii)
        data['fullname'].extend(chunk[0].tolist())
        data['label'].extend([0] * len(chunk))
        total_processed += len(chunk)
        print(f'Processed {total_processed} records from female_file fullnames')

    # Read male_file fullnames file
    for chunk in pd.read_csv(male_file, header=None, chunksize=chunksize):
        chunk[0] = chunk[0].apply(unicodeToAscii)
        data['fullname'].extend(chunk[0].tolist())
        data['label'].extend([1] * len(chunk))
        total_processed += len(chunk)
        print(f'Processed {total_processed} records from male_file fullnames')


    # Read andy_file fullnames file
    for chunk in pd.read_csv(andy_file, header=None, chunksize=chunksize):
        chunk[0] = chunk[0].apply(unicodeToAscii)
        data['fullname'].extend(chunk[0].tolist())
        data['label'].extend([2] * len(chunk))
        total_processed += len(chunk)
        print(f'Processed {total_processed} records from andy_file fullnames')

    print(f'Total processed records: {total_processed}')

    return data

def get_dataloaders(data, test_size=0.2, batch_size=256, max_length=15, num_workers=1):
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    df = pd.DataFrame(data)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    X_train, X_test, y_train, y_test = train_test_split(df['fullname'], df['label'], test_size=test_size, random_state=42)
    print('Creating FullnameDataset instances')
    train_dataset = FullnameDataset(X_train.tolist(), y_train.tolist(), tokenizer, max_length)
    test_dataset = FullnameDataset(X_test.tolist(), y_test.tolist(), tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader
