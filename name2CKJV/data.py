import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import glob
import unicodedata
import string
import os
import ipdb
from tqdm import tqdm

model_path = '/root/lanyun-tmp/duhaixing/bert-base-uncased/bert-base-uncased'

print(f"Current working directory: {os.getcwd()}")

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)

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


def findFiles(path): 
    return glob.glob(path)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) 
        if unicodedata.category(c) != 'Mn'  
        and c in all_letters 
    )

def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def load_fullnames(ea_file, non_ea_file):
    data = {'fullname': [], 'label': []}
    
    ea_fullnames = readLines(ea_file)
    data['fullname'].extend(ea_fullnames)
    data['label'].extend([1] * len(ea_fullnames))
    
    non_ea_fullnames = readLines(non_ea_file)
    data['fullname'].extend(non_ea_fullnames)
    data['label'].extend([0] * len(non_ea_fullnames))
    
    return data


def batch_tokenize_fullnames(tokenizer, fullnames, batch_size, max_length, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"Loading tokenized data from {save_path}")
        tokenized_data = torch.load(save_path)
        print(f"{save_path} ok!")
        return tokenized_data['input_ids'], tokenized_data['attention_masks']

    all_input_ids = []
    all_attention_masks = []
    for i in tqdm(range(0, len(fullnames), batch_size), desc="Tokenizing"):
        batch_fullnames = fullnames[i:i + batch_size]
        encodings = tokenizer(batch_fullnames, padding='max_length',truncation=True, return_tensors='pt', max_length=max_length)
        all_input_ids.append(encodings['input_ids'])
        all_attention_masks.append(encodings['attention_mask'])

    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_attention_masks = torch.cat(all_attention_masks, dim=0)
    
    torch.save({'input_ids': all_input_ids, 'attention_masks': all_attention_masks}, save_path)
    print(f"Tokenized data saved to {save_path}")
    
    return all_input_ids, all_attention_masks

def batch_resample(X, y, batch_size, resampler):
    resampled_X = []
    resampled_y = []
    for i in tqdm(range(0, len(X), batch_size), desc="Resampling"):
        batch_X = X[i:i + batch_size]
        batch_y = y[i:i + batch_size]
        if len(set(batch_y)) > 1:  
            X_resampled, y_resampled = resampler.fit_resample(batch_X, batch_y)
            resampled_X.append(torch.tensor(X_resampled))
            resampled_y.append(torch.tensor(y_resampled))
        else:
            resampled_X.append(torch.tensor(batch_X))
            resampled_y.append(torch.tensor(batch_y))
    resampled_X = torch.cat(resampled_X, dim=0)
    resampled_y = torch.cat(resampled_y, dim=0)
    return resampled_X, resampled_y


def get_dataloaders(data, test_size=0.2, batch_size=256, max_length=20):
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from imblearn.over_sampling import ADASYN
    from imblearn.under_sampling import TomekLinks
    
    df = pd.DataFrame(data)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    X_train, X_test, y_train, y_test = train_test_split(df['fullname'], df['label'], test_size=test_size, random_state=42)

    base_dir = "/root/lanyun-tmp/duhaixing/bert_classification_ea_nonea/tokenize"
    train_save_path = os.path.join(base_dir, f'train_tokenized_maxlen_{max_length}.pt')
    test_save_path = os.path.join(base_dir, f'test_tokenized_maxlen_{max_length}.pt')

    X_train_tokens, X_train_attention = batch_tokenize_fullnames(tokenizer, X_train.tolist(), batch_size=32, max_length=max_length, save_path=train_save_path)
    X_test_tokens, X_test_attention = batch_tokenize_fullnames(tokenizer, X_test.tolist(), batch_size=32, max_length=max_length, save_path=test_save_path)
    
    adasyn = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=5)
    X_resampled, y_resampled = batch_resample(X_train_tokens.numpy(), y_train, batch_size=10000, resampler=adasyn)

    tomek = TomekLinks()
    X_resampled, y_resampled = batch_resample(X_resampled.numpy(), y_resampled.numpy(), batch_size=10000, resampler=tomek)

    train_dataset = FullnameDataset(
        tokenizer.batch_decode(X_resampled, skip_special_tokens=True), 
        y_resampled, 
        tokenizer, 
        max_length
    )
    test_dataset = FullnameDataset(
        # tokenizer.batch_decode(X_test_tokens, skip_special_tokens=True), 
        X_test.tolist(),
        y_test.tolist(), 
        tokenizer, 
        max_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader



