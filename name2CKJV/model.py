# model.py

from transformers import BertForSequenceClassification

model_path = './bert-base-uncased/bert-base-uncased'

def get_model(num_labels=2):
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    return model
