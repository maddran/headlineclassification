# -*- coding: utf-8 -*-
import numpy as np
import time
import datetime

from sklearn.preprocessing import LabelEncoder
from transformers import XLMRobertaTokenizer

import torch 
from torch.utils.data import DataLoader
from transformers import XLMRobertaForSequenceClassification, AdamW

print("Downloading XLM-Roberta Tokenizer...")
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Setting device to {device}")

def encode_text(text, tokenizer):
  return tokenizer(text, 
            max_length=128,
            add_special_tokens = True,
            truncation=True, 
            padding='max_length')
            
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
            
def accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
    

def predict(data_loader, model, label_flag = True):
    predictions = []
    true_labels = []

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        if label_flag:
            labels = batch['labels'].to(device)

        with torch.no_grad(): 
            outputs = model(input_ids, 
                                    attention_mask=attention_mask, 
                                    token_type_ids=None) 

        logits = outputs[0]                   
        logits = logits.detach().cpu().numpy()

        predictions.append(logits)
        if label_flag:
            label_ids = labels.to('cpu').numpy()
            true_labels.append(label_ids)

    if label_flag:
        return predictions,true_labels
    else:
        return predictions


def get_le():
    le = LabelEncoder()
    le.classes_ = np.load('classes.npy')
    return le


def kth_largest(predictions_i, k, le=get_le()):
    kth = np.argsort(-predictions_i, axis=1)[:, k-1]
    preds = le.inverse_transform(list(kth))
    if k != 1:
        mask = np.where(np.bincount(np.where(predictions_i > 0)[0]) != k)[0]
        preds = np.array(
            [None if i in mask else val for i, val in enumerate(preds)])

    return preds


def predict_pipeline(text, labels, model, le=get_le()):

    batch_size = min(len(text), 16)

    encoded = encode_text(text, tokenizer)
    encoded_labels = le.transform(labels)
    dataset = Dataset(encoded, encoded_labels)
    loader = DataLoader(dataset, batch_size=batch_size)
    predictions, true_labels = predict(loader, model)

    preds = [np.argmax(predictions[i], axis=1).flatten()
            for i in range(len(true_labels))]
    flat_preds = np.concatenate(preds).ravel()
    pred_cats1 = le.inverse_transform(flat_preds)

    true_labels = np.concatenate(true_labels).ravel()
    true_cats = le.inverse_transform(true_labels)

    pred_cats2 = [kth_largest(predictions[i], 2, le).flatten()
                    for i in range(len(predictions))]
    pred_cats2 = np.concatenate(pred_cats2).ravel()

    return pred_cats1, pred_cats2, true_cats
