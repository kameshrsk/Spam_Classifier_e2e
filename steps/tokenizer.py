from transformers import AutoTokenizer

import pandas as pd

checkpoint="bert-base-uncased"

tokenizer=AutoTokenizer.from_pretrained(checkpoint)

def tokenize_data(data):
    return tokenizer(
        data['Message'],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    )