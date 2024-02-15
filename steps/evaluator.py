from zenml import step

from zenml.client import Client

import torch

from torch.utils.data import DataLoader

from transformers import BertForSequenceClassification

from sklearn.metrics import accuracy_score, f1_score

from tqdm.auto import tqdm

from typing import Tuple, Annotated

import mlflow

import pickle

from .tokenizer import tokenize_data

import logging



@step(enable_cache=False)
def evaluate_model(model:BertForSequenceClassification, testing_batch:DataLoader)->Tuple[
    Annotated[float, "accuracy"],
    Annotated[float, "f1"]
]:

    model.eval()
    preds, y_true=[], []

    device=('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    
    num_steps=len(testing_batch)

    progress_bar=tqdm(range(num_steps))

    for batch in testing_batch:

        batch={k:v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            output=model(**batch)

        logits=output.logits

        predictions=torch.argmax(logits, dim=-1)

        preds.extend(predictions.tolist())
        y_true.extend(batch['labels'].tolist())

        progress_bar.update(1)

    accuracy=accuracy_score(y_true, preds)

    f1=f1_score(y_true, preds, average='weighted')

    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("F1 Score", f1)

    if accuracy>=0.90:
        model_dict={"model": model, "tokenizer": tokenize_data, "accuracy": accuracy, "f1_score": f1_score}
        pickle_out=open("saved_model/model.pkl", 'wb')
        pickle.dump(model_dict, pickle_out)
        pickle_out.close()
        logging.info("Model is Saved in the Deployment folder")
    else:
        logging.info("Model is Not good enough to save")
    

    return accuracy, f1
