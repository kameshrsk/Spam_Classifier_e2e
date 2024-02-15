from transformers import BertForSequenceClassification

from steps.tokenizer import tokenizer

import torch

from zenml import step

from zenml.client import Client

from typing import Tuple, Annotated

from torch.utils.data import DataLoader

from tqdm.auto import tqdm

import mlflow

@step(enable_cache=True)
def train_model(training_batch:DataLoader, num_labels:int, num_epochs:int, learning_rate:float)-> BertForSequenceClassification:

    model=BertForSequenceClassification.from_pretrained("bert-base-uncased")


    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_steps=num_epochs*len(training_batch)

    device=('cuda' if torch.cuda.is_available() else 'cpu')

    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("learning_rate", learning_rate)
    

    model.to(device)

    progress_bar=tqdm(range(num_steps))

    model.train()

    for epoch in range(num_epochs):

        for batch in training_batch:

            batch={k:v.to(device) for k, v in batch.items()}

            output=model(**batch)

            loss=output.loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            progress_bar.update(1)

    components={
        "model":model,
        "tokenizer":tokenizer
    }

    mlflow.transformers.log_model(transformers_model=components, artifact_path="model")


    return model

