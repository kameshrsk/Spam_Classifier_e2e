import pandas as pd
import logging
from typing import Tuple, Annotated

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.utils import shuffle

from datasets import Dataset

from torch.utils.data import DataLoader

from .tokenizer import tokenize_data, tokenizer

from transformers import DataCollatorWithPadding

from zenml import step

@step(enable_cache=False)
def load_data(path:str)-> Tuple[
    Annotated[DataLoader, "training_batch"],
    Annotated[DataLoader, "testing_batch"]
]:

    data=pd.read_csv("C:/Users/kamesh/Desktop/Data Analysis/ML/MLOps/Spam Classifier/data/SMSSpamCollection", sep='\t', names=["labels", "Message"])

    ham=data[data['labels']=='ham'].sample(500)
    spam=data[data['labels']=='spam'].sample(500)

    data=pd.concat([ham, spam], axis=0)

    data=shuffle(data)
    data.reset_index(drop=True, inplace=True)

    lable_encoder=LabelEncoder()

    training_data, testing_data=train_test_split(data, test_size=0.2, random_state=101)

    training_data['labels']=lable_encoder.fit_transform(training_data['labels'])
    testing_data['labels']=lable_encoder.transform(testing_data['labels'])

    training_data=Dataset.from_pandas(training_data)
    testing_data=Dataset.from_pandas(testing_data)

    tokenized_training_data=training_data.map(tokenize_data, batched=True, remove_columns=['Message', '__index_level_0__'])
    tokenized_testing_data=testing_data.map(tokenize_data, batched=True, remove_columns=['Message', '__index_level_0__'])

    
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)

    training_batch=DataLoader(
        tokenized_training_data,
        shuffle=True,
        batch_size=8,
        collate_fn=data_collator
    )

    testing_batch=DataLoader(
        tokenized_testing_data,
        shuffle=True,
        batch_size=8,
        collate_fn=data_collator
    )

    return training_batch, testing_batch

