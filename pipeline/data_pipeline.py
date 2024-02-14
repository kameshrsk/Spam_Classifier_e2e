import logging

from zenml import pipeline

from steps.data_loader import load_data

@pipeline
def data_pipeline(path:str):
    training_data, testing_data=load_data(path=path)