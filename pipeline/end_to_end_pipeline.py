from zenml import pipeline

from steps.data_loader import load_data
from steps.trainer import train_model
from steps.evaluator import evaluate_model

@pipeline
def end_to_end_pipeline(path:str, num_labels:int, num_epochs:int, learning_rate:float):
    training_data, testing_data=load_data(path=path)
    model=train_model(training_data, num_labels, num_epochs, learning_rate)
    accuracy, f1_score=evaluate_model(model, testing_data)