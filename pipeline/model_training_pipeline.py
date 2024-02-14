from zenml import pipeline

from steps.trainer import train_model

from steps.evaluator import evaluate_model

from steps.load_artifacts import load_training_batch, load_testing_batch

from torch.utils.data import DataLoader

@pipeline
def model_training_pipeline(num_labels:int, num_epochs:int, learning_rate:float):
    training_batch=load_training_batch()
    model=train_model(training_batch, num_labels, num_epochs, learning_rate)
    testing_batch=load_testing_batch()
    accuracy, f1_score=evaluate_model(model, testing_batch)