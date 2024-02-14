from zenml import pipeline
from steps.evaluator import evaluate_model
from steps.load_artifacts import load_trained_model, load_testing_batch



import pickle
import logging

@pipeline
def model_evaluation_pipeline():
    testing_batch=load_testing_batch()
    model=load_trained_model()
    accuracy, f1_score=evaluate_model(model, testing_batch)