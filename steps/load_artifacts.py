from zenml import step
from zenml.client import Client
from torch.utils.data import DataLoader
from typing import Tuple, Union, Annotated
from transformers import BertForSequenceClassification

def load_artifact_from_pipeline(pipeline_name: str, step_name: str, output_name: str):
    pipeline = Client().get_pipeline(pipeline_name)
    last_run = pipeline.last_successful_run if pipeline.last_successful_run else pipeline.last_run
    step = last_run.steps[step_name]
    output = step.outputs
    artifact_id = output[output_name].id
    artifact = Client().get_artifact_version(artifact_id)
    return artifact.load()

@step(enable_cache=False)
def load_training_batch() -> DataLoader:
    return load_artifact_from_pipeline('data_pipeline', 'load_data', 'training_batch')

@step(enable_cache=False)
def load_testing_batch() -> DataLoader:
    return load_artifact_from_pipeline('data_pipeline', 'load_data', 'testing_batch')

@step(enable_cache=False)
def load_trained_model() -> BertForSequenceClassification:
    return load_artifact_from_pipeline('model_training_pipeline', 'train_model', 'output')

@step(enable_cache=False)
def load_scores() -> Tuple[float, float]:
    accuracy = load_artifact_from_pipeline('model_evaluation_pipeline', 'evaluate_model', 'accuracy')
    f1_score = load_artifact_from_pipeline('model_evaluation_pipeline', 'evaluate_model', 'f1')
    return accuracy, f1_score
