import click

import mlflow

from pipeline.data_pipeline import data_pipeline
from pipeline.model_training_pipeline import model_training_pipeline
from pipeline.model_evaluation_pipeline import model_evaluation_pipeline
from pipeline.end_to_end_pipeline import end_to_end_pipeline

@click.command(

    help="This is a Zenml Pipeline."
)

@click.option(
    "--load-data",
    is_flag=True,
    default=False,
    help="Run with this flag to create the dataset"
)
@click.option(
    "--train-model",
    is_flag=True,
    default=False,
    help="Run the Training Pipeline"
)

@click.option(
    "--evaluate-model",
    is_flag=True,
    default=None,
    help="Use this flag to evaluate the model from last successful training"
)

@click.option(
    "--end-to-end",
    is_flag=True,
    default=False,
    help="Use this to run all the pipelines in sequence"
)

@click.option(
    "--path",
    default=None,
    type=click.STRING,
    help="Give the path of the dataset"
)

@click.option(
    "--num-labels",
    default=2,
    type=click.INT,
    help="Enter Number of Labels to be classified"
)

@click.option(
    "--learning-rate",
    default=0.001,
    type=click.FLOAT,
    help="Enter the Learning rate"
)

@click.option(
    "--num-epochs",
    default=1,
    type=click.INT,
    help="Enter number of Epochs"
)

def main(load_data:bool=False, 
         path:str=None, 
         train_model:bool=False,
         training_artifact_id:str=None,
         num_labels:int=2,
         num_epochs:int=1,
         learning_rate:float=0.001,
         evaluate_model:bool=False,
         end_to_end:bool=False
):

    if load_data:

        data_pipeline(path)

    if train_model:

        model_training_pipeline(num_labels, num_epochs, learning_rate)

    if evaluate_model:

        model_evaluation_pipeline()

    if end_to_end:
        end_to_end_pipeline(path, num_labels, num_epochs, learning_rate)


if __name__=="__main__":

    mlflow.set_tracking_uri("http://ec2-15-207-108-193.ap-south-1.compute.amazonaws.com:5000/")

    main()
