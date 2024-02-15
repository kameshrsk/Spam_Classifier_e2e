import dvc.api
import pickle

with dvc.api.open(
    "saved_model/model.pkl",
    repo="https://github.com/kameshrsk/Spam_Classifier_e2e", mode='rb') as model_dict:

    model=pickle.load(model_dict)