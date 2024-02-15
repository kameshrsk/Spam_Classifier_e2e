import dvc.api

with dvc.api.open(
    "saved_model/model.pkl",
    repo="https://github.com/kameshrsk/Spam_Classifier_e2e") as model_dict:

    print(model_dict)