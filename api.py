from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from get_model import model as loaded_model

from message import Msg

import pickle

import torch

model_dict=loaded_model

model=model_dict['model']

tokenize_data=model_dict['tokenizer']

app=FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return {"Message":"Welcome to Spam message classification"}

@app.post('/classify')
def classify(data:Msg):
    data=data.dict()

    tokenized_input_data=tokenize_data(data)
    output=model(**tokenized_input_data)
    
    logits=output.logits

    prediction=torch.argmax(logits, dim=-1)

    if prediction==1:
        return {"Prediction":"Spam Message"}
    else:
        return {"Prediction":"NOT a Spam Message"}
    
if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
