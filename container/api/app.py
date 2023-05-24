# todo - add file description and docstring
"""
Machine learning module for Python
==================================

FastAPI application for serving machine learning models.

** Explain the purpose of the module in the ML
   deployment Pipeline **

** Explain how this module fits in the system architecture **

"""


import logging
import os
import pandas as pd
import pickle
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from fastapi import Depends, FastAPI

from container.api.dependencies import get_query_token, get_token_header
from container.api.internal import admin
from container.api.routers import hyperparameters, predictions
from container.model_package import anomaly_model

from container.model_package.anomaly_model.config.core import (
    DATASET_DIR,
    TRAINED_MODEL_DIR,
    config,
)

## API INSTANTIATION WITH OBJECTS
## ----------------------------------------------------------------

adb = anomaly_model.AnomalyModel()

# Instantiating FastAPI
app = FastAPI()


## API ENDPOINTS
## ----------------------------------------------------------------


app = FastAPI(dependencies=[Depends(get_query_token)])


app.include_router(hyperparameters.router)
app.include_router(predictions.router)
app.include_router(
    admin.router,
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(get_token_header)],
    responses={418: {"description": "I'm an AI/ML System"}},
)


@app.get("/")
async def root():
    return {"message": "Hello Bigger AI/ML Applications!"}


@app.get("/ping1")
async def ping():
    return {"message": "ok"}


@app.on_event("startup")
def load_model():
    classifier = "pipeline('anomaly', model=MODElS_PATH)"
    logging.info("Model loaded.")
    return classifier


@app.post("/invocations")
def invocations(request: Request):
    json_payload = "await request.json()"

    # inputs = [records["scope"] for records in json_payload]
    # output = [{"prediction": classifier(input)} for input in inputs]
    return "output"


# Defining the prediction endpoint without data validation
@app.post("/basic_predict")
async def basic_predict(request: Request):
    # Getting the JSON from the body of the request
    input_data = await request.json()

    # Converting JSON to Pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # Getting the prediction from the Logistic Regression model
    # pred = lr_model.predict(input_df)[0]

    return "pred"


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.1.0.1", port=8000, reload=True)
