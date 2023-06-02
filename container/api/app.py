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
import pandas as pd
from fastapi import FastAPI, Request
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

amb = anomaly_model.AnomalyModel()

# Instantiating FastAPI
app = FastAPI()

## API INSTANTIATION WITH FUNCTIONS
# app = FastAPI(dependencies=[Depends(get_query_token)])

# app.include_router(hyperparameters.router)
# app.include_router(predictions.router)
# app.include_router(
#    admin.router,
#    prefix="/admin",
#    tags=["admin"],
#    dependencies=[Depends(get_token_header)],
#    responses={418: {"description": "I'm an AI/ML System"}},
# )


@app.get("/")
async def root():
    return {"message": "Hello Bigger AI/ML Applications!"}


@app.get("/ping1")
async def ping():
    return {"message": "ok"}


@app.on_event("startup")
def load_model():
    classifier = "anomaly_model.AnomalyModel()"
    logging.info("Model loaded.")
    return classifier


@app.post("/predict")
async def basic_predict(input_data: list):

    # Converting dict to pandas dataframe
    input_df = pd.DataFrame(input_data)

    # Getting the prediction
    pred = amb.predict(input_df)

    return pred


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.1.0.1", port=8000, reload=True)
