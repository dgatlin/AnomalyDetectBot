# todo - add file description and docstring
"""
Machine learning module for Python
==================================

FastAPI application for serving machine learning models.

** Explain the purpose of the module in the ML
   deployment Pipeline **

** Explain how this module fits in the system architecture **

"""
import json
import logging

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from container.model_package import anomaly_model

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


class Item(BaseModel):
    one: float
    two: float
    three: float


class obs(BaseModel):
    obs: list[Item]


l = {"one": 1, "two": 2, "three": 3}

l2 = {"one": 20, "two": 12.66645094909174, "three": 15.8990837351338}


@app.post("/predict")
async def basic_predict(input_data: list[dict]):
    # Converting dict to pandas dataframe
    input_df = pd.DataFrame(input_data)

    # Getting the prediction
    pred = amb.predict(input_df)

    return pred


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.1.0.1", port=8000, reload=True)
