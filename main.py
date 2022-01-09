# The API app.
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from predictor.ml.data import process_data
from predictor.ml.model import inference
from predictor.ml.train_model import CAT_FEATURES
import os
import pickle


# DVC set-up for Heroku
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull -r censuscicdbucket") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Define the API handle
app = FastAPI()


# Data sample object (first of dataset), to ensure correct input formatting,
# and providing and example
class CensusData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(
        ...,
        example="Never-married",
        alias="marital-status"
    )
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(
        ...,
        example="United-States",
        alias="native-country"
    )


# Prediction response object, ensure correct formatting,
# and providing and example
class IncomePrediction(BaseModel):
    Income: str = Field(..., example=">50K")


@app.on_event("startup")
async def startup_event():
    """
    Loads the model artifacts (model, encoder, binarizer) on startpu
    """

    with open("model/model.pkl", "rb") as f:
        global MODEL
        MODEL = pickle.load(f)
    with open("model/encoder.pkl", "rb") as f:
        global ENCODER
        ENCODER = pickle.load(f)
    with open("model/label_binarizer.pkl", "rb") as f:
        global LB
        LB = pickle.load(f)


@app.get("/")
async def welcome(request: Request) -> str:
    """
    Provides a simple greeting on the root endpoint
    """
    return {"greeting": "Welcome to the REST API for the Income Predictor"}


@app.post("/invoke", response_model=IncomePrediction)
async def get_prediction(payload: CensusData) -> IncomePrediction:
    """
    Provides the model enpoint to predict on a provided sample
    Inputs
    ------
    payload : CensusData
        The provided data sample.
    Returns
    -------
    result : IncomePrediction
        The result, as response.
    """

    # Check if model artifacts are loaded and available
    if (not MODEL) or (not ENCODER) or (not LB):
        raise HTTPException(
            status_code=500,
            detail="Internal server problem, model artifacts are missing."
        )

    # Load the data to pandas, and process it
    payload_df = pd.DataFrame.from_dict([payload.dict(by_alias=True)])
    X, _, _, _ = process_data(
        payload_df,
        categorical_features=CAT_FEATURES,
        training=False,
        encoder=ENCODER,
        lb=LB
    )

    # Do the inference and format the output
    preds = inference(MODEL, X)
    result = {"Income": ">50K" if preds == 1 else "<=50K"}

    return result
