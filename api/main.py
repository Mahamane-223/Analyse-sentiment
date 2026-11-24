from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.features import FEATURES, MODEL


class Measurements(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    bill_length_mm: float = Field(gt=0)
    bill_depth_mm: float = Field(gt=0)
    flipper_length_mm: float = Field(gt=0)
    body_mass_g: float = Field(gt=0)


class Prediction(BaseModel):
    species: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL.is_file():
        raise RuntimeError("Model file is missing. Run python train.py first.")
    app.state.model = joblib.load(MODEL)
    yield


app = FastAPI(title="Palmer Penguins", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, error: RequestValidationError):
    details = [
        {"loc": item["loc"], "msg": item["msg"], "type": item["type"]}
        for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": details})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(measurements: Measurements, request: Request):
    sample = pd.DataFrame([measurements.model_dump()], columns=FEATURES)
    species = request.app.state.model.predict(sample)[0]
    return Prediction(species=str(species))