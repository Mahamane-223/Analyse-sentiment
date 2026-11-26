import logging
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.config import LABELS, MODEL_PATH

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 5000


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)


class Prediction(BaseModel):
    label: str
    proba: float = Field(ge=0, le=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model file is missing ({MODEL_PATH}). Run python src/train.py first."
        )
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="Analyse de sentiments", lifespan=lifespan)


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
def predict(review: Review, request: Request):
    model = request.app.state.model
    try:
        probas = model.predict_proba([review.text])[0]
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    best = int(probas.argmax())
    label = LABELS[int(model.classes_[best])]
    return Prediction(label=label, proba=round(float(probas[best]), 4))
