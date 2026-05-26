"""
FastAPI service — surface roughness classification.

Endpoints:
  POST /predict   — accepts an image file + grit value, returns label + confidence
  GET  /health    — liveness check
"""
import io
import os
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from api.predictor import Predictor
from config import CLASSES

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "./model.pth")

app = FastAPI(title="Surface Roughness Classifier", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

predictor: Optional[Predictor] = None


@app.on_event("startup")
def load_model():
    global predictor
    if not os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    predictor = Predictor(CHECKPOINT_PATH)
    print(f"Model loaded from {CHECKPOINT_PATH}")


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Microscopy image (PNG/JPEG)"),
    grit: int = Form(..., description="Grit value used during machining"),
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot read image: {e}")

    try:
        result = predictor.predict(image, grit)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictionResponse(**result)
