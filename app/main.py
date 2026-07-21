"""
MedVision FastAPI backend.

Serves the team-trained 3-class MobileNetV2 model (Bacterial / Normal / Viral) with
Grad-CAM explainability.
"""

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import model_utils

app = FastAPI(title="MedVision API", version="2.0.0")

ALLOWED_ORIGINS = os.environ.get("MEDVISION_ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    class_probabilities: dict[str, float] = {}
    heatmap_base64: str
    model_used: str = "MobileNetV2 (3-class: Bacterial/Normal/Viral, fine-tuned)"


@app.on_event("startup")
def _warm_up_model():
    model_utils.get_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict_xray(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Please upload a JPEG or PNG image.")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Image exceeds 10MB limit.")
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        arr, pil_img = model_utils.load_and_preprocess(raw_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    if not model_utils.looks_like_xray(pil_img):
        return PredictResponse(
            prediction="Invalid",
            confidence=0.0,
            class_probabilities={},
            heatmap_base64="",
            model_used="Pre-check (image does not resemble a grayscale X-ray)",
        )

    result = model_utils.predict(arr)
    heatmap_b64 = model_utils.generate_gradcam(arr)

    return PredictResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        class_probabilities=result["class_probabilities"],
        heatmap_base64=heatmap_b64,
    )
