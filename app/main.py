"""
MedVision FastAPI backend.

Serves the team-trained MobileNetV2 models for Chest X-ray (3-class pneumonia:
Bacterial / Normal / Viral) and Chest CT (4-class lung cancer: Adenocarcinoma /
Large Cell Carcinoma / Normal / Squamous Cell Carcinoma) with Grad-CAM explainability.
"""

import os
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import model_utils

app = FastAPI(title="MedVision API", version="2.1.0")

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("MEDVISION_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials="*" not in ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


class PredictResponse(BaseModel):
    modality: Literal["xray", "ct"] = "xray"
    prediction: str
    confidence: float
    class_probabilities: dict[str, float] = {}
    heatmap_base64: str
    model_used: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modalities": model_utils.modality_status(),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict_scan(
    file: UploadFile = File(...),
    modality: Literal["xray", "ct"] = Form("xray"),
):
    bundle = model_utils.get_bundle(modality)
    if not bundle.exists():
        label = "Chest CT" if modality == "ct" else "Chest X-ray"
        raise HTTPException(
            status_code=503,
            detail=f"{label} model ({os.path.basename(bundle.model_path)}) is not deployed on this server.",
        )

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

    if not model_utils.looks_like_medical_scan(pil_img):
        scan_label = "CT scan" if modality == "ct" else "X-ray"
        return PredictResponse(
            modality=modality,
            prediction="Invalid",
            confidence=0.0,
            class_probabilities={},
            heatmap_base64="",
            model_used=f"Pre-check (image does not resemble a grayscale {scan_label})",
        )

    result = bundle.predict(arr)
    heatmap_b64 = bundle.generate_gradcam(arr)

    return PredictResponse(
        modality=modality,
        prediction=result["prediction"],
        confidence=result["confidence"],
        class_probabilities=result["class_probabilities"],
        heatmap_base64=heatmap_b64,
        model_used=bundle.display_name,
    )
