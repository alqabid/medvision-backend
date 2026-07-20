# MedVision Backend (FastAPI + MobileNetV2 + Grad-CAM)

Real inference backend for the MedVision pneumonia-detection web app, matching the
architecture described in the project report (FastAPI backend, trained MobileNetV2
classifier, Grad-CAM explainability). This replaces the placeholder Gemini-based
`analyze-xray` Supabase edge function that was previously calling an LLM vision model
and mislabeling its output as `"MobileNetV2 (AI-assisted)"`.

## Model provenance

`model/pneumonia_mobilenetv2.h5` is a MobileNetV2 transfer-learning binary classifier
(Normal vs. Pneumonia) adapted from the open-source project **"Pneumonia Detection Using
Chest X-Ray Images" by Aditya Raj**. It is used here as an already-trained model, per the
report's stated approach. See `model_utils.py` for architecture details and the original
project for training/dataset specifics.

- Input: 224x224 RGB, rescaled to [0, 1] (not the standard `preprocess_input` [-1, 1] scaling)
- Architecture: MobileNetV2 (no top) -> GlobalAveragePooling2D -> Dense(128, relu) -> Dropout(0.3) -> Dense(1, sigmoid)
- Decision threshold: 0.55 (score >= 0.55 -> Pneumonia)

## Setup

```bash
cd medvision-backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Health check: `GET http://localhost:8000/health`

## API

### `POST /predict`

Multipart form upload, field name `file` (JPEG or PNG, max 10MB).

Response:
```json
{
  "prediction": "Pneumonia",
  "confidence": 92.4,
  "raw_score": 0.924,
  "heatmap_base64": "<base64 PNG — Grad-CAM overlay>",
  "model_used": "MobileNetV2 (transfer learning, fine-tuned)"
}
```

## Wiring into the existing React frontend

Replace the Supabase edge function call in your upload flow with a direct call to this
API. Example (drop-in for wherever `analyze-xray` was invoked):

```ts
const formData = new FormData();
formData.append("file", imageFile);

const response = await fetch(`${import.meta.env.VITE_MEDVISION_API_URL}/predict`, {
  method: "POST",
  body: formData,
});

if (!response.ok) throw new Error("Prediction failed");
const result = await response.json();
// result.prediction, result.confidence, result.heatmap_base64
```

Add `VITE_MEDVISION_API_URL=http://localhost:8000` (or your deployed URL) to the
frontend's `.env`.

You can still keep the Supabase `analyze-xray` function as a thin proxy (auth check +
forward to this FastAPI service + write to the `analyses`/`audit_logs` tables) if you
want to preserve the existing auth/history flow — just replace the Lovable AI gateway
call inside it with a `fetch()` to this API's `/predict` endpoint instead of calling
Gemini directly.

## Deployment notes

- Set `MEDVISION_ALLOWED_ORIGINS` to your deployed frontend URL(s), comma-separated —
  don't leave CORS wide open (`*`) in production.
- `tensorflow` is a heavy dependency; if deploying to a small container, consider
  `tensorflow-cpu` and pinning Python 3.11/3.12.
