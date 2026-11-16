# MedVision Backend (FastAPI + MobileNetV2 + Grad-CAM)

Inference backend for the MedVision diagnostic web app (FastAPI backend, trained MobileNetV2 classifiers, Grad-CAM explainability), supporting two chest imaging modalities:

1. **Chest X-ray (`xray`)** — 3-class pneumonia detection (`model/pneumonia_mobilenetv2_3class.h5`)
2. **Chest CT (`ct`)** — 4-class lung-cancer subtype classification (`model/ct_mobilenetv2_4class.h5`)

> **Important clinical note on the CT model:** The Chest CT model is trained on the Kaggle Chest CT-Scan Images dataset (`mohamedhanyyy/chest-ctscan-images`), which classifies **lung-cancer subtypes** (Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma) vs. Normal. **It is a lung-cancer CT classifier, not a pneumonia CT classifier.**

## Models & Classes

Both models are Keras MobileNetV2 transfer-learning models taking `224x224` RGB images rescaled to `[0, 1]` (not `preprocess_input` `[-1, 1]`). In both models, the MobileNetV2 backbone is saved as a nested submodel named `mobilenetv2_1.00_224` with last convolutional activation `out_relu`.

### Chest X-ray (`modality="xray"`, default)
- **File:** `model/pneumonia_mobilenetv2_3class.h5`
- **Dataset:** Kermany Chest X-Ray Pneumonia (`paultimothymooney/chest-xray-pneumonia`), split into Bacterial/Viral via filename metadata
- **Head layers:** `global_average_pooling2d` -> `dense` -> `dropout` -> `dense_1`

| Index | Class | Description |
| :---: | :--- | :--- |
| 0 | `Bacterial` | Bacterial pneumonia |
| 1 | `Normal` | Clear / healthy lung fields |
| 2 | `Viral` | Viral pneumonia |

### Chest CT (`modality="ct"`)
- **File:** `model/ct_mobilenetv2_4class.h5`
- **Dataset:** Chest CT-Scan Images (`mohamedhanyyy/chest-ctscan-images`) — **lung cancer CT, not pneumonia CT**
- **Head layers:** `gap` -> `dense_head` -> `dropout_head` -> `predictions`

| Index | Class | Description |
| :---: | :--- | :--- |
| 0 | `Adenocarcinoma` | Non-small cell lung cancer (adenocarcinoma) |
| 1 | `Large Cell Carcinoma` | Non-small cell lung cancer (large cell) |
| 2 | `Normal` | Normal chest CT |
| 3 | `Squamous Cell Carcinoma` | Non-small cell lung cancer (squamous cell) |

## Setup

```bash
cd medvision-backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## API

### `GET /health`

Returns server health and which modality model files are present on disk:

```json
{
  "status": "ok",
  "modalities": {
    "xray": true,
    "ct": true
  }
}
```

### `POST /predict`

Multipart form upload:
- `file` (required): JPEG or PNG image, max 10MB
- `modality` (optional): `"xray"` (default) or `"ct"`

If the requested modality's `.h5` model file is not present on the server, the endpoint returns **HTTP 503** with a descriptive `detail` message while the other modality continues to work.

#### `curl` Examples

```bash
# Chest X-ray (explicit or default)
curl -X POST http://localhost:8000/predict \
  -F "file=@sample_xray.jpg;type=image/jpeg" \
  -F "modality=xray"

# Chest CT (4-class lung cancer vs Normal)
curl -X POST http://localhost:8000/predict \
  -F "file=@sample_ct.png;type=image/png" \
  -F "modality=ct"
```

Response:
```json
{
  "modality": "ct",
  "prediction": "Adenocarcinoma",
  "confidence": 91.45,
  "class_probabilities": {
    "Adenocarcinoma": 91.45,
    "Large Cell Carcinoma": 3.12,
    "Normal": 1.08,
    "Squamous Cell Carcinoma": 4.35
  },
  "heatmap_base64": "<base64 PNG — Grad-CAM overlay>",
  "model_used": "MobileNetV2 (4-class CT: Adenocarcinoma/Large Cell/Normal/Squamous Cell, fine-tuned)"
}
```

## Wiring into the React frontend

```ts
const formData = new FormData();
formData.append("file", imageFile);
formData.append("modality", modality); // "xray" | "ct"

const response = await fetch(`${import.meta.env.VITE_MEDVISION_API_URL}/predict`, {
  method: "POST",
  body: formData,
});

if (!response.ok) {
  const err = await response.json().catch(() => null);
  throw new Error(err?.detail || "Prediction failed");
}
const result = await response.json();
```

## Deployment notes

- Set `MEDVISION_ALLOWED_ORIGINS` to your deployed frontend URL(s), comma-separated — don't leave CORS wide open (`*`) in production.
- `tensorflow` is a heavy dependency; if deploying to a small container, consider `tensorflow-cpu` and pinning Python 3.11/3.12.
