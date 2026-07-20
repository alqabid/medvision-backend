"""
Model loading, preprocessing, prediction, and Grad-CAM utilities for MedVision.

Model provenance
-----------------
The trained weights (model/pneumonia_mobilenetv2.h5) are a MobileNetV2 transfer-learning
classifier adapted from the open-source project "Pneumonia Detection Using Chest X-Ray
Images" by Aditya Raj (https://github.com/adityaraj/Pneumonia-Detection-Using-Chest-X-Ray-Images
-- see original README for the author's dataset/training details). It is used here as the
inference backbone for the MedVision web application, per the project report's declared use
of an already-trained MobileNetV2 model.

Architecture (flat, not nested) so Grad-CAM can hook the last conv activation directly:
    MobileNetV2(include_top=False) -> GlobalAveragePooling2D -> Dense(128, relu)
    -> Dropout(0.3) -> Dense(1, sigmoid)

Preprocessing matches the original training pipeline: resize to 224x224, rescale to [0, 1]
(NOT the standard keras.applications.mobilenet_v2.preprocess_input [-1, 1] scaling).
"""

import io
import base64
import os

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "pneumonia_mobilenetv2.h5")
IMG_SIZE = (224, 224)
LAST_CONV_LAYER_NAME = "out_relu"  # last conv activation before GlobalAveragePooling2D
CLASS_NAMES = ["Normal", "Pneumonia"]
THRESHOLD = 0.55  # matches the original app.py decision threshold

_model = None
_grad_model = None


def get_model():
    """Lazily load the Keras model once per process."""
    global _model, _grad_model
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH, compile=False)
        # Build a second model that exposes the last conv layer's activations
        # alongside the final prediction, for Grad-CAM.
        _grad_model = keras.models.Model(
            inputs=_model.inputs,
            outputs=[_model.get_layer(LAST_CONV_LAYER_NAME).output, _model.output],
        )
    return _model


def get_grad_model():
    get_model()  # ensures _grad_model is built
    return _grad_model


def load_and_preprocess(image_bytes: bytes) -> tuple[np.ndarray, Image.Image]:
    """Decode raw image bytes -> (model input array, PIL image resized for display)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, img


def predict(arr: np.ndarray) -> dict:
    model = get_model()
    score = float(model.predict(arr, verbose=0)[0][0])
    label = CLASS_NAMES[1] if score >= THRESHOLD else CLASS_NAMES[0]
    # Confidence: how far the score sits from the decision boundary, expressed
    # as a percentage in favor of the predicted class.
    confidence = score if score >= THRESHOLD else (1 - score)
    return {
        "prediction": label,
        "raw_score": score,
        "confidence": round(confidence * 100, 2),
    }


def generate_gradcam(arr: np.ndarray) -> str:
    """Run Grad-CAM and return a base64-encoded PNG of the heatmap overlay."""
    grad_model = get_grad_model()

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(arr)
        # Some Keras 3 functional models wrap a single output in a list — normalize.
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        loss = predictions[:, 0]  # single sigmoid output

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize heatmap to image size and colorize
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE)
    heatmap_arr = np.array(heatmap_img)

    # Apply a simple red-heat colormap without extra dependencies (no cv2/matplotlib required)
    colored = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
    colored[..., 0] = heatmap_arr  # red channel scales with activation
    colored[..., 1] = (heatmap_arr * 0.3).astype(np.uint8)  # slight green for warmth
    heatmap_rgba = Image.fromarray(colored).convert("RGBA")

    # Fade the heatmap alpha by intensity so low-activation areas stay transparent
    alpha = Image.fromarray(np.uint8(heatmap_arr * 0.6))
    heatmap_rgba.putalpha(alpha)

    orig_img = Image.fromarray(np.uint8(arr[0] * 255)).convert("RGBA").resize(IMG_SIZE)
    overlay = Image.alpha_composite(orig_img, heatmap_rgba).convert("RGB")

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
