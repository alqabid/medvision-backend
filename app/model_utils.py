"""
Model loading, preprocessing, prediction, and Grad-CAM utilities for MedVision.

Model provenance
-----------------
model/pneumonia_mobilenetv2_3class.h5 is a MobileNetV2 transfer-learning classifier,
trained by the project team specifically for this report, distinguishing 3 classes:
Bacterial Pneumonia, Normal, Viral Pneumonia (in that alphabetical order -- Keras assigns
label indices alphabetically by folder name).

Trained on the Kermany chest X-ray dataset (via Kaggle: paultimothymooney/chest-xray-pneumonia),
with pneumonia images split into Bacterial/Viral using filename metadata. Two-stage training:
frozen-base head training, then fine-tuning of the last ~30 MobileNetV2 layers.

Known limitation (documented, not hidden): Viral pneumonia recall is notably lower (~48%)
than Bacterial (~87%) and Normal (~91%), largely due to class imbalance in the training
data (Bacterial images outnumbered Viral roughly 2:1). The model tends to default to
"Bacterial" on ambiguous pneumonia cases. See project report for full evaluation metrics
and confusion matrix.

Architecture: MobileNetV2 is saved as a *nested* Functional submodel inside the outer model
(not flattened), so Grad-CAM needs to reach inside it explicitly -- see generate_gradcam().

Preprocessing: resize to 224x224, rescale to [0, 1] (matches training pipeline).
"""

import io
import base64
import os

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "pneumonia_mobilenetv2_3class.h5")
IMG_SIZE = (224, 224)
BASE_MODEL_LAYER_NAME = "mobilenetv2_1.00_224"  # nested MobileNetV2 submodel
LAST_CONV_LAYER_NAME = "out_relu"  # last conv activation, inside the nested submodel
HEAD_LAYER_NAMES = ["global_average_pooling2d", "dense", "dropout", "dense_1"]
CLASS_NAMES = ["Bacterial", "Normal", "Viral"]  # alphabetical order, matches training

_model = None
_last_conv_layer_model = None
_classifier_model = None


def get_model():
    global _model
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def _get_gradcam_submodels():
    """
    Build two small models for Grad-CAM:
      1. last_conv_layer_model: image -> last conv layer's feature maps (inside the
         nested MobileNetV2 submodel)
      2. classifier_model: those feature maps -> final prediction (replays the outer
         model's head layers: GAP -> Dense -> Dropout -> Dense)

    This split is necessary because the MobileNetV2 backbone is stored as a *nested*
    Functional submodel rather than being flattened into the outer model -- you can't
    directly build Model(inputs=outer.input, outputs=inner_layer.output) across that
    boundary, it raises a "graph disconnected" error.
    """
    global _last_conv_layer_model, _classifier_model
    if _last_conv_layer_model is not None and _classifier_model is not None:
        return _last_conv_layer_model, _classifier_model

    model = get_model()
    base_model = model.get_layer(BASE_MODEL_LAYER_NAME)
    last_conv_layer = base_model.get_layer(LAST_CONV_LAYER_NAME)

    _last_conv_layer_model = keras.Model(base_model.input, last_conv_layer.output)

    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in HEAD_LAYER_NAMES:
        x = model.get_layer(layer_name)(x)
    _classifier_model = keras.Model(classifier_input, x)

    return _last_conv_layer_model, _classifier_model


def looks_like_xray(img: Image.Image) -> bool:
    """
    Rough sanity check: chest X-rays are effectively grayscale (R, G, B channels are
    nearly identical at every pixel). Filters out obviously-wrong colorful uploads
    before they hit the classifier. Will not catch every wrong image (e.g. a grayscale
    photo of something else), but catches the common case cheaply.
    """
    small = img.convert("RGB").resize((64, 64))
    arr = np.array(small).astype("float32")
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    channel_spread = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean()
    return channel_spread < 12.0


def load_and_preprocess(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, img


def predict(arr: np.ndarray) -> dict:
    model = get_model()
    probs = model.predict(arr, verbose=0)[0]  # shape (3,)
    class_idx = int(np.argmax(probs))
    label = CLASS_NAMES[class_idx]
    confidence = float(probs[class_idx])
    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2),
        "class_probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASS_NAMES))
        },
    }


def generate_gradcam(arr: np.ndarray) -> str:
    """Run Grad-CAM (against the predicted class) and return a base64-encoded PNG overlay."""
    last_conv_layer_model, classifier_model = _get_gradcam_submodels()

    with tf.GradientTape() as tape:
        conv_outputs = last_conv_layer_model(arr)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE)
    heatmap_arr = np.array(heatmap_img)

    colored = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
    colored[..., 0] = heatmap_arr
    colored[..., 1] = (heatmap_arr * 0.3).astype(np.uint8)
    heatmap_rgba = Image.fromarray(colored).convert("RGBA")

    alpha = Image.fromarray(np.uint8(heatmap_arr * 0.6))
    heatmap_rgba.putalpha(alpha)

    orig_img = Image.fromarray(np.uint8(arr[0] * 255)).convert("RGBA").resize(IMG_SIZE)
    overlay = Image.alpha_composite(orig_img, heatmap_rgba).convert("RGB")

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
