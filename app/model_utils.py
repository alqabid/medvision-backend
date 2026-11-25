"""
Model loading, preprocessing, prediction, and Grad-CAM utilities for MedVision.

Supports two chest imaging modalities via per-modality ModelBundle instances:

1. Chest X-ray ("xray") — model/pneumonia_mobilenetv2_3class.h5
   - MobileNetV2 transfer-learning classifier distinguishing 3 classes:
     ["Bacterial", "Normal", "Viral"] (alphabetical order, matching Keras folder indices).
   - Trained on the Kermany chest X-ray dataset (Kaggle: paultimothymooney/chest-xray-pneumonia),
     with pneumonia images split into Bacterial/Viral using filename metadata.
   - Known limitation: Viral pneumonia recall (~48%) is lower than Bacterial (~87%) and
     Normal (~91%) due to ~2:1 Bacterial-to-Viral class imbalance in the training data.

2. Chest CT ("ct") — model/ct_mobilenetv2_4class.h5
   - MobileNetV2 transfer-learning classifier distinguishing 4 classes:
     ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]
     (alphabetical order, matching training).
   - Trained on the Kaggle Chest CT-Scan Images dataset (mohamedhanyyy/chest-ctscan-images),
     which is a lung-cancer CT dataset (NOT pneumonia CT).

Architecture & Grad-CAM
-----------------------
In both models, MobileNetV2 ("mobilenetv2_1.00_224", last conv layer "out_relu") is saved
as a *nested* Functional submodel inside the outer model (not flattened). Grad-CAM builds
two submodels per bundle:
  1. last_conv_layer_model: image -> last conv layer's feature maps
  2. classifier_model: feature maps -> final prediction through the outer head layers

Preprocessing: resize to 224x224 RGB, rescale to [0, 1] (not preprocess_input).
"""

import base64
from dataclasses import dataclass, field
import gc
import io
import os
import threading
from typing import Literal

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

Modality = Literal["xray", "ct"]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
IMG_SIZE = (224, 224)
Image.MAX_IMAGE_PIXELS = 25_000_000
_MODEL_LOCK = threading.Lock()


@dataclass
class ModelBundle:
    modality: Modality
    model_path: str
    class_names: list[str]
    display_name: str
    base_model_layer_name: str = "mobilenetv2_1.00_224"
    last_conv_layer_name: str = "out_relu"
    head_layer_names: list[str] = field(default_factory=list)
    _model: keras.Model | None = field(default=None, init=False, repr=False)
    _last_conv_layer_model: keras.Model | None = field(default=None, init=False, repr=False)
    _classifier_model: keras.Model | None = field(default=None, init=False, repr=False)

    def exists(self) -> bool:
        return os.path.isfile(self.model_path)

    def unload(self) -> None:
        self._model = None
        self._last_conv_layer_model = None
        self._classifier_model = None

    def get_model(self) -> keras.Model:
        if not self.exists():
            self.unload()
            raise FileNotFoundError(
                f"Model file for modality '{self.modality}' not found at {self.model_path}"
            )
        with _MODEL_LOCK:
            if self._model is None:
                # Keep at most one TensorFlow model in memory at a time (Render 512MB limit)
                evicted = False
                for other_modality, other_bundle in BUNDLES.items():
                    if other_modality != self.modality and other_bundle._model is not None:
                        other_bundle.unload()
                        evicted = True
                if evicted:
                    keras.backend.clear_session()
                    gc.collect()
                self._model = keras.models.load_model(self.model_path, compile=False)
            return self._model

    def _get_gradcam_submodels(self) -> tuple[keras.Model, keras.Model]:
        """
        Build two small models for Grad-CAM:
          1. last_conv_layer_model: image -> last conv layer's feature maps (inside the
             nested MobileNetV2 submodel)
          2. classifier_model: those feature maps -> final prediction (replays the outer
             model's head layers: GAP -> Dense -> Dropout -> Dense)
        """
        model = self.get_model()
        if self._last_conv_layer_model is not None and self._classifier_model is not None:
            return self._last_conv_layer_model, self._classifier_model

        base_model = model.get_layer(self.base_model_layer_name)
        last_conv_layer = base_model.get_layer(self.last_conv_layer_name)

        self._last_conv_layer_model = keras.Model(base_model.input, last_conv_layer.output)

        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer_name in self.head_layer_names:
            x = model.get_layer(layer_name)(x)
        self._classifier_model = keras.Model(classifier_input, x)

        return self._last_conv_layer_model, self._classifier_model

    def predict(self, arr: np.ndarray) -> dict:
        model = self.get_model()
        probs = model.predict(arr, verbose=0)[0]
        class_idx = int(np.argmax(probs))
        label = self.class_names[class_idx]
        confidence = float(probs[class_idx])
        return {
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "class_probabilities": {
                self.class_names[i]: round(float(probs[i]) * 100, 2)
                for i in range(len(self.class_names))
            },
        }

    def generate_gradcam(self, arr: np.ndarray) -> str:
        """Run Grad-CAM (against the predicted class) and return a base64-encoded PNG overlay."""
        last_conv_layer_model, classifier_model = self._get_gradcam_submodels()

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


BUNDLES: dict[Modality, ModelBundle] = {
    "xray": ModelBundle(
        modality="xray",
        model_path=os.path.join(MODEL_DIR, "pneumonia_mobilenetv2_3class.h5"),
        class_names=["Bacterial", "Normal", "Viral"],
        display_name="MobileNetV2 (3-class: Bacterial/Normal/Viral, fine-tuned)",
        base_model_layer_name="mobilenetv2_1.00_224",
        last_conv_layer_name="out_relu",
        head_layer_names=["global_average_pooling2d", "dense", "dropout", "dense_1"],
    ),
    "ct": ModelBundle(
        modality="ct",
        model_path=os.path.join(MODEL_DIR, "ct_mobilenetv2_4class.h5"),
        class_names=[
            "Adenocarcinoma",
            "Large Cell Carcinoma",
            "Normal",
            "Squamous Cell Carcinoma",
        ],
        display_name="MobileNetV2 (4-class CT: Adenocarcinoma/Large Cell/Normal/Squamous Cell, fine-tuned)",
        base_model_layer_name="mobilenetv2_1.00_224",
        last_conv_layer_name="out_relu",
        head_layer_names=["gap", "dense_head", "dropout_head", "predictions"],
    ),
}


def get_bundle(modality: Modality = "xray") -> ModelBundle:
    if modality not in BUNDLES:
        raise ValueError(f"Unsupported modality: {modality}")
    return BUNDLES[modality]


def warm_up_available_models() -> dict[str, bool]:
    status: dict[str, bool] = {}
    for modality, bundle in BUNDLES.items():
        if bundle.exists():
            bundle.get_model()
            status[modality] = True
        else:
            status[modality] = False
    return status


def modality_status() -> dict[str, bool]:
    return {modality: bundle.exists() for modality, bundle in BUNDLES.items()}


def looks_like_medical_scan(img: Image.Image) -> bool:
    """
    Rough sanity check: chest X-rays and CT slices are effectively grayscale (R, G, B
    channels are nearly identical at every pixel). Filters out obviously-wrong colorful
    uploads before they hit the classifier. Will not catch every wrong image (e.g. a
    grayscale photo of something else), but catches the common case cheaply.
    """
    small = img.convert("RGB").resize((64, 64))
    arr = np.array(small).astype("float32")
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    channel_spread = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean()
    return bool(channel_spread < 12.0)


# Backwards-compatible alias
looks_like_xray = looks_like_medical_scan


def load_and_preprocess(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, img


def predict(arr: np.ndarray, modality: Modality = "xray") -> dict:
    return get_bundle(modality).predict(arr)


def generate_gradcam(arr: np.ndarray, modality: Modality = "xray") -> str:
    return get_bundle(modality).generate_gradcam(arr)
