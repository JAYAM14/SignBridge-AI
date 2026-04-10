"""
SignBridge AI — CNN Classifier Wrapper (classifier.py)

Provides the CNNClassifier class used by gesture_detector.py as a
standalone wrapper, plus utility functions for building the model
architecture.
"""

import os
import logging
import pickle
from typing import Tuple, Optional, List

import numpy as np

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

def build_cnn_model(num_classes: int, input_dim: int = 63) -> "tf.keras.Model":
    """
    Build the CNN–LSTM model for ISL gesture classification.

    Input:  (batch, 63)  — flattened normalised landmarks
    Output: (batch, num_classes) — softmax probabilities

    Architecture:
      Dense → Reshape → Conv1D → MaxPool → Dropout → LSTM → Dense (softmax)
    """
    if not _TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required to build the model.")

    inputs = keras.Input(shape=(input_dim,), name="landmarks")

    # Dense projection to sequence-like representation
    x = keras.layers.Dense(128, activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Reshape((16, 8))(x)   # (batch, time_steps, features)

    # Convolutional feature extraction
    x = keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.Dropout(0.3)(x)

    # Temporal LSTM
    x = keras.layers.LSTM(64, return_sequences=False)(x)
    x = keras.layers.Dropout(0.4)(x)

    # Classification head
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="SignBridge_CNN_LSTM")
    return model


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------

class CNNClassifier:
    """
    Wraps a Keras model + label encoder for sign classification.
    """

    def __init__(self, model_path: str, label_encoder_path: str):
        self._model_path = model_path
        self._label_encoder_path = label_encoder_path
        self._model = None
        self._label_encoder = None
        self._load()

    def _load(self):
        if not _TF_AVAILABLE:
            return

        if os.path.exists(self._model_path):
            try:
                self._model = tf.keras.models.load_model(self._model_path)
                logger.info(f"Classifier loaded from {self._model_path}")
            except Exception as exc:
                logger.error(f"Failed to load classifier: {exc}")

        if os.path.exists(self._label_encoder_path):
            try:
                with open(self._label_encoder_path, "rb") as f:
                    self._label_encoder = pickle.load(f)
                logger.info("Label encoder loaded.")
            except Exception as exc:
                logger.error(f"Failed to load label encoder: {exc}")

    def is_ready(self) -> bool:
        return self._model is not None and self._label_encoder is not None

    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        landmarks: shape (63,) — normalised flattened landmarks
        Returns (label, confidence).
        """
        if not self.is_ready():
            return "unknown", 0.0

        x = landmarks.reshape(1, -1).astype(np.float32)
        probs = self._model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        label = self._label_encoder.inverse_transform([idx])[0]
        return label, confidence

    def predict_batch(self, landmarks_batch: np.ndarray) -> List[Tuple[str, float]]:
        """
        landmarks_batch: shape (N, 63)
        Returns list of (label, confidence) tuples.
        """
        if not self.is_ready():
            return [("unknown", 0.0)] * len(landmarks_batch)

        x = landmarks_batch.astype(np.float32)
        probs = self._model.predict(x, verbose=0)
        results = []
        for prob in probs:
            idx = int(np.argmax(prob))
            results.append((self._label_encoder.inverse_transform([idx])[0], float(prob[idx])))
        return results

    def get_labels(self) -> Optional[List[str]]:
        if self._label_encoder is None:
            return None
        return list(self._label_encoder.classes_)
