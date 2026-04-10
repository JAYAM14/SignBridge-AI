"""
SignBridge AI — Gesture Detector (gesture_detector.py)

Pipeline:
  1. Decode raw JPEG/PNG bytes → OpenCV frame
  2. Pre-process frame for hand_landmark.tflite
  3. Run tf.lite.Interpreter → 21 hand landmarks (x, y, z)
  4. Normalize landmarks relative to wrist anchor
  5. Flatten → feed into CNN classifier
  6. Return label + confidence

STRICT: Uses tf.lite.Interpreter ONLY — no MediaPipe.
"""

import os
import io
import logging
import pickle
from typing import Optional, Dict, Any, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports — only loaded when available
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    logger.warning("TensorFlow not installed — gesture detection disabled.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_LANDMARKS = 21          # MediaPipe hand model outputs 21 landmarks
LANDMARK_DIM = 3            # x, y, z per landmark
INPUT_SIZE = (256, 256)     # hand_landmark.tflite expected input size
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to report a gesture


class GestureDetector:
    """
    Wraps the TFLite hand landmark model and a Keras CNN classifier to
    predict ISL sign labels from raw image bytes.
    """

    def __init__(
        self,
        tflite_model_path: str,
        classifier_model_path: str,
        label_encoder_path: str,
    ):
        self._tflite_model_path = tflite_model_path
        self._classifier_model_path = classifier_model_path
        self._label_encoder_path = label_encoder_path

        self._landmark_interpreter: Optional[Any] = None
        self._classifier: Optional[Any] = None
        self._label_encoder: Optional[Any] = None
        self._ready = False

        self._load_models()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_models(self):
        if not _TF_AVAILABLE:
            logger.error("TensorFlow not available — cannot load models.")
            return

        # Load TFLite hand landmark model
        if os.path.exists(self._tflite_model_path):
            try:
                self._landmark_interpreter = tf.lite.Interpreter(
                    model_path=self._tflite_model_path
                )
                self._landmark_interpreter.allocate_tensors()
                self._input_details = self._landmark_interpreter.get_input_details()
                self._output_details = self._landmark_interpreter.get_output_details()
                logger.info("hand_landmark.tflite loaded successfully.")
            except Exception as exc:
                logger.error(f"Failed to load hand_landmark.tflite: {exc}")
        else:
            logger.warning(
                f"hand_landmark.tflite not found at {self._tflite_model_path}. "
                "Gesture detection will use fallback landmark simulation."
            )

        # Load Keras CNN classifier
        if os.path.exists(self._classifier_model_path):
            try:
                self._classifier = tf.keras.models.load_model(self._classifier_model_path)
                logger.info("CNN classifier loaded successfully.")
            except Exception as exc:
                logger.error(f"Failed to load CNN classifier: {exc}")
        else:
            logger.warning(
                f"Classifier model not found at {self._classifier_model_path}. "
                "Run model/train_model.py to train and generate the model."
            )

        # Load label encoder
        if os.path.exists(self._label_encoder_path):
            try:
                with open(self._label_encoder_path, "rb") as f:
                    self._label_encoder = pickle.load(f)
                logger.info("Label encoder loaded successfully.")
            except Exception as exc:
                logger.error(f"Failed to load label encoder: {exc}")
        else:
            logger.warning(
                f"Label encoder not found at {self._label_encoder_path}."
            )

        self._ready = (
            self._landmark_interpreter is not None
            and self._classifier is not None
            and self._label_encoder is not None
        )

    def is_ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        img_bytes: bytes,
        calibration_data: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Predict ISL sign from raw image bytes.

        Returns dict:
          { "label": str, "confidence": float, "landmarks": List[List[float]] }
        or None if no hand detected.
        """
        frame = self._decode_frame(img_bytes)
        if frame is None:
            return None

        landmarks = self._extract_landmarks(frame)
        if landmarks is None:
            return None

        # Apply per-user calibration offset if available
        if calibration_data is not None:
            landmarks = self._apply_calibration(landmarks, calibration_data)

        normalized = self._normalize_landmarks(landmarks)

        if not self._ready:
            # Return landmarks without classification when models are unavailable
            return {
                "label": "unknown",
                "confidence": 0.0,
                "landmarks": landmarks.tolist(),
                "message": "Classifier not loaded",
            }

        label, confidence = self._classify(normalized)

        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "label": None,
                "confidence": float(confidence),
                "landmarks": landmarks.tolist(),
                "message": "Low confidence",
            }

        return {
            "label": label,
            "confidence": float(confidence),
            "landmarks": landmarks.tolist(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_frame(self, img_bytes: bytes) -> Optional[np.ndarray]:
        """Decode JPEG/PNG bytes to BGR OpenCV array."""
        try:
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception as exc:
            logger.error(f"Frame decode error: {exc}")
            return None

    def _extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run TFLite hand landmark model.
        Returns array of shape (21, 3) or None if no hand detected.
        """
        if self._landmark_interpreter is None:
            # Fallback: simulate landmarks from frame edges (for testing)
            return self._simulate_landmarks(frame)

        try:
            # Resize and normalise input
            input_shape = self._input_details[0]["shape"]  # e.g. [1, 256, 256, 3]
            h, w = input_shape[1], input_shape[2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (w, h))
            input_data = resized.astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

            self._landmark_interpreter.set_tensor(
                self._input_details[0]["index"], input_data
            )
            self._landmark_interpreter.invoke()

            # The hand landmark model typically outputs a flat array of 21*3 values
            # Output tensor index 0 contains landmark coordinates
            raw_output = self._landmark_interpreter.get_tensor(
                self._output_details[0]["index"]
            )  # shape: (1, 63) or (1, 21, 3)

            landmarks = raw_output[0]  # Remove batch dim

            if landmarks.ndim == 1:
                # Flat (63,) → reshape to (21, 3)
                if landmarks.shape[0] >= NUM_LANDMARKS * LANDMARK_DIM:
                    landmarks = landmarks[: NUM_LANDMARKS * LANDMARK_DIM].reshape(
                        NUM_LANDMARKS, LANDMARK_DIM
                    )
                else:
                    return None
            elif landmarks.ndim == 2 and landmarks.shape[0] == NUM_LANDMARKS:
                pass  # already (21, 3)
            else:
                return None

            # Check hand presence — if all values are near zero, no hand detected
            if np.max(np.abs(landmarks)) < 1e-5:
                return None

            return landmarks.astype(np.float32)

        except Exception as exc:
            logger.error(f"Landmark extraction error: {exc}")
            return None

    def _simulate_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Fallback when TFLite model is absent: attempt simple skin-colour
        based centroid detection and return a zeroed landmark array.
        This is a degraded path — real detection requires hand_landmark.tflite.
        """
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 20, 70], dtype=np.uint8)
            upper = np.array([20, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) < 1000:
                return None

            # Return zero-filled landmarks (centroid only at wrist position)
            h, w = frame.shape[:2]
            M = cv2.moments(largest)
            if M["m00"] == 0:
                return None
            cx = M["m10"] / M["m00"] / w
            cy = M["m01"] / M["m00"] / h

            landmarks = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
            landmarks[0] = [cx, cy, 0.0]   # wrist anchor
            return landmarks
        except Exception:
            return None

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize 21 landmarks relative to wrist (landmark 0).
        Scale by the distance between wrist and middle-finger-MCP (landmark 9).
        Returns flat array of shape (63,).
        """
        wrist = landmarks[0].copy()
        normalized = landmarks - wrist  # translate wrist to origin

        # Compute scale factor
        ref_dist = np.linalg.norm(normalized[9])
        if ref_dist > 1e-6:
            normalized = normalized / ref_dist

        return normalized.flatten()  # (63,)

    def _apply_calibration(
        self, landmarks: np.ndarray, calibration_data: Dict
    ) -> np.ndarray:
        """Apply per-user calibration offset to raw landmarks."""
        offset = np.array(calibration_data.get("offset", np.zeros(3)), dtype=np.float32)
        scale = float(calibration_data.get("scale", 1.0))
        return (landmarks - offset) * scale

    def _classify(self, normalized: np.ndarray):
        """Run CNN classifier and decode label."""
        try:
            # Reshape for model input: (1, 63) or (1, 21, 3)
            x = normalized.reshape(1, -1).astype(np.float32)
            probs = self._classifier.predict(x, verbose=0)[0]
            idx = int(np.argmax(probs))
            confidence = float(probs[idx])
            label = self._label_encoder.inverse_transform([idx])[0]
            return label, confidence
        except Exception as exc:
            logger.error(f"Classification error: {exc}")
            return "unknown", 0.0
