"""
SignBridge AI — Flask Backend API Server (app.py)

Routes:
  POST /api/detect   — Receives base64 frame, returns sign label + confidence
  POST /api/speech   — Receives text + language, returns audio file
  POST /api/mapper   — Receives word/phrase, returns ISL gesture token sequence
  GET  /api/status   — Health check
  POST /api/calibrate — Stores calibration data for a user session
"""

import os
import sys
import logging
import base64
import json
import io
import time

import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Path setup — allow imports from project root and backend folder
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, BACKEND_DIR)

from gesture_detector import GestureDetector
from tts_engine import TTSEngine
from nlp_mapper import NLPMapper
from calibration import Calibration

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SignBridgeAPI")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "frontend"), static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ---------------------------------------------------------------------------
# Module initialisation
# ---------------------------------------------------------------------------
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmark.tflite")
CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, "model", "signbridge_model.h5")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
AUDIO_CACHE_DIR = os.path.join(BACKEND_DIR, "audio_cache")
KEYPOINTS_DIR = os.path.join(BASE_DIR, "frontend", "assets", "keypoints")

os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

gesture_detector = GestureDetector(
    tflite_model_path=TFLITE_MODEL_PATH,
    classifier_model_path=CLASSIFIER_MODEL_PATH,
    label_encoder_path=LABEL_ENCODER_PATH,
)

tts_engine = TTSEngine(cache_dir=AUDIO_CACHE_DIR)
nlp_mapper = NLPMapper(keypoints_dir=KEYPOINTS_DIR)
calibration = Calibration()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/status", methods=["GET"])
def status():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "version": "1.0.0",
        "modules": {
            "gesture_detector": gesture_detector.is_ready(),
            "tts_engine": True,
            "nlp_mapper": True,
        },
        "timestamp": time.time(),
    })


@app.route("/api/detect", methods=["POST"])
def detect():
    """
    Receive a webcam frame (base64 JPEG/PNG) and return gesture prediction.

    Request JSON:
      { "frame": "<base64-encoded image>", "session_id": "optional" }

    Response JSON:
      { "label": "pain", "confidence": 0.94, "landmarks": [[x,y,z],...] }
    """
    data = request.get_json(force=True, silent=True)
    if not data or "frame" not in data:
        return jsonify({"error": "Missing 'frame' field"}), 400

    try:
        # Decode base64 image
        img_b64 = data["frame"]
        # Strip data-URL prefix if present
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(img_b64)

        session_id = data.get("session_id")
        calibration_data = calibration.get(session_id) if session_id else None

        result = gesture_detector.predict(img_bytes, calibration_data=calibration_data)

        if result is None:
            return jsonify({"label": None, "confidence": 0.0, "landmarks": [], "message": "No hand detected"})

        return jsonify(result)

    except Exception as exc:
        logger.exception("Error in /api/detect")
        return jsonify({"error": "Detection failed. Check server logs."}), 500


@app.route("/api/speech", methods=["POST"])
def speech():
    """
    Convert text to speech audio.

    Request JSON:
      { "text": "Hello", "lang": "en" }   lang: en | ta | hi

    Response: audio/mpeg stream (MP3)
    """
    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    lang = data.get("lang", "en").lower()

    if not text:
        return jsonify({"error": "Empty text"}), 400

    supported_langs = {"en", "ta", "hi"}
    if lang not in supported_langs:
        lang = "en"

    try:
        audio_path = tts_engine.synthesize(text, lang=lang)
        return send_file(audio_path, mimetype="audio/mpeg", as_attachment=False)
    except Exception as exc:
        logger.exception("Error in /api/speech")
        return jsonify({"error": "Speech synthesis failed. Check server logs."}), 500


@app.route("/api/mapper", methods=["POST"])
def mapper():
    """
    Map a word or phrase to ISL gesture token sequence.

    Request JSON:
      { "text": "I have pain" }

    Response JSON:
      { "tokens": ["i", "have", "pain"], "sequences": { "pain": [...keypoints...] } }
    """
    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    try:
        result = nlp_mapper.map_phrase(text)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Error in /api/mapper")
        return jsonify({"error": "Mapping failed. Check server logs."}), 500


@app.route("/api/calibrate", methods=["POST"])
def calibrate():
    """
    Store calibration landmarks for a user session.

    Request JSON:
      { "session_id": "abc123", "landmarks": [[x,y,z], ...] }
    """
    data = request.get_json(force=True, silent=True)
    if not data or "session_id" not in data or "landmarks" not in data:
        return jsonify({"error": "Missing fields"}), 400

    session_id = data["session_id"]
    landmarks = data["landmarks"]

    try:
        calibration.store(session_id, landmarks)
        return jsonify({"status": "calibrated", "session_id": session_id})
    except Exception as exc:
        logger.exception("Error in /api/calibrate")
        return jsonify({"error": "Calibration failed. Check server logs."}), 500


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    logger.info(f"Starting SignBridge AI backend on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
