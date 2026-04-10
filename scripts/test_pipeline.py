"""
SignBridge AI — End-to-End Pipeline Test (test_pipeline.py)

Tests:
  1. Backend API reachability (status endpoint)
  2. Gesture detection (/api/detect) with a synthetic frame
  3. TTS synthesis (/api/speech)
  4. NLP mapper (/api/mapper)
  5. Calibration (/api/calibrate)
  6. GestureDetector module directly (unit test)
  7. TTSEngine module directly (unit test)
  8. NLPMapper module directly (unit test)

Usage:
    # With server running:
    python scripts/test_pipeline.py --base-url http://localhost:5000

    # Module unit tests only (no server required):
    python scripts/test_pipeline.py --unit-only
"""

import os
import sys
import base64
import json
import argparse
import logging
import io
import time

import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PASS = "✅ PASS"
FAIL = "❌ FAIL"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get(url: str, timeout: int = 5):
    try:
        import urllib.request
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except Exception as exc:
        return None, str(exc)


def http_post(url: str, data: dict, timeout: int = 10):
    try:
        import urllib.request
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except Exception as exc:
        return None, str(exc)


def make_test_frame() -> str:
    """Create a base64-encoded synthetic frame (black image)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a white hand-like shape for skin detection fallback
    cv2.circle(frame, (320, 300), 80, (200, 180, 170), -1)
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf.tobytes()).decode()


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------

def test_status(base_url: str) -> bool:
    status, data = http_get(f"{base_url}/api/status")
    ok = status == 200 and isinstance(data, dict) and data.get("status") == "ok"
    print(f"  {PASS if ok else FAIL}  GET /api/status  → {data}")
    return ok


def test_detect(base_url: str) -> bool:
    frame_b64 = make_test_frame()
    status, data = http_post(f"{base_url}/api/detect", {"frame": frame_b64})
    ok = status == 200 and isinstance(data, dict)
    print(f"  {PASS if ok else FAIL}  POST /api/detect → {data}")
    return ok


def test_speech(base_url: str) -> bool:
    try:
        import urllib.request
        body = json.dumps({"text": "I have pain", "lang": "en"}).encode()
        req = urllib.request.Request(
            f"{base_url}/api/speech",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            ok = r.status == 200 and len(r.read()) > 0
    except Exception as exc:
        ok = False
        logger.warning(f"Speech test error: {exc}")
    print(f"  {PASS if ok else FAIL}  POST /api/speech → audio bytes")
    return ok


def test_mapper(base_url: str) -> bool:
    status, data = http_post(f"{base_url}/api/mapper", {"text": "I have chest pain"})
    ok = status == 200 and "tokens" in data
    print(f"  {PASS if ok else FAIL}  POST /api/mapper → {data.get('tokens') if isinstance(data, dict) else data}")
    return ok


def test_calibrate(base_url: str) -> bool:
    lm = np.random.rand(21, 3).tolist()
    status, data = http_post(
        f"{base_url}/api/calibrate",
        {"session_id": "test_session_001", "landmarks": lm},
    )
    ok = status == 200 and isinstance(data, dict) and data.get("status") == "calibrated"
    print(f"  {PASS if ok else FAIL}  POST /api/calibrate → {data}")
    return ok


# ---------------------------------------------------------------------------
# Unit tests (no server required)
# ---------------------------------------------------------------------------

def unit_test_nlp_mapper() -> bool:
    from backend.nlp_mapper import NLPMapper
    kp_dir = os.path.join(BASE_DIR, "frontend", "assets", "keypoints")
    mapper = NLPMapper(keypoints_dir=kp_dir)

    tokens, unknown = mapper._tokenise("I have chest pain and I need water")
    expected_signs = {"chest_pain", "water"}
    ok = bool(expected_signs.intersection(set(tokens)))
    print(f"  {PASS if ok else FAIL}  NLPMapper.tokenise → tokens={tokens}, unknown={unknown}")
    return ok


def unit_test_tts() -> bool:
    from backend.tts_engine import TTSEngine
    cache_dir = os.path.join("/tmp", "signbridge_tts_test")
    engine = TTSEngine(cache_dir=cache_dir)
    try:
        path = engine.synthesize("Test", lang="en")
        ok = os.path.exists(path) and os.path.getsize(path) > 0
    except Exception as exc:
        ok = False
        logger.warning(f"TTS unit test error: {exc}")
    print(f"  {PASS if ok else FAIL}  TTSEngine.synthesize → {path if ok else 'failed'}")
    return ok


def unit_test_calibration() -> bool:
    from backend.calibration import Calibration
    cal = Calibration()
    lm = np.random.rand(5, 21, 3).tolist()
    result = cal.store("unit_test_001", lm)
    retrieved = cal.get("unit_test_001")
    ok = retrieved is not None and "offset" in retrieved and "scale" in retrieved
    scale_str = f"{retrieved.get('scale'):.4f}" if retrieved else "N/A"
    print(f"  {PASS if ok else FAIL}  Calibration.store/get → scale={scale_str}")
    return ok


def unit_test_gesture_detector() -> bool:
    """Test GestureDetector with a synthetic frame (no model required)."""
    from backend.gesture_detector import GestureDetector
    detector = GestureDetector(
        tflite_model_path="/nonexistent.tflite",
        classifier_model_path="/nonexistent.h5",
        label_encoder_path="/nonexistent.pkl",
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 300), 80, (200, 180, 170), -1)
    _, buf = cv2.imencode(".jpg", frame)
    result = detector.predict(buf.tobytes())
    # Should return None (no real model) or a dict
    ok = result is None or isinstance(result, dict)
    print(f"  {PASS if ok else FAIL}  GestureDetector.predict → {result}")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_api_tests(base_url: str) -> int:
    results = []
    print("\n== API Tests ==")
    results.append(test_status(base_url))
    results.append(test_detect(base_url))
    results.append(test_speech(base_url))
    results.append(test_mapper(base_url))
    results.append(test_calibrate(base_url))
    passed = sum(results)
    print(f"\nAPI: {passed}/{len(results)} passed")
    return len(results) - passed


def run_unit_tests() -> int:
    results = []
    print("\n== Unit Tests ==")
    results.append(unit_test_nlp_mapper())
    results.append(unit_test_tts())
    results.append(unit_test_calibration())
    results.append(unit_test_gesture_detector())
    passed = sum(results)
    print(f"\nUnit: {passed}/{len(results)} passed")
    return len(results) - passed


def parse_args():
    parser = argparse.ArgumentParser(description="SignBridge AI Pipeline Test")
    parser.add_argument("--base-url", default="http://localhost:5000", help="Backend base URL")
    parser.add_argument("--unit-only", action="store_true", help="Run unit tests only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    failures = 0
    failures += run_unit_tests()

    if not args.unit_only:
        failures += run_api_tests(args.base_url)

    sys.exit(0 if failures == 0 else 1)
