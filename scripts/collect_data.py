"""
SignBridge AI — Webcam Data Collection (collect_data.py)

Captures 30 seconds of hand gesture data per sign from a webcam,
using the TFLite hand landmark model to extract 21 landmarks per frame.

Usage:
    python scripts/collect_data.py --sign help --duration 30
    python scripts/collect_data.py --sign pain  --duration 30

Output:
    Appends rows to dataset/keypoints.csv with columns:
      label, x0, y0, z0, ..., x20, y20, z20
"""

import os
import sys
import time
import argparse
import logging

import cv2
import numpy as np

# Allow imports from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.gesture_detector import GestureDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TFLITE_PATH = os.path.join(BASE_DIR, "hand_landmark.tflite")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "model", "signbridge_model.h5")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
CSV_PATH = os.path.join(BASE_DIR, "dataset", "keypoints.csv")

WINDOW_NAME = "SignBridge AI — Data Collection"


def collect(sign: str, duration: int):
    detector = GestureDetector(
        tflite_model_path=TFLITE_PATH,
        classifier_model_path=CLASSIFIER_PATH,
        label_encoder_path=LABEL_ENCODER_PATH,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam.")
        sys.exit(1)

    # CSV header
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    write_header = not os.path.exists(CSV_PATH)
    csv_file = open(CSV_PATH, "a")
    if write_header:
        cols = ["label"] + [f"{c}{i}" for i in range(21) for c in ("x", "y", "z")]
        csv_file.write(",".join(cols) + "\n")

    frame_count = 0
    start = None

    logger.info(f"Collecting data for sign: '{sign}' | Duration: {duration}s")
    logger.info("Press SPACE to start recording, Q to quit.")

    recording = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()

            if recording:
                elapsed = time.time() - start
                remaining = max(0, duration - elapsed)

                # Encode frame and run detection
                _, buf = cv2.imencode(".jpg", frame)
                result = detector.predict(buf.tobytes())

                if result and result.get("landmarks"):
                    lm = np.array(result["landmarks"]).flatten()
                    row = [sign] + lm.tolist()
                    csv_file.write(",".join(map(str, row)) + "\n")
                    csv_file.flush()
                    frame_count += 1

                    # Draw landmarks on frame
                    h, w = frame.shape[:2]
                    landmarks = np.array(result["landmarks"])
                    for lx, ly, _ in landmarks:
                        cx, cy = int(lx * w), int(ly * h)
                        cv2.circle(display, (cx, cy), 3, (0, 255, 100), -1)

                cv2.putText(display, f"RECORDING: {remaining:.1f}s | Frames: {frame_count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if elapsed >= duration:
                    logger.info(f"Collection complete: {frame_count} frames for '{sign}'")
                    break
            else:
                cv2.putText(display, f"Sign: {sign} | Press SPACE to start",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" ") and not recording:
                recording = True
                start = time.time()
                logger.info("Recording started!")
    finally:
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Saved {frame_count} samples to {CSV_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect ISL gesture data")
    parser.add_argument("--sign", required=True, help="Sign label to collect")
    parser.add_argument("--duration", type=int, default=30, help="Collection duration (seconds)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect(args.sign, args.duration)
