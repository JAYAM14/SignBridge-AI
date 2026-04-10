"""
SignBridge AI — Keypoint Extraction Script (extract_keypoints.py)

Processes images in dataset/signs/<label>/ directories, extracts
21 hand landmarks using the TFLite model, and appends them to
dataset/keypoints.csv.

Usage:
    python scripts/extract_keypoints.py
    python scripts/extract_keypoints.py --signs-dir dataset/signs
"""

import os
import sys
import argparse
import logging

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.gesture_detector import GestureDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TFLITE_PATH = os.path.join(BASE_DIR, "hand_landmark.tflite")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "model", "signbridge_model.h5")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
DEFAULT_SIGNS_DIR = os.path.join(BASE_DIR, "dataset", "signs")
CSV_PATH = os.path.join(BASE_DIR, "dataset", "keypoints.csv")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def extract_all(signs_dir: str, overwrite: bool = False):
    detector = GestureDetector(
        tflite_model_path=TFLITE_PATH,
        classifier_model_path=CLASSIFIER_PATH,
        label_encoder_path=LABEL_ENCODER_PATH,
    )

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    mode = "w" if overwrite else "a"
    csv_file = open(CSV_PATH, mode)
    write_header = overwrite or not os.path.exists(CSV_PATH)
    if write_header:
        cols = ["label"] + [f"{c}{i}" for i in range(21) for c in ("x", "y", "z")]
        csv_file.write(",".join(cols) + "\n")

    total = 0
    skipped = 0

    for label in sorted(os.listdir(signs_dir)):
        label_dir = os.path.join(signs_dir, label)
        if not os.path.isdir(label_dir):
            continue

        count = 0
        for fname in sorted(os.listdir(label_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue

            img_path = os.path.join(label_dir, fname)
            frame = cv2.imread(img_path)
            if frame is None:
                skipped += 1
                continue

            _, buf = cv2.imencode(".jpg", frame)
            result = detector.predict(buf.tobytes())

            if result and result.get("landmarks"):
                lm = np.array(result["landmarks"]).flatten()
                row = [label] + lm.tolist()
                csv_file.write(",".join(map(str, row)) + "\n")
                csv_file.flush()
                count += 1
                total += 1
            else:
                skipped += 1

        if count > 0:
            logger.info(f"  '{label}': {count} keypoints extracted.")

    csv_file.close()
    logger.info(f"Done. Total: {total} rows written, {skipped} skipped.")
    logger.info(f"Output: {CSV_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract hand keypoints from sign images")
    parser.add_argument(
        "--signs-dir", default=DEFAULT_SIGNS_DIR,
        help="Root directory containing per-sign subdirectories"
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_all(args.signs_dir, overwrite=args.overwrite)
