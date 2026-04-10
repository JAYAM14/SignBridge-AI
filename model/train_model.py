"""
SignBridge AI — CNN Model Training Script (train_model.py)

Trains the ISL gesture classifier from keypoints.csv.

Usage:
    cd model/
    python train_model.py [--data ../dataset/keypoints.csv]
                          [--output-dir .]
                          [--epochs 50]
                          [--batch-size 32]

Outputs:
    signbridge_model.h5      — Keras model
    model.tflite             — TFLite quantised model
    label_encoder.pkl        — sklearn LabelEncoder
"""

import os
import sys
import argparse
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras

from backend.classifier import build_cnn_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "keypoints.csv")
DEFAULT_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: str):
    """
    Load keypoints.csv.

    Expected columns:
      label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20
      → 1 label + 63 feature columns

    Returns (X, y, label_encoder).
    """
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError("keypoints.csv must have a 'label' column.")

    y_raw = df["label"].values
    X = df.drop(columns=["label"]).values.astype(np.float32)

    if X.shape[1] != 63:
        raise ValueError(
            f"Expected 63 feature columns (21 landmarks × 3), got {X.shape[1]}."
        )

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    logger.info(f"Loaded {len(X)} samples, {len(le.classes_)} classes: {list(le.classes_)}")
    return X, y, le


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    X, y, le = load_data(args.data)

    num_classes = len(le.classes_)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    model = build_cnn_model(num_classes=num_classes, input_dim=63)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor="val_accuracy"
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=5, min_lr=1e-5, monitor="val_loss"
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "best_model.h5"),
            save_best_only=True,
            monitor="val_accuracy",
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save Keras model
    h5_path = os.path.join(args.output_dir, "signbridge_model.h5")
    model.save(h5_path)
    logger.info(f"Saved Keras model: {h5_path}")

    # Save TFLite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = os.path.join(args.output_dir, "model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    logger.info(f"Saved TFLite model: {tflite_path}")

    # Save label encoder
    le_path = os.path.join(args.output_dir, "label_encoder.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    logger.info(f"Saved label encoder: {le_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train SignBridge AI gesture classifier")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to keypoints.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
