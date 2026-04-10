# SignBridge AI — Setup Guide

## Prerequisites

- Python 3.8 or later
- pip
- A webcam (for gesture detection)
- A modern browser (Chrome/Edge recommended for Web Speech API)

---

## 1. Install Python Dependencies

```bash
cd SignBridge-AI/
pip install -r requirements.txt
```

---

## 2. Download the TFLite Hand Landmark Model

SignBridge AI uses **only** the TensorFlow Lite hand landmark model — no MediaPipe runtime required.

Place `hand_landmark.tflite` in the **project root** (`SignBridge-AI/hand_landmark.tflite`).

### Download options:

**Option A — MediaPipe model file (recommended)**
```bash
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task \
     -O hand_landmark.tflite
```

**Option B — Manually**
Download from https://developers.google.com/mediapipe/solutions/vision/hand_landmarker and rename the `.task` / `.tflite` file to `hand_landmark.tflite`.

> The model is loaded exclusively via `tf.lite.Interpreter(model_path=...)` — no MediaPipe Python package needed.

---

## 3. Collect Training Data (Optional — for custom model)

Use the data collection script to record 30 seconds of each sign:

```bash
# Collect 30 seconds of "pain" sign
python scripts/collect_data.py --sign pain --duration 30

# Repeat for all 20 signs
python scripts/collect_data.py --sign help
python scripts/collect_data.py --sign headache
# ...
```

Data is saved to `dataset/keypoints.csv`.

### Or use existing images:

If you have images in `dataset/signs/<label>/`, extract keypoints in bulk:

```bash
python scripts/extract_keypoints.py --overwrite
```

---

## 4. Train the Classifier

```bash
python model/train_model.py --epochs 50
```

Outputs:
- `model/signbridge_model.h5`
- `model/model.tflite`
- `model/label_encoder.pkl`

---

## 5. Run the System

```bash
python run.py
```

Or start the backend manually:

```bash
cd backend/
python app.py
```

Then open `http://localhost:5000` in your browser.

---

## 6. Hand Calibration (Recommended)

Before using gesture detection, calibrate for your hand:

1. Click **⚙ Calibrate** in the top bar
2. Or navigate to `http://localhost:5000/calibration.html`
3. Hold your hand in an open-palm pose for 30 seconds

---

## 7. Run Tests

```bash
# Unit tests only (no server needed)
python scripts/test_pipeline.py --unit-only

# Full pipeline (requires running server)
python scripts/test_pipeline.py
```

---

## Supported Languages

| Code | Language | gTTS Code |
|------|----------|-----------|
| `en` | English  | `en`      |
| `ta` | Tamil    | `ta`      |
| `hi` | Hindi    | `hi`      |

---

## Performance Targets

| Metric            | Target  |
|-------------------|---------|
| Detection latency | < 500ms |
| Avatar FPS        | 24+ FPS |
| Startup time      | < 5s    |

---

## Module 3 — USCL Architecture

### Universal Sign Concept Layer (Future Implementation)

```
ISL Sign → USCL Abstract ID → Target Sign Language
```

#### USCL JSON Schema

```json
{
  "uscl_id": "USCL_001",
  "concept": "pain",
  "description": "Physical discomfort or hurt",
  "category": "medical_symptom",
  "mappings": {
    "isl": { "token": "pain", "keypoint_file": "pain.json" },
    "asl": { "token": "asl_pain_23", "reference": "ASL_SignBank_023" },
    "bsl": { "token": "bsl_pain_07", "reference": "BSL_Dict_007" },
    "jsl": { "token": "jsl_pain_11", "reference": "JSL_Dict_011" }
  }
}
```

#### Conversion Pipeline

```
Input ISL Token
     ↓
ISL Vocabulary Lookup → USCL_ID
     ↓
Target Language Resolver (e.g., "asl")
     ↓
Keypoint / Video Sequence Loader
     ↓
Avatar / Output Renderer
```

#### Example Mappings

| Concept     | USCL ID    | ISL     | ASL      | BSL     |
|-------------|------------|---------|----------|---------|
| pain        | USCL_001   | pain    | asl_023  | bsl_007 |
| help        | USCL_002   | help    | asl_001  | bsl_001 |
| emergency   | USCL_003   | emergency | asl_042 | bsl_019 |
| water       | USCL_004   | water   | asl_155  | bsl_088 |

---

## Troubleshooting

**Camera not working:**
- Ensure browser has camera permission
- Use HTTPS or localhost (required for `getUserMedia`)

**`hand_landmark.tflite` errors:**
- Verify the file is in the project root
- Check TensorFlow version: `python -c "import tensorflow as tf; print(tf.__version__)"`

**gTTS network errors:**
- gTTS requires internet for first generation, then caches
- For fully offline: pre-cache common phrases, then disconnect

**Classifier not found:**
- Run `python model/train_model.py` to generate `signbridge_model.h5`
- The system works in landmark-only mode without the classifier
