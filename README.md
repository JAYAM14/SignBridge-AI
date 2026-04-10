# SignBridge AI 🤝

**Real-Time, Offline, Two-Way Indian Sign Language (ISL) Communication Platform for Hospitals**

SignBridge AI bridges the communication gap between deaf/mute patients and healthcare providers by enabling:
- **Patient → Doctor**: ISL gestures detected via webcam → spoken text output
- **Doctor → Patient**: Speech/text input → 3D avatar signing in ISL

---

## 🚀 Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Place hand_landmark.tflite in project root (see docs/SETUP.md)

# 3. Launch the system
python run.py
```

The frontend opens automatically at `http://localhost:5000`.

---

## 📁 Project Structure

```
SignBridge-AI/
├── backend/
│   ├── app.py              Flask API server
│   ├── gesture_detector.py TFLite hand landmark + CNN inference
│   ├── classifier.py       CNN model wrapper
│   ├── tts_engine.py       gTTS multilingual TTS (cached)
│   ├── nlp_mapper.py       Word → ISL sign token mapper
│   └── calibration.py      Per-user hand calibration
├── model/
│   └── train_model.py      CNN training script
├── dataset/
│   ├── signs/              Per-sign image folders (20 signs)
│   └── keypoints.csv       Extracted hand landmarks (generated)
├── frontend/
│   ├── index.html          Main two-way UI
│   ├── style.css           Dark navy glassmorphism theme
│   ├── app.js              Module 1 — gesture detection
│   ├── speech.js           Module 2 — speech recognition
│   ├── avatar.js           Three.js 3D avatar
│   ├── calibration.html    Calibration UI
│   └── assets/keypoints/   JSON keypoint sequences (21 signs)
├── mobile/
│   └── android_webview.html Mobile-optimised UI
├── scripts/
│   ├── collect_data.py     Webcam gesture data collection
│   ├── extract_keypoints.py Batch keypoint extraction
│   └── test_pipeline.py    End-to-end pipeline tests
├── docs/
│   ├── SETUP.md
│   └── API_DOCS.md
├── requirements.txt
└── run.py
```

---

## 🧩 Modules

### Module 1 — ISL Gesture → Speech/Text
Captures webcam frames → runs `hand_landmark.tflite` via `tf.lite.Interpreter` → extracts 21 landmarks → CNN classifier → label + confidence → gTTS speech output.

### Module 2 — Speech/Text → ISL Avatar
Web Speech API → text → NLP mapper → ISL keypoint sequences → Three.js 3D avatar animation.

### Module 3 — Universal Sign Concept Layer (USCL) *(Architecture)*
Cross-language translation architecture: ISL → USCL → ASL/BSL/JSL. See `docs/SETUP.md`.

---

## 🩺 Supported Signs (20)
`help`, `pain`, `headache`, `stomach_pain`, `chest_pain`, `emergency`, `stop`,
`call_doctor`, `no_pain`, `head`, `chest`, `stomach`, `back`, `hand`, `leg`,
`yes`, `no`, `thank_you`, `water`, `medicine`

---

## ⚙️ Tech Stack
- **Backend**: Python 3.8+, Flask 2.3+, TensorFlow Lite, OpenCV, gTTS
- **Frontend**: HTML5, CSS3, Vanilla JS, Three.js (WebGL)
- **AI**: `hand_landmark.tflite` + custom CNN-LSTM classifier

---

## 📄 License
MIT License