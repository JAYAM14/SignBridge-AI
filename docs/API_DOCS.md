# SignBridge AI — API Documentation

Base URL: `http://localhost:5000`

All request and response bodies are JSON unless noted.

---

## GET /api/status

Health check. Returns module readiness.

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "modules": {
    "gesture_detector": true,
    "tts_engine": true,
    "nlp_mapper": true
  },
  "timestamp": 1700000000.0
}
```

---

## POST /api/detect

Detect ISL sign from a webcam frame.

**Request:**
```json
{
  "frame": "<base64-encoded JPEG/PNG>",
  "session_id": "optional_session_id"
}
```

- `frame`: Base64-encoded image. Data-URL prefix (`data:image/jpeg;base64,`) is stripped automatically.
- `session_id`: Optional. If a calibration session exists for this ID, calibration adjustments are applied.

**Response (sign detected):**
```json
{
  "label": "pain",
  "confidence": 0.94,
  "landmarks": [
    [0.12, 0.34, 0.01],
    ...
  ]
}
```

**Response (no hand detected):**
```json
{
  "label": null,
  "confidence": 0.0,
  "landmarks": [],
  "message": "No hand detected"
}
```

**Response (low confidence):**
```json
{
  "label": null,
  "confidence": 0.38,
  "landmarks": [[...]],
  "message": "Low confidence"
}
```

**Response (models not loaded):**
```json
{
  "label": "unknown",
  "confidence": 0.0,
  "landmarks": [[...]],
  "message": "Classifier not loaded"
}
```

---

## POST /api/speech

Convert text to MP3 audio.

**Request:**
```json
{
  "text": "I have pain",
  "lang": "en"
}
```

Supported `lang` values: `en`, `ta`, `hi`

**Response:** `audio/mpeg` binary stream (MP3).

The response is cached; repeat requests for the same `text` + `lang` are served from disk.

---

## POST /api/mapper

Map a text phrase to ISL gesture token sequence + keypoint data.

**Request:**
```json
{
  "text": "I have chest pain and need water"
}
```

**Response:**
```json
{
  "tokens": ["chest_pain", "water"],
  "sequences": {
    "chest_pain": {
      "sign": "chest_pain",
      "fps": 24,
      "frames": [
        [[0.0, 0.0, 0.0], ...],
        ...
      ],
      "hand": "right",
      "description": "ISL sign for 'chest_pain'"
    },
    "water": { ... }
  },
  "unknown_tokens": ["i", "have", "and", "need"]
}
```

- `tokens`: Recognised ISL sign tokens (in order)
- `sequences`: Keypoint animation data per token (only for tokens with JSON files)
- `unknown_tokens`: Words that couldn't be mapped to ISL signs

---

## POST /api/calibrate

Store per-user hand calibration data.

**Request:**
```json
{
  "session_id": "user_abc123",
  "landmarks": [
    [[0.1, 0.2, 0.0], ...],
    ...
  ]
}
```

- `session_id`: Unique identifier for the client session (store in `sessionStorage`)
- `landmarks`: Array of 21-landmark observations. Accepts:
  - Single observation: `[[x,y,z] × 21]`
  - Multiple observations: `[[[x,y,z] × 21] × N]`

**Response:**
```json
{
  "status": "calibrated",
  "session_id": "user_abc123"
}
```

---

## Error Responses

All endpoints return structured errors:

```json
{ "error": "Description of the error" }
```

| HTTP Status | Meaning                      |
|-------------|------------------------------|
| 400         | Bad request / missing fields |
| 404         | Endpoint not found           |
| 500         | Internal server error        |

---

## Landmark Format

All landmark arrays follow the 21-joint MediaPipe hand model layout:

| Index | Joint           |
|-------|-----------------|
| 0     | Wrist           |
| 1-4   | Thumb (CMC→TIP) |
| 5-8   | Index (MCP→TIP) |
| 9-12  | Middle (MCP→TIP)|
| 13-16 | Ring (MCP→TIP)  |
| 17-20 | Pinky (MCP→TIP) |

Coordinates are normalised floats in range `[0, 1]` for x/y and near-zero for z.

---

## Supported Sign Labels

```
help, pain, headache, stomach_pain, chest_pain, emergency, stop,
call_doctor, no_pain, head, chest, stomach, back, hand, leg,
yes, no, thank_you, water, medicine
```
