# eKYC System

An electronic Know Your Customer (eKYC) verification system built with FastAPI. It verifies a person's identity by comparing a government-issued ID or passport photo against a live selfie, with anti-spoofing liveness detection to prevent photo/screen attacks.

## Features

- **Face Matching** — Compares the face on a government ID/passport with a live selfie using `face-recognition`
- **Liveness Detection** — Anti-spoofing powered by MiniFASNetV2-SE (ONNX), rejects photos or screens
- **Session-based Verification** — Upload a document once and verify against multiple webcam frames
- **REST API** — Clean FastAPI endpoints with automatic Swagger docs at `/docs`
- **Built-in Frontend** — Static web UI served from `/`

## Tech Stack

- **Backend:** FastAPI + Uvicorn
- **Face Recognition:** `face-recognition` (dlib-based)
- **Liveness Detection:** MiniFASNetV2-SE via ONNX Runtime
- **Image Processing:** OpenCV, Pillow, NumPy

## Project Structure

```
ekyc-system/
├── app/
│   ├── main.py               # FastAPI app setup
│   ├── routes/
│   │   ├── verify.py         # POST /api/verify (single-shot)
│   │   └── session.py        # POST /api/document/upload & /api/verify/frame
│   ├── services/
│   │   ├── face_service.py   # Face encoding & comparison
│   │   ├── antispoof_service.py  # Liveness detection (ONNX)
│   │   └── session_store.py  # In-memory session management
│   └── utils/
│       └── image_utils.py    # Image loading & validation
├── models/
│   └── anti_spoof/           # ONNX model files (auto-downloaded if missing)
├── static/                   # Frontend (HTML/CSS/JS)
├── requirements.txt
└── run.py                    # Entry point
```

## Getting Started

### Prerequisites

- Python 3.10+
- CMake (required by `dlib` for face-recognition)

### Installation

```bash
# Clone the repository
git clone https://github.com/Laligence-Dev/ekyc-system.git
cd ekyc-system

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python run.py
```

The server starts at `http://localhost:8000`.

- **API docs:** `http://localhost:8000/docs`
- **Frontend:** `http://localhost:8000`
- **Health check:** `http://localhost:8000/api/health`

> The anti-spoofing ONNX model is downloaded automatically on first startup if not present in `models/anti_spoof/`.

## API Endpoints

### Single-shot Verification

```
POST /api/verify
```

Upload a document image and a selfie in one request.

| Field      | Type | Description                        |
|------------|------|------------------------------------|
| `document` | File | Government ID or passport image    |
| `selfie`   | File | Live selfie photo                  |

**Response:**
```json
{
  "match": true,
  "distance": 0.38,
  "is_live": true,
  "liveness_score": 1.23,
  "document_faces_detected": 1,
  "selfie_faces_detected": 1
}
```

---

### Session-based Verification (for webcam flows)

**Step 1 — Upload document:**
```
POST /api/document/upload
```
Returns a `session_id`.

**Step 2 — Verify webcam frame:**
```
POST /api/verify/frame
```

| Field        | Type   | Description              |
|--------------|--------|--------------------------|
| `frame`      | File   | Webcam frame image       |
| `session_id` | string | Session ID from step 1   |

## License

&copy; Laligence Dev. All rights reserved.
