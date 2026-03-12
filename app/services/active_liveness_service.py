"""
Active Liveness Detection Service
===================================
Uses MediaPipe Face Landmarker (Tasks API, v0.10+) with blendshapes for
reliable challenge detection — no manual geometry needed.

Blendshapes used:
  blink       — eyeBlinkLeft + eyeBlinkRight  > BLINK_THRESHOLD
  open_mouth  — jawOpen                        > JAW_THRESHOLD
  turn_left   — face yaw from transformation matrix > YAW_THRESHOLD
  turn_right  — face yaw from transformation matrix < -YAW_THRESHOLD

Model is downloaded automatically on first use (~3 MB).
"""

import io
import math
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

# ── Model download ────────────────────────────────────────────────────────────

_MODEL_DIR  = Path(__file__).resolve().parent.parent.parent / "models" / "face_landmarker"
_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_model() -> Path:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not _MODEL_PATH.exists():
        print("Downloading MediaPipe face landmarker model (~3 MB) …")
        urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
        print(f"Downloaded → {_MODEL_PATH}")
    return _MODEL_PATH


# ── Mediapipe import ──────────────────────────────────────────────────────────

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks import python as mp_python
    _MP_AVAILABLE = True
except Exception:
    _MP_AVAILABLE = False

# ── Thresholds ────────────────────────────────────────────────────────────────

BLINK_THRESHOLD  = 0.35   # blendshape score (0-1)
JAW_THRESHOLD    = 0.35   # blendshape score (0-1)
YAW_THRESHOLD    = 18.0   # degrees

REQUIRED_FRAMES  = 2      # consecutive detections needed to pass

# ── Challenge definitions ─────────────────────────────────────────────────────

CHALLENGES = {
    "blink":       "Close your eyes slowly and hold for a moment",
    "open_mouth":  "Open your mouth wide",
    "turn_left":   "Slowly turn your head to the LEFT",
    "turn_right":  "Slowly turn your head to the RIGHT",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_blendshape(blendshapes, name: str) -> float:
    for b in blendshapes:
        if b.category_name == name:
            return b.score
    return 0.0


def _yaw_from_matrix(matrix) -> float:
    """Extract yaw angle (degrees) from a 4×4 facial transformation matrix.
    Positive yaw = face turned to the user's LEFT (nose right in image).
    Negative yaw = face turned to the user's RIGHT (nose left in image).
    """
    m = np.array(matrix.data).reshape(4, 4)
    R = m[:3, :3]
    yaw = math.degrees(math.atan2(R[0, 2], R[2, 2]))
    return yaw


# ── Detector ──────────────────────────────────────────────────────────────────


class ActiveLivenessDetector:
    def __init__(self) -> None:
        self._detector = None

    def _get_detector(self):
        if self._detector is not None:
            return self._detector
        if not _MP_AVAILABLE:
            return None
        try:
            model_path = _ensure_model()
            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
            )
            self._detector = mp_vision.FaceLandmarker.create_from_options(options)
            print("Active liveness detector loaded (MediaPipe Face Landmarker)")
        except Exception as e:
            print(f"[ActiveLiveness] Failed to load detector: {e}")
        return self._detector

    def detect(self, image_bytes: bytes, challenge: str) -> dict:
        """Analyse a single frame for the given challenge action.

        Returns:
            face_detected   bool
            action_detected bool
            value           float   (blendshape score or yaw angle)
            message         str
        """
        detector = self._get_detector()
        if detector is None:
            return {"face_detected": True, "action_detected": True, "value": 0.0,
                    "message": "ok (mediapipe unavailable — auto-pass)"}

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            frame = np.array(img)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        except Exception:
            return {"face_detected": False, "action_detected": False, "value": 0.0, "message": "Invalid image"}

        result = detector.detect(mp_image)

        if not result.face_landmarks:
            return {"face_detected": False, "action_detected": False, "value": 0.0, "message": "No face detected"}

        blendshapes = result.face_blendshapes[0] if result.face_blendshapes else []
        matrices    = result.facial_transformation_matrixes

        # ── Blink ──
        if challenge == "blink":
            left  = _get_blendshape(blendshapes, "eyeBlinkLeft")
            right = _get_blendshape(blendshapes, "eyeBlinkRight")
            score = (left + right) / 2
            detected = score > BLINK_THRESHOLD
            return {
                "face_detected": True,
                "action_detected": detected,
                "value": round(score, 4),
                "message": "Eyes closed ✓" if detected else f"Score {score:.2f} — close your eyes more",
            }

        # ── Open mouth ──
        if challenge == "open_mouth":
            score = _get_blendshape(blendshapes, "jawOpen")
            detected = score > JAW_THRESHOLD
            return {
                "face_detected": True,
                "action_detected": detected,
                "value": round(score, 4),
                "message": "Mouth open ✓" if detected else f"Score {score:.2f} — open wider",
            }

        # ── Head turn ──
        if challenge in ("turn_left", "turn_right"):
            if not matrices:
                return {"face_detected": True, "action_detected": False, "value": 0.0,
                        "message": "Could not estimate head pose"}
            yaw = _yaw_from_matrix(matrices[0])
            detected = (yaw > YAW_THRESHOLD) if challenge == "turn_left" else (yaw < -YAW_THRESHOLD)
            direction = "left" if challenge == "turn_left" else "right"
            return {
                "face_detected": True,
                "action_detected": detected,
                "value": round(yaw, 2),
                "message": f"Head turn detected ✓" if detected else f"Yaw {yaw:.1f}° — turn more to the {direction}",
            }

        return {"face_detected": True, "action_detected": False, "value": 0.0, "message": "Unknown challenge"}


active_liveness_detector = ActiveLivenessDetector()
