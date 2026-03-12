"""
Active Liveness Detection Service
===================================
Uses MediaPipe Face Mesh (468 landmarks) to detect challenge actions:

  blink       — Eye Aspect Ratio drops below threshold (eyes closed)
  open_mouth  — Mouth Aspect Ratio exceeds threshold (mouth open)
  turn_left   — Nose deviates to the left relative to eye midpoint
  turn_right  — Nose deviates to the right relative to eye midpoint
"""

import io

import numpy as np
from PIL import Image

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
    _face_mesh_module = mp.solutions.face_mesh
except ImportError:
    _MP_AVAILABLE = False

# ── Landmark indices ──────────────────────────────────────────────────────────

_LEFT_EYE  = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
_MOUTH_TOP = 13    # inner upper lip
_MOUTH_BOT = 14    # inner lower lip
_MOUTH_L   = 78    # left corner
_MOUTH_R   = 308   # right corner
_NOSE_TIP  = 4
_L_EYE_OUT = 33
_R_EYE_OUT = 263

# ── Thresholds ────────────────────────────────────────────────────────────────

EAR_CLOSED     = 0.20   # EAR below this = eyes closed
MAR_OPEN       = 0.40   # MAR above this = mouth open
TURN_THRESHOLD = 0.07   # nose deviation ratio to eye distance
REQUIRED_FRAMES = 2     # consecutive frames with action detected to pass

# ── Challenge definitions ─────────────────────────────────────────────────────

CHALLENGES = {
    "blink":       "Close your eyes slowly and hold for a moment",
    "open_mouth":  "Open your mouth wide",
    "turn_left":   "Slowly turn your head to the LEFT",
    "turn_right":  "Slowly turn your head to the RIGHT",
}

# ── Geometry helpers ──────────────────────────────────────────────────────────


def _d(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def _pt(lm, idx, w, h):
    return (lm[idx].x * w, lm[idx].y * h)


def _ear(lm, eye_idx, w, h) -> float:
    p = [_pt(lm, i, w, h) for i in eye_idx]
    return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2.0 * _d(p[0], p[3]) + 1e-6)


def _mar(lm, w, h) -> float:
    v = _d(_pt(lm, _MOUTH_TOP, w, h), _pt(lm, _MOUTH_BOT, w, h))
    h_ = _d(_pt(lm, _MOUTH_L, w, h),  _pt(lm, _MOUTH_R, w, h))
    return v / (h_ + 1e-6)


def _nose_dev(lm) -> float:
    """Nose x deviation relative to eye midpoint, normalised by eye span.
    Positive  → nose is to the right in the image  (user turns their LEFT).
    Negative  → nose is to the left  in the image  (user turns their RIGHT).
    """
    le_x = lm[_L_EYE_OUT].x
    re_x = lm[_R_EYE_OUT].x
    mid_x = (le_x + re_x) / 2
    span = abs(re_x - le_x) + 1e-6
    return (lm[_NOSE_TIP].x - mid_x) / span


# ── Detector ──────────────────────────────────────────────────────────────────


class ActiveLivenessDetector:
    def __init__(self) -> None:
        self._mesh = None

    def _get_mesh(self):
        if self._mesh is None and _MP_AVAILABLE:
            self._mesh = _face_mesh_module.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        return self._mesh

    def detect(self, image_bytes: bytes, challenge: str) -> dict:
        """Analyse a single frame for the given challenge action.

        Returns:
            face_detected   bool
            action_detected bool
            value           float   (EAR / MAR / deviation — for debugging)
            message         str
        """
        if not _MP_AVAILABLE:
            return {"face_detected": True, "action_detected": True, "value": 0.0, "message": "ok (mediapipe unavailable)"}

        mesh = self._get_mesh()

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return {"face_detected": False, "action_detected": False, "value": 0.0, "message": "Invalid image"}

        frame = np.array(img)
        h, w = frame.shape[:2]

        results = mesh.process(frame)
        if not results.multi_face_landmarks:
            return {"face_detected": False, "action_detected": False, "value": 0.0, "message": "No face detected"}

        lm = results.multi_face_landmarks[0].landmark

        if challenge == "blink":
            avg_ear = (_ear(lm, _LEFT_EYE, w, h) + _ear(lm, _RIGHT_EYE, w, h)) / 2
            detected = avg_ear < EAR_CLOSED
            return {
                "face_detected": True,
                "action_detected": detected,
                "value": round(avg_ear, 4),
                "message": "Eyes closed ✓" if detected else f"EAR {avg_ear:.3f} — close your eyes more",
            }

        if challenge == "open_mouth":
            mar = _mar(lm, w, h)
            detected = mar > MAR_OPEN
            return {
                "face_detected": True,
                "action_detected": detected,
                "value": round(mar, 4),
                "message": "Mouth open ✓" if detected else f"MAR {mar:.3f} — open wider",
            }

        if challenge in ("turn_left", "turn_right"):
            dev = _nose_dev(lm)
            detected = (dev > TURN_THRESHOLD) if challenge == "turn_left" else (dev < -TURN_THRESHOLD)
            return {
                "face_detected": True,
                "action_detected": detected,
                "value": round(dev, 4),
                "message": "Head turn detected ✓" if detected else f"deviation {dev:.3f} — turn more",
            }

        return {"face_detected": True, "action_detected": False, "value": 0.0, "message": "Unknown challenge"}


active_liveness_detector = ActiveLivenessDetector()
