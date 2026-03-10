"""
Face anti-spoofing / liveness detection using MiniFASNetV2-SE (ONNX).

Based on: https://github.com/SuriAI/face-antispoof-onnx
Model:    MiniFASNetV2-SE quantized (~600 KB, 98% accuracy on CelebA-Spoof)
Input:    128x128 RGB face crop
Output:   2 logits [real_logit, spoof_logit]
Decision: logit_diff = real_logit - spoof_logit  >=  threshold → real
"""

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "anti_spoof"

MODEL_URL = (
    "https://raw.githubusercontent.com/SuriAI/face-antispoof-onnx/"
    "refs/heads/main/models/best_model_quantized.onnx"
)
MODEL_NAME = "best_model_quantized.onnx"

MODEL_IMG_SIZE = 128
BBOX_EXPANSION_FACTOR = 1.5
LOGIT_THRESHOLD = 0.0  # logit_diff >= 0 means real


# ──────────────────────────────────────────────
#  Preprocessing (matches SuriAI pipeline exactly)
# ──────────────────────────────────────────────

def _crop_face(
    image: np.ndarray,
    face_location: tuple[int, int, int, int],
    expansion: float = BBOX_EXPANSION_FACTOR,
) -> np.ndarray:
    """Extract a square face crop with expansion and reflection padding.

    Args:
        image: Full RGB image (HWC uint8).
        face_location: (top, right, bottom, left) from face_recognition.
        expansion: Bbox expansion factor around the face center.

    Returns:
        Square RGB crop of the face region.
    """
    top, right, bottom, left = face_location

    # Convert to x, y, w, h (absolute coordinates, not offsets)
    x1, y1, x2, y2 = left, top, right, bottom
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return image

    max_dim = max(w, h)
    center_x = x1 + w / 2
    center_y = y1 + h / 2

    # Expanded square crop
    crop_size = int(max_dim * expansion)
    cx = int(center_x - crop_size / 2)
    cy = int(center_y - crop_size / 2)

    img_h, img_w = image.shape[:2]

    # Clamp to image bounds
    crop_x1 = max(0, cx)
    crop_y1 = max(0, cy)
    crop_x2 = min(img_w, cx + crop_size)
    crop_y2 = min(img_h, cy + crop_size)

    # Padding needed if bbox goes outside image
    top_pad = max(0, -cy)
    left_pad = max(0, -cx)
    bottom_pad = max(0, (cy + crop_size) - img_h)
    right_pad = max(0, (cx + crop_size) - img_w)

    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

    if top_pad or bottom_pad or left_pad or right_pad:
        cropped = cv2.copyMakeBorder(
            cropped, top_pad, bottom_pad, left_pad, right_pad,
            cv2.BORDER_REFLECT_101,
        )

    return cropped


def _preprocess(face_crop: np.ndarray, img_size: int = MODEL_IMG_SIZE) -> np.ndarray:
    """Resize with letterboxing, normalize to [0,1], convert to NCHW.

    Matches SuriAI preprocess() exactly.
    """
    old_h, old_w = face_crop.shape[:2]
    ratio = float(img_size) / max(old_h, old_w)
    new_h, new_w = int(old_h * ratio), int(old_w * ratio)

    interp = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(face_crop, (new_w, new_h), interpolation=interp)

    # Letterbox padding (center the image, fill with reflection)
    delta_w = img_size - new_w
    delta_h = img_size - new_h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_REFLECT_101
    )

    # HWC uint8 → CHW float32 [0, 1]
    tensor = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0)  # add batch dim


# ──────────────────────────────────────────────
#  Model download
# ──────────────────────────────────────────────

def _ensure_model_downloaded() -> Path:
    """Download the ONNX model if it doesn't exist locally."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / MODEL_NAME
    if not path.exists():
        print(f"Downloading {MODEL_NAME} from SuriAI...")
        urllib.request.urlretrieve(MODEL_URL, str(path))
        print(f"Downloaded {MODEL_NAME} ({path.stat().st_size / 1024:.0f} KB)")
    return path


# ──────────────────────────────────────────────
#  Liveness detector
# ──────────────────────────────────────────────

class LivenessDetector:
    """MiniFASNetV2-SE anti-spoofing via ONNX Runtime.

    Uses logit-difference classification:
        logit_diff = real_logit - spoof_logit
        is_live = logit_diff >= threshold
    """

    def __init__(self) -> None:
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Download model (if needed) and load ONNX session."""
        if self._initialized:
            return

        model_path = _ensure_model_downloaded()

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._initialized = True
        print("Liveness detector loaded (MiniFASNetV2-SE quantized, 128x128)")

    def check_liveness(
        self,
        image: np.ndarray,
        face_location: tuple[int, int, int, int],
        threshold: float = LOGIT_THRESHOLD,
    ) -> dict:
        """Run liveness detection on a face region.

        Args:
            image: Full RGB image as numpy array (HWC uint8).
            face_location: (top, right, bottom, left) from face_recognition.
            threshold: Logit-difference threshold (default 0.0).

        Returns:
            dict with is_live, liveness_score, real_logit, spoof_logit.
        """
        self.initialize()

        # Crop face with expansion
        face_crop = _crop_face(image, face_location, BBOX_EXPANSION_FACTOR)

        # Preprocess to model input (1, 3, 128, 128)
        tensor = _preprocess(face_crop, MODEL_IMG_SIZE)

        # Inference
        logits = self._session.run(None, {self._input_name: tensor})[0][0]

        real_logit = float(logits[0])
        spoof_logit = float(logits[1])
        logit_diff = real_logit - spoof_logit

        is_live = logit_diff >= threshold

        return {
            "is_live": is_live,
            "liveness_score": round(logit_diff, 4),
            "real_logit": round(real_logit, 4),
            "spoof_logit": round(spoof_logit, 4),
            "threshold": threshold,
        }


# Singleton
liveness_detector = LivenessDetector()
