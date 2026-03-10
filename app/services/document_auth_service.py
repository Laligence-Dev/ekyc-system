"""
Document Authenticity Service
==============================

Two-layer defense against fraudulent document uploads:

  Layer 1 — ELA (Error Level Analysis)
    Detects JPEG manipulation artifacts introduced by image editing tools.
    Works immediately with NO trained model. Any edited region causes compression
    inconsistencies that ELA exposes.

  Layer 2 — Document Fraud Classifier (EfficientNet-B0, ONNX)
    Classifies the image into one of three classes:
      0 — real_document  : genuine passport or government ID
      1 — not_document   : selfie, random photo, screenshot, etc.
      2 — fake_document  : edited, spliced, or digitally altered document

    Only active when models/doc_fraud/doc_fraud_detector.onnx is present.
    Train it with:  python scripts/train_doc_fraud_detector.py --help

Decision logic
--------------
  If classifier is available:
    - Reject if top class is not_document  (confidence > threshold)
    - Reject if top class is fake_document (any confidence)
    - Warn  if ELA score is high, regardless of classifier
  If classifier is NOT available:
    - Reject if ELA score is above HARD_REJECT threshold
    - Warn  if ELA score is above WARN threshold
"""

import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "doc_fraud"
_ONNX_PATH = _MODELS_DIR / "doc_fraud_detector.onnx"
_CLASS_NAMES_PATH = _MODELS_DIR / "class_names.txt"

# ── ELA thresholds ────────────────────────────────────────────────────────────

ELA_WARN_THRESHOLD = 0.12        # flag as suspicious
ELA_HARD_REJECT_THRESHOLD = 0.22 # reject without classifier (fallback mode)

# ── classifier thresholds ─────────────────────────────────────────────────────

NOT_DOC_CONFIDENCE_THRESHOLD = 0.60   # reject if model says "not_document" with this confidence
FAKE_DOC_CONFIDENCE_THRESHOLD = 0.50  # reject if model says "fake_document" with this confidence

# ── model input ───────────────────────────────────────────────────────────────

_IMG_SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── ELA ───────────────────────────────────────────────────────────────────────


def _ela_score(image_bytes: bytes, quality: int = 95) -> tuple[float, float]:
    """Compute ELA mean and max-region scores.

    Re-saves the image at a known JPEG quality and measures the pixel
    difference. Unmodified regions are consistent; edited regions show
    higher residual values.

    Returns:
        (mean_score, max_region_score) — both in [0, 1].
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return 0.0, 0.0

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    orig = np.array(img, dtype=np.float32)
    comp = np.array(resaved, dtype=np.float32)

    # Amplify difference (×10) to make edits visible
    diff = np.abs(orig - comp) * 10.0 / 255.0

    mean_score = float(diff.mean())

    # Max score over 16×16 non-overlapping blocks (highlights local tampered regions)
    h, w = diff.shape[:2]
    block = 16
    block_means = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            block_means.append(diff[y : y + block, x : x + block].mean())
    max_region_score = float(max(block_means)) if block_means else mean_score

    return round(mean_score, 4), round(max_region_score, 4)


# ── classifier ────────────────────────────────────────────────────────────────


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _preprocess(image_bytes: bytes) -> np.ndarray | None:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((_IMG_SIZE, _IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        return arr.transpose(2, 0, 1)[None]  # (1, 3, H, W)
    except Exception:
        return None


class _DocumentClassifier:
    def __init__(self) -> None:
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self._classes: list[str] = ["real_document", "not_document", "fake_document"]

    def _try_load(self) -> bool:
        if self._session is not None:
            return True
        if not _ONNX_PATH.exists():
            return False
        try:
            self._session = ort.InferenceSession(
                str(_ONNX_PATH), providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            if _CLASS_NAMES_PATH.exists():
                self._classes = _CLASS_NAMES_PATH.read_text().strip().splitlines()
            print("Document fraud classifier loaded →", _ONNX_PATH)
            return True
        except Exception as e:
            print(f"[DocumentClassifier] Failed to load model: {e}")
            return False

    @property
    def available(self) -> bool:
        return self._try_load()

    def classify(self, image_bytes: bytes) -> dict | None:
        """Run classifier. Returns dict or None if model unavailable / error."""
        if not self._try_load():
            return None
        tensor = _preprocess(image_bytes)
        if tensor is None:
            return None
        try:
            logits = self._session.run(None, {self._input_name: tensor})[0][0]
            probs = _softmax(logits)
            top_idx = int(probs.argmax())
            return {
                "class": self._classes[top_idx],
                "class_index": top_idx,
                "confidence": round(float(probs[top_idx]), 4),
                "scores": {c: round(float(p), 4) for c, p in zip(self._classes, probs)},
            }
        except Exception as e:
            print(f"[DocumentClassifier] Inference error: {e}")
            return None


_classifier = _DocumentClassifier()


# ── public API ────────────────────────────────────────────────────────────────


def check_document_authenticity(image_bytes: bytes) -> dict:
    """Validate that the uploaded image is a genuine, unaltered document.

    Returns a dict:
        is_authentic   bool   — True if the document passes all checks
        reason         str    — Human-readable explanation (only set on failure)
        ela_mean       float  — ELA mean score (lower = less suspicious)
        ela_max_region float  — ELA max block score
        ela_suspicious bool   — True if ELA indicates possible tampering
        classifier     dict | None — classifier output (None if model not loaded)
    """
    ela_mean, ela_max = _ela_score(image_bytes)
    ela_suspicious = ela_mean > ELA_WARN_THRESHOLD or ela_max > ELA_WARN_THRESHOLD * 1.5

    clf_result = _classifier.classify(image_bytes)

    result = {
        "is_authentic": True,
        "reason": None,
        "ela_mean": ela_mean,
        "ela_max_region": ela_max,
        "ela_suspicious": ela_suspicious,
        "classifier": clf_result,
    }

    # ── Layer 2: classifier checks (if model loaded) ──────────────────────────
    if clf_result is not None:
        cls = clf_result["class"]
        conf = clf_result["confidence"]

        if cls == "not_document" and conf >= NOT_DOC_CONFIDENCE_THRESHOLD:
            result["is_authentic"] = False
            result["reason"] = (
                "The uploaded image does not appear to be a government-issued document. "
                "Please upload a clear photo of your passport or ID card."
            )
            return result

        if cls == "fake_document" and conf >= FAKE_DOC_CONFIDENCE_THRESHOLD:
            result["is_authentic"] = False
            result["reason"] = (
                "The document appears to have been digitally altered or is not genuine. "
                "Please upload an unedited photo of your original document."
            )
            return result

    # ── Layer 1: ELA hard-reject (fallback when no model, or extra signal) ────
    if ela_mean > ELA_HARD_REJECT_THRESHOLD or ela_max > ELA_HARD_REJECT_THRESHOLD * 1.5:
        result["is_authentic"] = False
        result["reason"] = (
            "The document image shows signs of digital manipulation. "
            "Please upload an original, unedited photo of your document."
        )
        return result

    return result
