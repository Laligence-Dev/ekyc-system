"""
Screen Replay Attack Detector
==============================
Detects when a user presents a face displayed on a digital screen
(phone, tablet, monitor) instead of a real live face.

No model required — pure OpenCV + numpy signal processing.

Signals (weighted fusion):
  1. FFT moiré      (0.30) — periodic peaks from screen pixel grid
  2. Bezel / edge   (0.25) — rectangular border around face context
  3. Specular       (0.20) — large flat glare vs. point highlights on skin
  4. Color corr     (0.15) — gamma-stacking raises R/G/B channel correlation
  5. Temporal       (0.10) — luminance oscillation from screen refresh rate

Decision:
  score >= 0.65 → is_screen_attack = True  (hard reject)
  score >= 0.45 → suspicious = True        (soft flag: require both detectors)
  score <  0.45 → clear                    (delegate to MiniFASNet only)
"""

import threading
from collections import deque

import cv2
import numpy as np

# ── Thresholds ────────────────────────────────────────────────────────────────

HARD_REJECT_THRESHOLD = 0.65
SOFT_FLAG_THRESHOLD   = 0.45

# ── Weights ───────────────────────────────────────────────────────────────────

W_FFT      = 0.30
W_BEZEL    = 0.25
W_SPECULAR = 0.20
W_COLOR    = 0.15
W_TEMPORAL = 0.10

# ── FFT parameters ────────────────────────────────────────────────────────────

FFT_SIZE              = 128   # resize face crop before FFT
FFT_LOW_MASK_R        = 8     # exclude DC lobe radius
FFT_HIGH_MASK_R       = 55    # exclude noise floor radius
FFT_PEAK_THRESHOLD    = 0.75  # fraction of max magnitude to count as peak
FFT_MIN_PEAKS         = 3     # peaks needed to score positively

# ── Bezel parameters ─────────────────────────────────────────────────────────

BEZEL_EXPAND          = 0.45  # expand face bbox by this fraction for context
BEZEL_CANNY_LOW       = 50
BEZEL_CANNY_HIGH      = 150
BEZEL_HOUGH_THRESH    = 40
BEZEL_MIN_LINE        = 60
BEZEL_MAX_GAP         = 10
BEZEL_ANGLE_TOL       = 8     # degrees from 0° or 90°

# ── Specular parameters ───────────────────────────────────────────────────────

SPECULAR_PERCENTILE   = 97.0
SPECULAR_AREA_MIN     = 0.08  # highlight area / face area threshold

# ── Color parameters ──────────────────────────────────────────────────────────

COLOR_CORR_THRESH     = 0.96  # cross-channel correlation above = suspicious
COLOR_STD_THRESH      = 15.0  # low std also required

# ── Temporal parameters ───────────────────────────────────────────────────────

TEMPORAL_BUF          = 5
TEMPORAL_CV_THRESH    = 0.008  # coefficient of variation threshold


# ── Helpers ───────────────────────────────────────────────────────────────────

def _crop_face(image: np.ndarray, face_location: tuple, expand: float = 0.0) -> np.ndarray:
    top, right, bottom, left = face_location
    h, w = image.shape[:2]
    if expand > 0:
        dh = int((bottom - top) * expand)
        dw = int((right - left) * expand)
        top    = max(0, top - dh)
        bottom = min(h, bottom + dh)
        left   = max(0, left - dw)
        right  = min(w, right + dw)
    return image[top:bottom, left:right]


def _to_gray(crop: np.ndarray) -> np.ndarray:
    if crop.ndim == 2:
        return crop
    return cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)


# ── Signal 1: FFT moiré ───────────────────────────────────────────────────────

def _fft_score(face_crop: np.ndarray) -> float:
    gray = _to_gray(face_crop)
    if gray.size == 0:
        return 0.0

    resized = cv2.resize(gray, (FFT_SIZE, FFT_SIZE)).astype(np.float32)

    # Hann window reduces spectral leakage from image edges
    hann = np.outer(np.hanning(FFT_SIZE), np.hanning(FFT_SIZE))
    windowed = resized * hann

    f       = np.fft.fft2(windowed)
    fshift  = np.fft.fftshift(f)
    mag     = np.log1p(np.abs(fshift))

    # Annular mask: exclude DC lobe and noise floor
    cy, cx  = FFT_SIZE // 2, FFT_SIZE // 2
    Y, X    = np.ogrid[:FFT_SIZE, :FFT_SIZE]
    dist    = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask    = (dist >= FFT_LOW_MASK_R) & (dist <= FFT_HIGH_MASK_R)

    region  = mag * mask
    if region.max() == 0:
        return 0.0

    norm    = region / region.max()
    peaks   = int((norm > FFT_PEAK_THRESHOLD).sum())

    return min(1.0, peaks / (FFT_MIN_PEAKS * 4))


# ── Signal 2: Bezel / screen edge ────────────────────────────────────────────

def _bezel_score(image: np.ndarray, face_location: tuple) -> float:
    context = _crop_face(image, face_location, expand=BEZEL_EXPAND)
    if context.size == 0:
        return 0.0

    gray    = _to_gray(context)
    edges   = cv2.Canny(gray, BEZEL_CANNY_LOW, BEZEL_CANNY_HIGH)
    lines   = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=BEZEL_HOUGH_THRESH,
        minLineLength=BEZEL_MIN_LINE,
        maxLineGap=BEZEL_MAX_GAP,
    )

    if lines is None:
        return 0.0

    h_lines, v_lines = 0, 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle <= BEZEL_ANGLE_TOL or angle >= (180 - BEZEL_ANGLE_TOL):
            h_lines += 1
        elif abs(angle - 90) <= BEZEL_ANGLE_TOL:
            v_lines += 1

    # Rectangle requires both horizontal and vertical pairs
    has_rect = h_lines >= 2 and v_lines >= 2
    total    = h_lines + v_lines

    if not has_rect:
        return min(0.3, total * 0.05)
    return min(1.0, 0.5 + total * 0.05)


# ── Signal 3: Specular highlights ────────────────────────────────────────────

def _specular_score(face_crop: np.ndarray) -> float:
    gray = _to_gray(face_crop)
    if gray.size == 0:
        return 0.0

    threshold   = np.percentile(gray, SPECULAR_PERCENTILE)
    mask        = (gray >= threshold).astype(np.uint8) * 255
    area_ratio  = mask.sum() / 255 / gray.size

    if area_ratio < SPECULAR_AREA_MIN:
        return 0.0

    # Convex hull fill — screens have large diffuse glare (high fill)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    all_pts = np.vstack(contours)
    hull    = cv2.convexHull(all_pts)
    hull_area = cv2.contourArea(hull)
    bbox_area = gray.shape[0] * gray.shape[1]

    fill = hull_area / (bbox_area + 1e-8)
    return min(1.0, fill * 2.0) if area_ratio >= SPECULAR_AREA_MIN else 0.0


# ── Signal 4: Color channel correlation ──────────────────────────────────────

def _color_score(face_crop: np.ndarray) -> float:
    if face_crop.ndim != 3 or face_crop.shape[2] < 3:
        return 0.0

    r = face_crop[:, :, 0].flatten().astype(np.float32)
    g = face_crop[:, :, 1].flatten().astype(np.float32)
    b = face_crop[:, :, 2].flatten().astype(np.float32)

    rg_corr = float(np.corrcoef(r, g)[0, 1])
    gb_corr = float(np.corrcoef(g, b)[0, 1])
    mean_corr = (rg_corr + gb_corr) / 2.0

    stds = [r.std(), g.std(), b.std()]
    mean_std = float(np.mean(stds))

    # Both high correlation AND low std required to avoid false positives
    if mean_corr < COLOR_CORR_THRESH or mean_std > COLOR_STD_THRESH:
        return 0.0

    corr_excess = (mean_corr - COLOR_CORR_THRESH) / (1.0 - COLOR_CORR_THRESH + 1e-8)
    return min(1.0, corr_excess * 2.0)


# ── Screen Replay Detector ────────────────────────────────────────────────────

class ScreenReplayDetector:
    """Heuristic screen replay attack detector using five signal fusion."""

    def __init__(self) -> None:
        self._buffers: dict[str, deque] = {}
        self._lock = threading.Lock()

    def check(
        self,
        image: np.ndarray,
        face_location: tuple[int, int, int, int],
        session_id: str | None = None,
    ) -> dict:
        """
        Args:
            image:         Full RGB frame (HWC uint8).
            face_location: (top, right, bottom, left) from face_recognition.
            session_id:    Optional session ID for temporal signal.

        Returns:
            is_screen_attack  bool
            suspicious        bool
            screen_score      float
            signals           dict  (per-signal scores for diagnostics)
        """
        face_crop = _crop_face(image, face_location)

        if face_crop.size == 0:
            return self._result(0.0, {})

        s_fft      = _fft_score(face_crop)
        s_bezel    = _bezel_score(image, face_location)
        s_specular = _specular_score(face_crop)
        s_color    = _color_score(face_crop)
        s_temporal = self._temporal_score(_to_gray(face_crop), session_id)

        signals = {
            "fft":      round(s_fft, 4),
            "bezel":    round(s_bezel, 4),
            "specular": round(s_specular, 4),
            "color":    round(s_color, 4),
            "temporal": round(s_temporal, 4),
        }

        score = (
            W_FFT      * s_fft      +
            W_BEZEL    * s_bezel    +
            W_SPECULAR * s_specular +
            W_COLOR    * s_color    +
            W_TEMPORAL * s_temporal
        )

        return self._result(score, signals)

    def _result(self, score: float, signals: dict) -> dict:
        return {
            "is_screen_attack": score >= HARD_REJECT_THRESHOLD,
            "suspicious":       score >= SOFT_FLAG_THRESHOLD,
            "screen_score":     round(score, 4),
            "signals":          signals,
        }

    def _temporal_score(self, gray_crop: np.ndarray, session_id: str | None) -> float:
        if session_id is None or gray_crop.size == 0:
            return 0.0

        mean_lum = float(gray_crop.mean())

        with self._lock:
            if session_id not in self._buffers:
                self._buffers[session_id] = deque(maxlen=TEMPORAL_BUF)
            buf = self._buffers[session_id]
            buf.append(mean_lum)

            if len(buf) < 3:
                return 0.0

            arr = np.array(buf, dtype=np.float32)
            mean = arr.mean()
            if mean < 1e-8:
                return 0.0
            cv = arr.std() / mean

        if cv < TEMPORAL_CV_THRESH:
            return 0.0
        return min(1.0, (cv - TEMPORAL_CV_THRESH) / TEMPORAL_CV_THRESH)

    def cleanup_session(self, session_id: str) -> None:
        with self._lock:
            self._buffers.pop(session_id, None)


# Singleton
screen_replay_detector = ScreenReplayDetector()
