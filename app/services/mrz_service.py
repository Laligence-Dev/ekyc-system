"""
MRZ (Machine Readable Zone) Service
=====================================
Pipeline:
  1. Detection  — OpenCV morphological ops find the MRZ strip
                  (same idea as YOLO: locate a bounding box for the text zone)
  2. OCR        — PaddleOCR reads the cropped MRZ region
  3. Parser     — ICAO 9303 TD3 / TD1 field extraction + check-digit validation
  4. Fallback   — passporteye if the pipeline above finds nothing
"""

import io
import re
from datetime import date, datetime

import cv2
import numpy as np
from PIL import Image

# ── MRZ character helpers ─────────────────────────────────────────────────────

_INVALID_RE  = re.compile(r"[^A-Z0-9<]")
_DIGIT_FIXES = str.maketrans("OIlBSZGT", "01185267")


def _normalize(text: str) -> str:
    t = text.upper().replace(" ", "<").replace(".", "<").replace("-", "<")
    return _INVALID_RE.sub("", t)


def _fix_numeric(s: str) -> str:
    return s.translate(_DIGIT_FIXES)


def _clean(s: str) -> str:
    return s.replace("<", " ").strip()


# ── Check digit (ICAO 9303) ───────────────────────────────────────────────────

_W = [7, 3, 1]
_CV = {c: i for i, c in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
_CV["<"] = 0


def _check_digit(s: str) -> str:
    return str(sum(_CV.get(c, 0) * _W[i % 3] for i, c in enumerate(s)) % 10)


# ── Date helpers ──────────────────────────────────────────────────────────────

def _parse_date(yymmdd: str) -> date | None:
    try:
        return datetime.strptime(yymmdd.strip("<"), "%y%m%d").date()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Detection: find the MRZ strip using morphological operations
# ─────────────────────────────────────────────────────────────────────────────

def _detect_mrz_crop(image_bytes: bytes) -> np.ndarray:
    """
    Locate the MRZ zone in a passport / ID card image.

    Strategy (works like a lightweight detector):
      - Convert to grayscale and upscale to ≥ 1800px wide
      - Search the bottom 45 % of the image (MRZ is always there)
      - Binarise + horizontal dilation to merge text into solid bands
      - Pick the two (TD3) or three (TD1) widest horizontal bands
      - Return the tightly-cropped MRZ strip, padded a little for safety

    Falls back to the bottom 30 % of the image if no bands found.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Upscale so small text is readable
    if img.width < 1800:
        scale = 1800 / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.LANCZOS)

    gray = np.array(img)
    h, w = gray.shape

    # Search only the bottom half — MRZ never appears in the top
    search_start = int(h * 0.55)
    roi = gray[search_start:, :]

    # Binarise (Otsu) then invert so text is white
    _, binary = cv2.threshold(roi, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate horizontally to merge characters in the same line into solid bars
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.04), 1))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Keep bands that span ≥ 60 % of the width (MRZ lines are wide)
    bands = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw >= w * 0.60:
            bands.append((y, ch))

    if len(bands) >= 2:
        bands.sort(key=lambda b: b[0])
        y_top    = max(0, bands[0][0] - 10)
        last     = bands[-1]
        y_bottom = min(roi.shape[0], last[0] + last[1] + 10)
        crop = roi[y_top:y_bottom, :]
    else:
        # Fallback: bottom 30 %
        crop = gray[int(h * 0.70):, :]

    return crop


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — OCR: PaddleOCR on the cropped strip
# ─────────────────────────────────────────────────────────────────────────────

_paddle_ocr = None


def _get_paddle():
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR
        print("Loading PaddleOCR model (first use)…")
        _paddle_ocr = PaddleOCR(
            use_angle_cls=False,   # MRZ is always horizontal
            lang="en",
            show_log=False,
        )
        print("PaddleOCR ready.")
    return _paddle_ocr


def _ocr_mrz_strip(crop: np.ndarray) -> list[str]:
    """
    Run PaddleOCR on the MRZ strip.
    Returns text lines sorted top-to-bottom, normalised to valid MRZ chars.
    """
    ocr = _get_paddle()

    # Upscale the strip so characters are large enough for PaddleOCR
    scale = max(1.0, 120 / crop.shape[0])   # aim for ~120 px tall strip
    if scale > 1.0:
        new_h = int(crop.shape[0] * scale)
        new_w = int(crop.shape[1] * scale)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # PaddleOCR expects BGR or RGB; feed grayscale as-is (it handles it)
    result = ocr.ocr(crop, cls=False)

    lines = []
    if result and result[0]:
        # Sort by vertical centre
        items = sorted(result[0], key=lambda r: (r[0][0][1] + r[0][2][1]) / 2)
        for item in items:
            text = item[1][0]
            norm = _normalize(text)
            if norm:
                lines.append(norm)

    print(f"[MRZ] PaddleOCR raw lines: {lines}")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Parser: TD3 / TD1
# ─────────────────────────────────────────────────────────────────────────────

def _pad(s: str, n: int) -> str:
    return (s + "<" * n)[:n]


def _parse_td3(lines: list[str]) -> dict | None:
    """Try to parse a TD3 (passport, 2 × 44 chars) MRZ."""
    for i in range(len(lines) - 1):
        l1 = _pad(lines[i],     44)
        l2 = _pad(lines[i + 1], 44)

        if l1[0] not in "PIV":
            continue

        doc_num = _fix_numeric(l2[0:9])
        nat     = l2[10:13]
        dob_raw = _fix_numeric(l2[13:19])
        sex     = l2[20] if l2[20] in "MF<" else "<"
        exp_raw = _fix_numeric(l2[21:27])

        v_doc = _check_digit(doc_num) == l2[9]
        v_dob = _check_digit(dob_raw) == l2[19]
        v_exp = _check_digit(exp_raw) == l2[27]
        score = sum([v_doc, v_dob, v_exp])

        names = l1[5:44]
        if "<<" in names:
            sn, gn = names.split("<<", 1)
        else:
            sn, gn = names, ""

        return {
            "format": "TD3",
            "document_type":   _clean(l1[0:2]),
            "country":         _clean(l1[2:5]),
            "surname":         _clean(sn),
            "given_names":     _clean(gn),
            "document_number": _clean(doc_num),
            "nationality":     _clean(nat),
            "date_of_birth":   dob_raw,
            "expiry_date":     exp_raw,
            "sex":             sex,
            "valid_score":     score,
            "valid":           score >= 2,
        }
    return None


def _parse_td1(lines: list[str]) -> dict | None:
    """Try to parse a TD1 (ID card, 3 × 30 chars) MRZ."""
    for i in range(len(lines) - 2):
        l1 = _pad(lines[i],     30)
        l2 = _pad(lines[i + 1], 30)
        l3 = _pad(lines[i + 2], 30)

        if l1[0] not in "IA":
            continue

        doc_num = _fix_numeric(l1[5:14])
        dob_raw = _fix_numeric(l2[0:6])
        sex     = l2[7] if l2[7] in "MF<" else "<"
        exp_raw = _fix_numeric(l2[8:14])
        nat     = l2[15:18]

        v_doc = _check_digit(doc_num) == l1[14]
        v_dob = _check_digit(dob_raw) == l2[6]
        v_exp = _check_digit(exp_raw) == l2[14]
        score = sum([v_doc, v_dob, v_exp])

        names = l3
        if "<<" in names:
            sn, gn = names.split("<<", 1)
        else:
            sn, gn = names, ""

        return {
            "format": "TD1",
            "document_type":   _clean(l1[0:2]),
            "country":         _clean(l1[2:5]),
            "surname":         _clean(sn),
            "given_names":     _clean(gn),
            "document_number": _clean(doc_num),
            "nationality":     _clean(nat),
            "date_of_birth":   dob_raw,
            "expiry_date":     exp_raw,
            "sex":             sex,
            "valid_score":     score,
            "valid":           score >= 2,
        }
    return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Fallback: passporteye
# ─────────────────────────────────────────────────────────────────────────────

def _passporteye_fallback(image_bytes: bytes) -> dict | None:
    try:
        from passporteye import read_mrz
        mrz = read_mrz(io.BytesIO(image_bytes))
        if mrz is None:
            return None
        d = mrz.to_dict()
        return {
            "format":          "passporteye",
            "document_type":   _clean(d.get("type", "")),
            "country":         _clean(d.get("country", "")),
            "surname":         _clean(d.get("surname", "")),
            "given_names":     _clean(d.get("names", "")),
            "document_number": _fix_numeric(_clean(d.get("number", ""))),
            "nationality":     _clean(d.get("nationality", "")),
            "date_of_birth":   _fix_numeric(d.get("date_of_birth", "")),
            "expiry_date":     _fix_numeric(d.get("expiry_date", "")),
            "sex":             _clean(d.get("sex", "")),
            "valid_score":     1 if d.get("valid_score", 0) >= 50 else 0,
            "valid":           d.get("valid_score", 0) >= 50,
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_mrz(image_bytes: bytes) -> dict:
    """
    Full pipeline: detect → OCR → parse → validate.
    Returns a normalised result dict.
    """
    parsed = None

    try:
        crop  = _detect_mrz_crop(image_bytes)
        lines = _ocr_mrz_strip(crop)
        parsed = _parse_td3(lines) or _parse_td1(lines)
    except Exception as e:
        print(f"[MRZ] Pipeline error: {e}")

    if parsed is None:
        print("[MRZ] Main pipeline found nothing — trying passporteye fallback")
        parsed = _passporteye_fallback(image_bytes)

    if parsed is None:
        return {"found": False, "error": "No MRZ detected in the document image"}

    dob    = _parse_date(parsed["date_of_birth"])
    expiry = _parse_date(parsed["expiry_date"])
    today  = date.today()

    expired    = bool(expiry and expiry < today)
    expiry_str = expiry.strftime("%Y-%m-%d") if expiry else ""
    dob_str    = dob.strftime("%Y-%m-%d")    if dob    else ""

    print(f"[MRZ] Result ({parsed['format']}): {parsed['surname']} {parsed['given_names']} | DOB {dob_str} | Exp {expiry_str} | valid={parsed['valid']}")

    return {
        "found":           True,
        "valid":           parsed["valid"],
        "valid_score":     parsed["valid_score"],
        "expired":         expired,
        "document_type":   parsed.get("document_type", ""),
        "country":         parsed.get("country", ""),
        "surname":         parsed.get("surname", ""),
        "given_names":     parsed.get("given_names", ""),
        "document_number": parsed.get("document_number", ""),
        "nationality":     parsed.get("nationality", ""),
        "date_of_birth":   dob_str,
        "expiry_date":     expiry_str,
        "sex":             parsed.get("sex", ""),
        "format":          parsed.get("format", ""),
    }
