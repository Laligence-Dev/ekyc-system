"""
MRZ (Machine Readable Zone) Service
=====================================
Uses EasyOCR (deep-learning) to read MRZ text, with passporteye as fallback.

Strategy:
  1. Preprocess image (upscale, grayscale, high contrast)
  2. Run EasyOCR on full image → filter for MRZ-like lines
  3. Parse TD3 (passport, 2×44) or TD1 (ID card, 3×30) format
  4. Fix common digit/letter OCR confusions in numeric fields
  5. Validate check digits and expiry date
  6. Fall back to passporteye if EasyOCR finds no MRZ lines
"""

import io
import re
from datetime import date, datetime

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ── Lazy EasyOCR reader (downloads ~100 MB model on first use) ────────────────

_easyocr_reader = None


def _get_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        print("Loading EasyOCR model (first use — may take a moment)...")
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("EasyOCR ready.")
    return _easyocr_reader


# ── Image preprocessing ───────────────────────────────────────────────────────

def _preprocess(image_bytes: bytes) -> np.ndarray:
    """Upscale + high-contrast grayscale for best OCR results."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")

    min_width = 2000
    if img.width < min_width:
        scale = min_width / img.width
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS,
        )

    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=2))
    img = ImageEnhance.Contrast(img).enhance(2.0)
    return np.array(img)


# ── MRZ character helpers ─────────────────────────────────────────────────────

_INVALID_CHARS = re.compile(r"[^A-Z0-9<]")
_DIGIT_FIXES   = str.maketrans("OIlBSZGT", "01185267")


def _normalize(text: str) -> str:
    """Uppercase, map space/dot/dash → <, strip invalid chars."""
    text = text.upper().replace(" ", "<").replace(".", "<").replace("-", "<")
    return _INVALID_CHARS.sub("", text)


def _fix_numeric(s: str) -> str:
    """Fix letter→digit OCR errors in numeric-only MRZ fields."""
    return s.translate(_DIGIT_FIXES)


def _clean(s: str) -> str:
    return s.replace("<", " ").strip()


# ── Check digit validation (ICAO 9303) ───────────────────────────────────────

_WEIGHTS = [7, 3, 1]
_CHARVAL = {c: i for i, c in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
_CHARVAL["<"] = 0


def _check_digit(s: str) -> str:
    total = sum(_CHARVAL.get(c, 0) * _WEIGHTS[i % 3] for i, c in enumerate(s))
    return str(total % 10)


# ── Date helpers ──────────────────────────────────────────────────────────────

def _parse_date(yymmdd: str) -> date | None:
    try:
        return datetime.strptime(yymmdd.strip("<"), "%y%m%d").date()
    except Exception:
        return None


# ── MRZ line detection ────────────────────────────────────────────────────────

def _find_mrz_lines(ocr_results) -> list[str]:
    """
    Filter EasyOCR results for MRZ-like lines.
    MRZ lines: ≥20 chars after normalisation, contain at least one '<'.
    Returns lines sorted top-to-bottom.
    """
    candidates = []
    for (bbox, text, _conf) in ocr_results:
        norm = _normalize(text)
        if len(norm) >= 20 and "<" in norm:
            y = (bbox[0][1] + bbox[2][1]) / 2
            candidates.append((y, norm))

    candidates.sort(key=lambda x: x[0])
    return [t for _, t in candidates]


def _pad(line: str, length: int) -> str:
    """Pad or truncate a line to exact length."""
    return (line + "<" * length)[:length]


# ── TD3 parser (passport, 2 lines × 44 chars) ────────────────────────────────

def _parse_td3(lines: list[str]) -> dict | None:
    # Find two consecutive lines that are each close to 44 chars
    for i in range(len(lines) - 1):
        l1 = _pad(lines[i],     44)
        l2 = _pad(lines[i + 1], 44)

        if l1[0] not in "PIV":
            continue

        # Line 2 numeric fields
        doc_num  = _fix_numeric(l2[0:9])
        nat      = l2[10:13]
        dob_raw  = _fix_numeric(l2[13:19])
        sex      = l2[20] if l2[20] in "MF<" else "<"
        exp_raw  = _fix_numeric(l2[21:27])

        # Check digits
        valid_doc = _check_digit(doc_num) == l2[9]
        valid_dob = _check_digit(dob_raw) == l2[19]
        valid_exp = _check_digit(exp_raw) == l2[27]
        valid_score = sum([valid_doc, valid_dob, valid_exp])

        # Parse names from line 1
        names_field = l1[5:44]
        if "<<" in names_field:
            surname_raw, given_raw = names_field.split("<<", 1)
        else:
            surname_raw, given_raw = names_field, ""

        return {
            "format":          "TD3",
            "document_type":   _clean(l1[0:2]),
            "country":         _clean(l1[2:5]),
            "surname":         _clean(surname_raw),
            "given_names":     _clean(given_raw),
            "document_number": _clean(doc_num),
            "nationality":     _clean(nat),
            "date_of_birth":   dob_raw,
            "expiry_date":     exp_raw,
            "sex":             sex,
            "valid_score":     valid_score,   # 0-3
            "valid":           valid_score >= 2,
        }
    return None


# ── TD1 parser (ID card, 3 lines × 30 chars) ─────────────────────────────────

def _parse_td1(lines: list[str]) -> dict | None:
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

        valid_doc = _check_digit(doc_num) == l1[14]
        valid_dob = _check_digit(dob_raw) == l2[6]
        valid_exp = _check_digit(exp_raw) == l2[14]
        valid_score = sum([valid_doc, valid_dob, valid_exp])

        names_field = l3
        if "<<" in names_field:
            surname_raw, given_raw = names_field.split("<<", 1)
        else:
            surname_raw, given_raw = names_field, ""

        return {
            "format":          "TD1",
            "document_type":   _clean(l1[0:2]),
            "country":         _clean(l1[2:5]),
            "surname":         _clean(surname_raw),
            "given_names":     _clean(given_raw),
            "document_number": _clean(doc_num),
            "nationality":     _clean(nat),
            "date_of_birth":   dob_raw,
            "expiry_date":     exp_raw,
            "sex":             sex,
            "valid_score":     valid_score,
            "valid":           valid_score >= 2,
        }
    return None


# ── Passporteye fallback ──────────────────────────────────────────────────────

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
            "valid_score":     d.get("valid_score", 0),
            "valid":           d.get("valid_score", 0) >= 50,
        }
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def extract_mrz(image_bytes: bytes) -> dict:
    """
    Extract and validate MRZ data from a document image.
    Returns a normalised dict — see keys below.
    """
    parsed = None

    # ── 1. Try EasyOCR ──
    try:
        reader  = _get_reader()
        arr     = _preprocess(image_bytes)
        results = reader.readtext(arr, detail=1, paragraph=False)
        lines   = _find_mrz_lines(results)
        parsed  = _parse_td3(lines) or _parse_td1(lines)
    except Exception as e:
        print(f"[MRZ] EasyOCR error: {e}")

    # ── 2. Fall back to passporteye ──
    if parsed is None:
        parsed = _passporteye_fallback(image_bytes)

    if parsed is None:
        return {"found": False, "error": "No MRZ detected in the document image"}

    # ── 3. Parse and validate dates ──
    dob    = _parse_date(parsed["date_of_birth"])
    expiry = _parse_date(parsed["expiry_date"])
    today  = date.today()

    expired    = bool(expiry and expiry < today)
    expiry_str = expiry.strftime("%Y-%m-%d") if expiry else ""
    dob_str    = dob.strftime("%Y-%m-%d")    if dob    else ""

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
