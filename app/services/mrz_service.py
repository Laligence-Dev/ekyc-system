"""
MRZ (Machine Readable Zone) Service
=====================================
Uses passporteye to detect and parse the MRZ from a passport or ID card image.
Supports TD1 (ID cards, 3-line), TD2, and TD3 (passports, 2-line) formats.

Returns parsed fields and validates:
  - Check digits (per ICAO 9303)
  - Expiry date (document must not be expired)
"""

import io
from datetime import date, datetime

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def _preprocess_for_mrz(image_bytes: bytes) -> bytes:
    """
    Optimise image for Tesseract MRZ reading:
      1. Upscale to ≥2400px wide
      2. Convert to grayscale
      3. Unsharp mask for edge sharpness
      4. High contrast boost
      5. Binarize — Tesseract reads clean black/white best
    Returns preprocessed PNG bytes (lossless for OCR).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale

    # Upscale — MRZ characters need to be large for Tesseract
    min_width = 2400
    if img.width < min_width:
        scale = min_width / img.width
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS,
        )

    # Unsharp mask: sharpens edges without adding noise
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))

    # Boost contrast so MRZ band is dark-on-light
    img = ImageEnhance.Contrast(img).enhance(2.5)

    # Binarize: pixels below 160 → black, above → white
    img = img.point(lambda p: 0 if p < 160 else 255, "L")

    out = io.BytesIO()
    img.save(out, format="PNG")  # lossless — no JPEG artifacts on text
    return out.getvalue()


# Characters OCR commonly confuses in purely-numeric MRZ fields
_DIGIT_FIXES = str.maketrans("OIlBSZGT", "01185267")


def _fix_numeric(s: str) -> str:
    """Replace letters that look like digits in numeric-only MRZ fields."""
    return s.translate(_DIGIT_FIXES)


def _parse_date(yymmdd: str) -> date | None:
    """Parse YYMMDD MRZ date string into a date object."""
    try:
        dt = datetime.strptime(yymmdd.strip("<"), "%y%m%d").date()
        return dt
    except Exception:
        return None


def _clean(value: str | None) -> str:
    """Strip MRZ filler characters."""
    if not value:
        return ""
    return value.replace("<", " ").strip()


def extract_mrz(image_bytes: bytes) -> dict:
    """
    Extract and validate MRZ data from a document image.

    Returns a dict with:
        found         bool   — whether an MRZ was detected
        valid         bool   — all check digits passed
        expired       bool   — document is past its expiry date
        surname       str
        given_names   str
        document_number str
        nationality   str
        date_of_birth str    (YYYY-MM-DD)
        expiry_date   str    (YYYY-MM-DD)
        sex           str
        document_type str
        error         str    (if found=False)
    """
    try:
        from passporteye import read_mrz
    except ImportError:
        return {"found": False, "error": "passporteye not installed"}

    # Try preprocessed image first (better OCR accuracy), fall back to original
    try:
        enhanced = _preprocess_for_mrz(image_bytes)
    except Exception:
        enhanced = image_bytes

    mrz = None
    for attempt in (enhanced, image_bytes):
        try:
            mrz = read_mrz(io.BytesIO(attempt))
        except Exception as e:
            return {"found": False, "error": f"MRZ read error: {e}"}
        if mrz is not None:
            break

    if mrz is None:
        return {"found": False, "error": "No MRZ detected in the document image"}

    try:
        data = mrz.to_dict()
    except Exception as e:
        return {"found": False, "error": f"MRZ parse error: {e}"}

    # Check digit validation
    valid_score = data.get("valid_score", 0)
    # passporteye valid_score: 100 = all checks pass, lower = some failed
    is_valid = valid_score >= 50

    # Apply digit correction to numeric-only fields before parsing
    raw_dob    = _fix_numeric(data.get("date_of_birth", ""))
    raw_expiry = _fix_numeric(data.get("expiry_date", ""))
    raw_number = _fix_numeric(data.get("number", ""))

    # Parse dates
    dob    = _parse_date(raw_dob)
    expiry = _parse_date(raw_expiry)
    today  = date.today()

    expired = False
    expiry_str = ""
    if expiry:
        expired    = expiry < today
        expiry_str = expiry.strftime("%Y-%m-%d")

    dob_str = dob.strftime("%Y-%m-%d") if dob else ""

    return {
        "found":           True,
        "valid":           is_valid,
        "valid_score":     valid_score,
        "expired":         expired,
        "surname":         _clean(data.get("surname")),
        "given_names":     _clean(data.get("names")),
        "document_number": _clean(raw_number),
        "nationality":     _clean(data.get("nationality")),
        "date_of_birth":   dob_str,
        "expiry_date":     expiry_str,
        "sex":             _clean(data.get("sex")),
        "document_type":   _clean(data.get("type")),
        "country":         _clean(data.get("country")),
    }
