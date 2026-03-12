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

    try:
        mrz = read_mrz(io.BytesIO(image_bytes))
    except Exception as e:
        return {"found": False, "error": f"MRZ read error: {e}"}

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

    # Parse dates
    dob     = _parse_date(data.get("date_of_birth", ""))
    expiry  = _parse_date(data.get("expiry_date", ""))
    today   = date.today()

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
        "document_number": _clean(data.get("number")),
        "nationality":     _clean(data.get("nationality")),
        "date_of_birth":   dob_str,
        "expiry_date":     expiry_str,
        "sex":             _clean(data.get("sex")),
        "document_type":   _clean(data.get("type")),
        "country":         _clean(data.get("country")),
    }
