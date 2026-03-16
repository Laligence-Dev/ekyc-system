from datetime import date
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.antispoof_service import liveness_detector
from app.services.screen_replay_service import screen_replay_detector
from app.services.document_auth_service import check_document_authenticity
from app.services.face_service import compare_faces, extract_face_encoding
from app.services.mrz_service import extract_mrz
from app.services.session_store import session_store
from app.utils.image_utils import load_image_from_bytes, validate_image

MIN_AGE = 18

_VALID_COUNTRIES = {
    "AFG","ALB","DZA","AND","AGO","ATG","ARG","ARM","AUS","AUT","AZE","BHS","BHR",
    "BGD","BRB","BLR","BEL","BLZ","BEN","BTN","BOL","BIH","BWA","BRA","BRN","BGR",
    "BFA","BDI","CPV","KHM","CMR","CAN","CAF","TCD","CHL","CHN","COL","COM","COD",
    "COG","CRI","CIV","HRV","CUB","CYP","CZE","DNK","DJI","DOM","ECU","EGY","SLV",
    "GNQ","ERI","EST","SWZ","ETH","FJI","FIN","FRA","GAB","GMB","GEO","DEU","GHA",
    "GRC","GRD","GTM","GIN","GNB","GUY","HTI","HND","HUN","ISL","IND","IDN","IRN",
    "IRQ","IRL","ISR","ITA","JAM","JPN","JOR","KAZ","KEN","KIR","PRK","KOR","KWT",
    "KGZ","LAO","LVA","LBN","LSO","LBR","LBY","LIE","LTU","LUX","MDG","MWI","MYS",
    "MDV","MLI","MLT","MHL","MRT","MUS","MEX","FSM","MDA","MCO","MNG","MNE","MAR",
    "MOZ","MMR","NAM","NRU","NPL","NLD","NZL","NIC","NER","NGA","MKD","NOR","OMN",
    "PAK","PLW","PAN","PNG","PRY","PER","PHL","POL","PRT","QAT","ROU","RUS","RWA",
    "KNA","LCA","VCT","WSM","SMR","STP","SAU","SEN","SRB","SYC","SLE","SGP","SVK",
    "SVN","SLB","SOM","ZAF","SSD","ESP","LKA","SDN","SUR","SWE","CHE","SYR","TWN",
    "TJK","TZA","THA","TLS","TGO","TON","TTO","TUN","TUR","TKM","TUV","UGA","UKR",
    "ARE","GBR","USA","URY","UZB","VUT","VEN","VNM","YEM","ZMB","ZWE",
    "UNO","UNA","UNK","XXX","EUE",
}

router = APIRouter()


@router.post("/document/upload")
async def upload_document(
    document: UploadFile = File(..., description="Government ID or passport image"),
):
    """Upload a government ID / passport. Extracts the face and creates a session."""
    doc_bytes = await document.read()

    error = validate_image(document.content_type, len(doc_bytes))
    if error:
        raise HTTPException(status_code=422, detail=f"Document image: {error}")

    auth = check_document_authenticity(doc_bytes)
    if not auth["is_authentic"]:
        raise HTTPException(status_code=400, detail=auth["reason"])

    doc_image = load_image_from_bytes(doc_bytes)
    doc_encoding, doc_face_count, _ = extract_face_encoding(doc_image)

    if doc_encoding is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the document image. Please upload a clearer photo.",
        )

    # MRZ extraction and IAL2 validation
    mrz = extract_mrz(doc_bytes)

    if mrz["found"]:
        # Expired document
        if mrz["expired"]:
            raise HTTPException(
                status_code=400,
                detail=f"Document has expired (expiry date: {mrz['expiry_date']}). Please use a valid document.",
            )

        # Unreadable expiry — cannot confirm document is valid
        if not mrz.get("expiry_date"):
            raise HTTPException(
                status_code=400,
                detail="Could not read the document expiry date. Please upload a clearer photo of your document.",
            )

        # Document type check
        doc_type = mrz.get("document_type", "")
        if doc_type and doc_type[0] not in ("P", "I", "A", "C", "V"):
            raise HTTPException(
                status_code=400,
                detail="The document type is not a recognised government-issued identity document.",
            )

        # Country code check
        country = mrz.get("country", "").strip()
        if country and country not in _VALID_COUNTRIES:
            raise HTTPException(
                status_code=400,
                detail=f"Document issuing country '{country}' is not recognised.",
            )

        # Age check
        dob_str = mrz.get("date_of_birth")
        if dob_str:
            try:
                dob = date.fromisoformat(dob_str)
                if (date.today() - dob).days // 365 < MIN_AGE:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Applicant must be at least {MIN_AGE} years old to verify.",
                    )
            except ValueError:
                pass

    session_id = session_store.create_session(doc_encoding, doc_face_count)

    return {
        "session_id": session_id,
        "document_faces_detected": doc_face_count,
        "mrz": {
            "found":           mrz["found"],
            "surname":         mrz.get("surname", ""),
            "given_names":     mrz.get("given_names", ""),
            "document_number": mrz.get("document_number", ""),
            "nationality":     mrz.get("nationality", ""),
            "date_of_birth":   mrz.get("date_of_birth", ""),
            "expiry_date":     mrz.get("expiry_date", ""),
            "sex":             mrz.get("sex", ""),
            "document_type":   mrz.get("document_type", ""),
            "country":         mrz.get("country", ""),
            "valid":           mrz.get("valid", False),
        },
        "message": "Document processed successfully. Ready for verification.",
    }


@router.post("/verify/frame")
async def verify_frame(
    frame: UploadFile = File(..., description="Webcam frame image"),
    session_id: str = Form(..., description="Session ID from document upload"),
):
    """Compare a webcam frame against the stored document face encoding."""
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Please re-upload the document.",
        )

    frame_bytes = await frame.read()

    error = validate_image(frame.content_type, len(frame_bytes))
    if error:
        raise HTTPException(status_code=422, detail=f"Frame image: {error}")

    frame_image = load_image_from_bytes(frame_bytes)
    frame_encoding, frame_face_count, face_location = extract_face_encoding(frame_image)

    # No face in frame — transient state, not an error
    if frame_encoding is None:
        return {
            "match": False,
            "face_detected": False,
            "message": "No face detected in frame.",
        }

    # Multiple faces — inform but don't error
    if frame_face_count > 1:
        return {
            "match": False,
            "face_detected": True,
            "message": "Multiple faces detected. Please ensure only one face is visible.",
        }

    # Stage 1: Screen replay heuristic (fast, no ONNX)
    screen = screen_replay_detector.check(frame_image, face_location, session_id=session_id)
    if screen["is_screen_attack"]:
        return {
            "match": False,
            "face_detected": True,
            "is_live": False,
            "screen_attack": True,
            "screen_score": screen["screen_score"],
            "message": "Screen replay attack detected. Please show your real face to the camera.",
        }

    # Stage 2: MiniFASNet passive liveness
    liveness = liveness_detector.check_liveness(frame_image, face_location)

    # Soft flag: suspicious screen + failed liveness → reject
    if screen["suspicious"] and not liveness["is_live"]:
        return {
            "match": False,
            "face_detected": True,
            "is_live": False,
            "screen_attack": True,
            "screen_score": screen["screen_score"],
            "liveness_score": liveness["liveness_score"],
            "message": "Liveness check failed. Please use a real face, not a photo or screen.",
        }

    if not liveness["is_live"]:
        return {
            "match": False,
            "face_detected": True,
            "is_live": False,
            "screen_attack": False,
            "liveness_score": liveness["liveness_score"],
            "message": "Liveness check failed. Please use a real face, not a photo or screen.",
        }

    result = compare_faces(session["encoding"], frame_encoding)
    result["face_detected"] = True
    result["is_live"] = True
    result["liveness_score"] = liveness["liveness_score"]
    result["selfie_faces_detected"] = frame_face_count

    return result
