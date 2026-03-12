from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.antispoof_service import liveness_detector
from app.services.document_auth_service import check_document_authenticity
from app.services.face_service import compare_faces, extract_face_encoding
from app.services.mrz_service import extract_mrz
from app.services.session_store import session_store
from app.utils.image_utils import load_image_from_bytes, validate_image

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

    # MRZ extraction (non-blocking — warn but don't reject if not found)
    mrz = extract_mrz(doc_bytes)

    # Hard block: document is expired
    if mrz["found"] and mrz["expired"]:
        raise HTTPException(
            status_code=400,
            detail=f"Document has expired (expiry date: {mrz['expiry_date']}). Please use a valid document.",
        )

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

    # Liveness check — reject photos/screens
    liveness = liveness_detector.check_liveness(frame_image, face_location)
    if not liveness["is_live"]:
        return {
            "match": False,
            "face_detected": True,
            "is_live": False,
            "liveness_score": liveness["liveness_score"],
            "message": "Liveness check failed. Please use a real face, not a photo or screen.",
        }

    result = compare_faces(session["encoding"], frame_encoding)
    result["face_detected"] = True
    result["is_live"] = True
    result["liveness_score"] = liveness["liveness_score"]
    result["selfie_faces_detected"] = frame_face_count

    return result
