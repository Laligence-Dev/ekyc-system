from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.antispoof_service import liveness_detector
from app.services.face_service import compare_faces, extract_face_encoding
from app.utils.image_utils import load_image_from_bytes, validate_image

router = APIRouter()


@router.post("/verify")
async def verify_faces(
    document: UploadFile = File(..., description="Government ID or passport image"),
    selfie: UploadFile = File(..., description="Live selfie photo"),
):
    """Compare a face on a government ID/passport with a selfie photo."""

    # Read files
    doc_bytes = await document.read()
    selfie_bytes = await selfie.read()

    # Validate document image
    error = validate_image(document.content_type, len(doc_bytes))
    if error:
        raise HTTPException(status_code=422, detail=f"Document image: {error}")

    # Validate selfie image
    error = validate_image(selfie.content_type, len(selfie_bytes))
    if error:
        raise HTTPException(status_code=422, detail=f"Selfie image: {error}")

    # Load images as numpy arrays
    doc_image = load_image_from_bytes(doc_bytes)
    selfie_image = load_image_from_bytes(selfie_bytes)

    # Extract face from document
    doc_encoding, doc_face_count, _ = extract_face_encoding(doc_image)
    if doc_encoding is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the document image. Please upload a clearer photo.",
        )

    # Extract face from selfie
    selfie_encoding, selfie_face_count, selfie_face_location = extract_face_encoding(selfie_image)
    if selfie_encoding is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the selfie. Please take a clearer photo.",
        )
    if selfie_face_count > 1:
        raise HTTPException(
            status_code=400,
            detail="Multiple faces detected in the selfie. Please ensure only one face is visible.",
        )

    # Liveness check on selfie
    liveness = liveness_detector.check_liveness(selfie_image, selfie_face_location)
    if not liveness["is_live"]:
        raise HTTPException(
            status_code=400,
            detail="Liveness check failed. Please use a real face, not a photo or screen.",
        )

    # Compare faces
    result = compare_faces(doc_encoding, selfie_encoding)
    result["document_faces_detected"] = doc_face_count
    result["selfie_faces_detected"] = selfie_face_count
    result["is_live"] = True
    result["liveness_score"] = liveness["liveness_score"]

    return result
