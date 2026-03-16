from datetime import date
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.antispoof_service import liveness_detector
from app.services.document_auth_service import check_document_authenticity
from app.services.face_service import compare_faces, extract_face_encoding
from app.services.mrz_service import extract_mrz
from app.utils.image_utils import load_image_from_bytes, validate_image

MIN_AGE = 18

# Valid ISO 3166-1 alpha-3 country codes (ICAO 9303 travel document issuers)
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
    # ICAO special codes
    "UNO","UNA","UNK","XXX","EUE",
}

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

    # Validate document authenticity (not a photo, not fake/edited)
    auth = check_document_authenticity(doc_bytes)
    if not auth["is_authentic"]:
        raise HTTPException(status_code=400, detail=auth["reason"])

    # Extract and validate MRZ (expiry, document data)
    mrz = extract_mrz(doc_bytes)
    if mrz.get("found"):
        # Expiry check
        if mrz.get("expired"):
            raise HTTPException(
                status_code=400,
                detail="The document has expired. Please provide a valid, unexpired document.",
            )

        # Document type check — must be passport (P) or ID card (I/A/C)
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
                detail=f"Document issuing country '{country}' is not recognised. Please use a valid government-issued document.",
            )

        # Age check — applicant must be MIN_AGE or older
        dob_str = mrz.get("date_of_birth")
        if dob_str:
            try:
                dob = date.fromisoformat(dob_str)
                age = (date.today() - dob).days // 365
                if age < MIN_AGE:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Applicant must be at least {MIN_AGE} years old to verify.",
                    )
            except ValueError:
                pass  # unparseable DOB — do not block

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
    result["mrz"] = mrz

    return result
