import numpy as np
import face_recognition

MATCH_THRESHOLD = 0.6


def extract_face_encoding(image: np.ndarray) -> tuple[np.ndarray | None, int, tuple | None]:
    """Detect faces in an image and return the encoding of the first face found.

    Returns:
        (encoding, face_count, face_location) — encoding/face_location are None if no faces detected.
        face_location is (top, right, bottom, left).
    """
    face_locations = face_recognition.face_locations(image, model="hog")
    face_count = len(face_locations)

    if face_count == 0:
        return None, 0, None

    encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
    return encodings[0], face_count, face_locations[0]


def compare_faces(
    doc_encoding: np.ndarray,
    selfie_encoding: np.ndarray,
    threshold: float = MATCH_THRESHOLD,
) -> dict:
    """Compare two face encodings and return match result with confidence.

    Returns:
        dict with match, confidence, distance, and threshold.
    """
    distance = face_recognition.face_distance([doc_encoding], selfie_encoding)[0]
    distance = float(distance)
    match = distance <= threshold
    confidence = round(max(0.0, 1.0 - distance), 4)

    return {
        "match": match,
        "confidence": confidence,
        "distance": round(distance, 4),
        "threshold": threshold,
    }
