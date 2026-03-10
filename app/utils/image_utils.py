import io

import numpy as np
from PIL import Image

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def validate_image(content_type: str | None, size: int) -> str | None:
    """Validate image file type and size. Returns error message or None if valid."""
    if content_type not in ALLOWED_TYPES:
        return f"Invalid file type '{content_type}'. Allowed: JPEG, PNG."
    if size > MAX_FILE_SIZE:
        return f"File too large ({size} bytes). Maximum: {MAX_FILE_SIZE} bytes."
    return None


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Convert raw image bytes to a numpy RGB array for face_recognition."""
    image = Image.open(io.BytesIO(data))
    image = image.convert("RGB")
    return np.array(image)
