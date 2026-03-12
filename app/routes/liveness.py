import random

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.active_liveness_service import (
    CHALLENGES,
    REQUIRED_FRAMES,
    active_liveness_detector,
)
from app.services.session_store import session_store

router = APIRouter()


@router.get("/liveness/challenge")
async def get_liveness_challenge(session_id: str):
    """Assign a random liveness challenge to the session and return it."""
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    if session.get("challenge_passed"):
        return {
            "challenge": session["challenge"],
            "instruction": CHALLENGES[session["challenge"]],
            "passed": True,
        }

    if not session.get("challenge"):
        challenge = random.choice(list(CHALLENGES.keys()))
        session_store.assign_challenge(session_id, challenge)
        session = session_store.get_session(session_id)

    return {
        "challenge": session["challenge"],
        "instruction": CHALLENGES[session["challenge"]],
        "passed": False,
    }


@router.post("/liveness/check")
async def check_liveness_challenge(
    frame: UploadFile = File(...),
    session_id: str = Form(...),
):
    """Submit a webcam frame and check if the challenge action was performed."""
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    if session.get("challenge_passed"):
        return {"passed": True, "face_detected": True, "frames_ok": REQUIRED_FRAMES, "message": "Already passed."}

    challenge = session.get("challenge")
    if not challenge:
        raise HTTPException(status_code=400, detail="No challenge assigned. Call /api/liveness/challenge first.")

    frame_bytes = await frame.read()
    result = active_liveness_detector.detect(frame_bytes, challenge)

    if not result["face_detected"]:
        session_store.reset_challenge_frames(session_id)
        return {"passed": False, "face_detected": False, "frames_ok": 0, "message": result["message"]}

    if result["action_detected"]:
        frames_ok = session_store.increment_challenge_frames(session_id)
        if frames_ok >= REQUIRED_FRAMES:
            session_store.mark_challenge_passed(session_id)
            return {"passed": True, "face_detected": True, "frames_ok": frames_ok, "message": "Liveness challenge passed!"}
        return {"passed": False, "face_detected": True, "frames_ok": frames_ok, "message": f"Hold it... ({frames_ok}/{REQUIRED_FRAMES})"}

    session_store.reset_challenge_frames(session_id)
    return {"passed": False, "face_detected": True, "frames_ok": 0, "message": result["message"]}
