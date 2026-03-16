import uuid
import time
import threading
from typing import Optional

import numpy as np

SESSION_TTL = 600  # 10 minutes
CLEANUP_INTERVAL = 60  # seconds


class SessionStore:
    """Thread-safe in-memory store for document face encodings, keyed by session UUID."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._start_cleanup_thread()

    def create_session(self, encoding: np.ndarray, face_count: int) -> str:
        session_id = str(uuid.uuid4())
        with self._lock:
            self._sessions[session_id] = {
                "encoding": encoding,
                "face_count": face_count,
                "created_at": time.time(),
                "challenge": None,
                "challenge_passed": False,
                "challenge_frames_ok": 0,
            }
        return session_id

    def assign_challenge(self, session_id: str, challenge: str) -> bool:
        with self._lock:
            if session_id not in self._sessions:
                return False
            self._sessions[session_id]["challenge"] = challenge
            self._sessions[session_id]["challenge_passed"] = False
            self._sessions[session_id]["challenge_frames_ok"] = 0
            return True

    def increment_challenge_frames(self, session_id: str) -> int:
        with self._lock:
            if session_id not in self._sessions:
                return 0
            self._sessions[session_id]["challenge_frames_ok"] += 1
            return self._sessions[session_id]["challenge_frames_ok"]

    def reset_challenge_frames(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["challenge_frames_ok"] = 0

    def mark_challenge_passed(self, session_id: str) -> bool:
        with self._lock:
            if session_id not in self._sessions:
                return False
            self._sessions[session_id]["challenge_passed"] = True
            return True

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if (time.time() - session["created_at"]) >= SESSION_TTL:
                del self._sessions[session_id]
                return None
            return session

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
        self._cleanup_screen_buffer(session_id)

    def _cleanup_expired(self) -> None:
        now = time.time()
        with self._lock:
            expired = [
                sid
                for sid, data in self._sessions.items()
                if (now - data["created_at"]) >= SESSION_TTL
            ]
            for sid in expired:
                del self._sessions[sid]
        for sid in expired:
            self._cleanup_screen_buffer(sid)

    @staticmethod
    def _cleanup_screen_buffer(session_id: str) -> None:
        try:
            from app.services.screen_replay_service import screen_replay_detector
            screen_replay_detector.cleanup_session(session_id)
        except Exception:
            pass

    def _start_cleanup_thread(self) -> None:
        def run() -> None:
            while True:
                time.sleep(CLEANUP_INTERVAL)
                self._cleanup_expired()

        t = threading.Thread(target=run, daemon=True)
        t.start()


# Singleton — shared across all route handlers
session_store = SessionStore()
