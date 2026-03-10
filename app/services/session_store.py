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
            }
        return session_id

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

    def _start_cleanup_thread(self) -> None:
        def run() -> None:
            while True:
                time.sleep(CLEANUP_INTERVAL)
                self._cleanup_expired()

        t = threading.Thread(target=run, daemon=True)
        t.start()


# Singleton — shared across all route handlers
session_store = SessionStore()
