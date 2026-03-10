import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes.session import router as session_router
from app.routes.verify import router as verify_router

app = FastAPI(
    title="Face Recognition Verification API",
    description="Verify identity by comparing a government ID photo with a live selfie.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(verify_router, prefix="/api")
app.include_router(session_router, prefix="/api")


@app.on_event("startup")
async def preload_models():
    from app.services.antispoof_service import liveness_detector
    liveness_detector.initialize()


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


# Serve frontend static files — mounted AFTER API routes so /api/* takes priority
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
