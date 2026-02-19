"""
Lightweight backend server — serves all ML model endpoints.

No Postgres, Redis, or blockchain dependencies required.
Use this for model verification and frontend integration testing.

Run:
    cd backend && python3 server_lite.py
"""
import os

# ── Force JAX to CPU-only mode ────────────────────────────────
# JAX + CUDA bindings can deadlock during GPU initialization on some systems.
# MediaPipe depends on JAX, so this must be set BEFORE any imports.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # prevent CUDA init in JAX/mediapipe

import logging
import sys
import time
import uuid
import hashlib
import base64
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path setup ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

# Weight paths — resolved directly (no config.py import to avoid TF cascade)
XCEPTION_WEIGHTS = PROJECT_ROOT / "ml" / "deepfake" / "weights" / "xception_celeb_df.pth"
AUDIO_WEIGHTS = PROJECT_ROOT / "ml" / "audio" / "weights" / "audio_cnn_lstm.pth"
EMOTION_WEIGHTS = PROJECT_ROOT / "ml" / "emotion" / "weights" / "emotion_model.pth"

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("server_lite")

# ── ML Model Singletons ──────────────────────────────────────
deepfake_detector = None
audio_detector = None
liveness_engine = None
emotion_scorer = None

# ── In-Memory Session Store (no DB needed) ───────────────────
# { session_id: { wallet, nonce, challenge_type, challenge_prompt, enrolled_face, scores, ... } }
sessions: dict[str, dict] = {}
enrolled_faces: dict[str, np.ndarray] = {}  # wallet_address → face image

CHALLENGES = [
    {"type": "SMILE_NATURALLY", "prompt": "Please smile naturally at the camera 😊"},
    {"type": "BLINK_TWICE", "prompt": "Please blink twice slowly 👁️"},
    {"type": "LEFT", "prompt": "Please turn your head to the left ⬅️"},
    {"type": "RIGHT", "prompt": "Please turn your head to the right ➡️"},
    {"type": "UP", "prompt": "Please look up ⬆️"},
]

PASS_THRESHOLD = 0.45  # minimum final_score to pass verification


def load_models():
    """Load all ML models for verification pipeline."""
    global deepfake_detector, audio_detector, liveness_engine, emotion_scorer

    # Import directly from module files — bypass models/__init__.py
    # to avoid triggering face_match → deepface → retinaface cascade
    import importlib.util

    def _import_from_file(module_name: str, file_path: str):
        """Import a module directly from its file path, skipping __init__.py."""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    models_dir = BACKEND_DIR / "models"

    # 1. Deepfake detector (XceptionNet)
    try:
        wp = str(XCEPTION_WEIGHTS)
        logger.info(f"Loading deepfake detector from {wp}")
        vid_mod = _import_from_file("video_detector", str(models_dir / "video_detector.py"))
        deepfake_detector = vid_mod.VideoDeepfakeDetector(weights_path=wp)
        logger.info(
            f"  ✅ Deepfake detector: loaded={deepfake_detector.is_loaded}, "
            f"params={sum(p.numel() for p in deepfake_detector.model.parameters()):,}"
        )
    except Exception as e:
        logger.error(f"  ❌ Failed to load deepfake detector: {e}")

    # 2. Audio detector (CNN-LSTM)
    try:
        wp = str(AUDIO_WEIGHTS)
        logger.info(f"Loading audio detector from {wp}")
        aud_mod = _import_from_file("audio_detector", str(models_dir / "audio_detector.py"))
        audio_detector = aud_mod.AudioDeepfakeDetector(weights_path=wp)
        logger.info(
            f"  ✅ Audio detector: loaded={audio_detector.is_loaded}, "
            f"params={sum(p.numel() for p in audio_detector.model.parameters()):,}"
        )
    except Exception as e:
        logger.error(f"  ❌ Failed to load audio detector: {e}")

    # 3. Liveness engine (MediaPipe-based — no weights needed)
    try:
        logger.info("Loading liveness engine (MediaPipe)...")
        live_mod = _import_from_file("face_liveness", str(models_dir / "face_liveness.py"))
        liveness_engine = live_mod.LivenessEngine()
        logger.info("  ✅ Liveness engine loaded (3-layer: active + optical flow + rPPG)")
    except Exception as e:
        logger.error(f"  ❌ Failed to load liveness engine: {e}")

    # 4. Emotion scorer (EmotionCNN + Duchenne via MediaPipe)
    try:
        wp = str(EMOTION_WEIGHTS) if EMOTION_WEIGHTS.exists() else None
        logger.info(f"Loading emotion scorer (weights={wp or 'random-init'})...")
        emo_mod = _import_from_file("emotion_detector", str(models_dir / "emotion_detector.py"))
        emotion_scorer = emo_mod.EmotionAuthenticityScorer(weights_path=wp)
        logger.info("  ✅ Emotion scorer loaded (CNN + Duchenne AU analysis)")
    except Exception as e:
        logger.error(f"  ❌ Failed to load emotion scorer: {e}")


# ── FastAPI App ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    load_models()
    logger.info("=" * 50)
    logger.info("  Model server ready — all 4 engines.")
    logger.info("  Docs: http://localhost:8000/docs")
    logger.info("=" * 50)
    yield
    logger.info("Shutting down server...")
    if liveness_engine is not None:
        try:
            liveness_engine.close()
        except Exception:
            pass
    if emotion_scorer is not None:
        try:
            emotion_scorer.close()
        except Exception:
            pass
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Proof-of-Life Model Server (Lite)",
    version="1.0.0-lite",
    description="Lightweight ML model server for deepfake & audio detection — no DB/Redis required.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permissive for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
#  Health & Status
# ══════════════════════════════════════════════════════════════

@app.get("/api/v1/health")
async def health():
    return {
        "status": "healthy",
        "version": "1.0.0-lite",
        "mode": "model-only",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/models/status")
async def model_status():
    """Return load status of all ML models — matches full backend schema."""
    return {
        "deepfake_detector": {
            "loaded": deepfake_detector is not None and deepfake_detector.is_loaded,
            "weights": str(XCEPTION_WEIGHTS),
        },
        "audio_detector": {
            "loaded": audio_detector is not None and audio_detector.is_loaded,
            "weights": str(AUDIO_WEIGHTS),
        },
        "liveness_engine": {
            "loaded": liveness_engine is not None,
        },
        "emotion_scorer": {
            "loaded": emotion_scorer is not None,
        },
    }


# ══════════════════════════════════════════════════════════════
#  Deepfake Detection Test
# ══════════════════════════════════════════════════════════════

@app.post("/api/v1/test/deepfake")
async def test_deepfake(image: UploadFile = File(...)):
    """Test endpoint — run deepfake detection on a single uploaded image."""
    if deepfake_detector is None:
        raise HTTPException(status_code=503, detail="Deepfake detector not loaded")

    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    t0 = time.time()
    result = deepfake_detector.predict_frame(frame)
    elapsed = time.time() - t0

    return {
        "model_loaded": deepfake_detector.is_loaded,
        "inference_time_ms": round(elapsed * 1000, 1),
        **result,
    }


# ══════════════════════════════════════════════════════════════
#  Audio Detection Test
# ══════════════════════════════════════════════════════════════

@app.post("/api/v1/test/audio")
async def test_audio(audio: UploadFile = File(...)):
    """Test endpoint — run audio deepfake detection on an uploaded audio file."""
    if audio_detector is None:
        raise HTTPException(status_code=503, detail="Audio detector not loaded")

    audio_bytes = await audio.read()
    if len(audio_bytes) < 500:
        raise HTTPException(status_code=400, detail="Audio file too short")

    t0 = time.time()
    result = audio_detector.predict_from_bytes(audio_bytes)
    elapsed = time.time() - t0

    return {
        "model_loaded": audio_detector.is_loaded,
        "inference_time_ms": round(elapsed * 1000, 1),
        **result,
    }


# ══════════════════════════════════════════════════════════════
#  Liveness Detection Test
# ══════════════════════════════════════════════════════════════

@app.post("/api/v1/test/liveness")
async def test_liveness(
    image: UploadFile = File(...),
    challenge_type: str = "BLINK_TWICE",
):
    """Test endpoint — run liveness detection on a single frame."""
    if liveness_engine is None:
        raise HTTPException(status_code=503, detail="Liveness engine not loaded")

    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    t0 = time.time()
    result = liveness_engine.process_frame(frame, challenge_type)
    elapsed = time.time() - t0

    return {
        "challenge_type": challenge_type,
        "inference_time_ms": round(elapsed * 1000, 1),
        **result,
    }


# ══════════════════════════════════════════════════════════════
#  Emotion Scoring Test
# ══════════════════════════════════════════════════════════════

@app.post("/api/v1/test/emotion")
async def test_emotion(
    image: UploadFile = File(...),
    challenge_type: str = "SMILE_NATURALLY",
):
    """Test endpoint — run emotion scoring on a single image."""
    if emotion_scorer is None:
        raise HTTPException(status_code=503, detail="Emotion scorer not loaded")

    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    t0 = time.time()
    result = emotion_scorer.score_authenticity(frame, challenge_type)
    elapsed = time.time() - t0

    return {
        "challenge_type": challenge_type,
        "inference_time_ms": round(elapsed * 1000, 1),
        **result,
    }


# ══════════════════════════════════════════════════════════════
#  Quick Inference (JSON body with base64 image)
# ══════════════════════════════════════════════════════════════

# ── Helper: decode uploaded image ─────────────────────────────
def _decode_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return frame


# ══════════════════════════════════════════════════════════════
#  ENROLLMENT — Capture reference face
# ══════════════════════════════════════════════════════════════

@app.post("/api/v1/enroll")
async def enroll_face(
    wallet_address: str = Form(...),
    image: UploadFile = File(...),
):
    """Enroll a user's face for later comparison during verification."""
    image_bytes = await image.read()
    frame = _decode_image(image_bytes)

    # Store the enrolled face
    enrolled_faces[wallet_address.lower()] = frame
    logger.info(f"Enrolled face for wallet {wallet_address[:10]}… ({frame.shape})")

    return {
        "success": True,
        "wallet_address": wallet_address,
        "message": "Face enrolled successfully",
    }


# ══════════════════════════════════════════════════════════════
#  CHALLENGE — Issue a randomized liveness challenge
# ══════════════════════════════════════════════════════════════

class ChallengeRequest(BaseModel):
    wallet_address: str

@app.post("/api/v1/challenge")
async def issue_challenge(req: ChallengeRequest):
    """Issue a random liveness challenge for the user to perform."""
    import random

    wallet = req.wallet_address.lower()
    challenge = random.choice(CHALLENGES)
    session_id = str(uuid.uuid4())
    nonce = secrets.token_hex(16)

    # Reset liveness engine state for new session
    if liveness_engine is not None:
        try:
            liveness_engine.reset()
        except Exception:
            pass

    sessions[session_id] = {
        "wallet_address": wallet,
        "nonce": nonce,
        "challenge_type": challenge["type"],
        "challenge_prompt": challenge["prompt"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scores": {
            "deepfake_frames": [],
            "liveness_frames": [],
            "emotion_frames": [],
            "audio_fake_prob": None,
        },
        "completed": False,
    }

    logger.info(f"Challenge issued: session={session_id[:8]}… type={challenge['type']}")

    return {
        "session_id": session_id,
        "nonce": nonce,
        "challenge_type": challenge["type"],
        "challenge_prompt": challenge["prompt"],
    }


# ══════════════════════════════════════════════════════════════
#  VERIFY/STREAM — Process captured video frames
# ══════════════════════════════════════════════════════════════

@app.post("/api/v1/verify/stream")
async def verify_stream(
    session_id: str = Form(...),
    nonce: str = Form(...),
    wallet_address: str = Form(...),
    frames: list[UploadFile] = File(...),
):
    """
    Process a batch of captured video frames:
    - Deepfake detection on each frame
    - Liveness challenge verification on each frame
    - Emotion scoring on each frame
    Returns per-frame scores and aggregated results.
    """
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["nonce"] != nonce:
        raise HTTPException(status_code=403, detail="Invalid nonce")

    challenge_type = session["challenge_type"]
    deepfake_scores = []
    liveness_scores = []
    emotion_scores = []
    frame_results = []

    t0 = time.time()

    for i, frame_file in enumerate(frames):
        raw = await frame_file.read()
        try:
            frame = _decode_image(raw)
        except Exception:
            continue

        result_i = {"frame": i}

        # Deepfake detection
        if deepfake_detector is not None:
            try:
                df_result = deepfake_detector.predict_frame(frame)
                fake_prob = df_result.get("fake_probability", 0.5)
                deepfake_scores.append(fake_prob)
                result_i["deepfake"] = fake_prob
            except Exception as e:
                logger.warning(f"Deepfake error on frame {i}: {e}")

        # Liveness detection
        if liveness_engine is not None:
            try:
                live_result = liveness_engine.process_frame(frame, challenge_type)
                live_score = live_result.get("liveness_score", 0.0)
                liveness_scores.append(live_score)
                result_i["liveness"] = live_score
                result_i["challenge_verified"] = live_result.get("challenge_verified", False)
            except Exception as e:
                logger.warning(f"Liveness error on frame {i}: {e}")

        # Emotion detection
        if emotion_scorer is not None:
            try:
                emo_result = emotion_scorer.score_authenticity(frame, challenge_type)
                emo_score = emo_result.get("score", 0.0)
                emotion_scores.append(emo_score)
                result_i["emotion"] = emo_score
                result_i["detected_emotion"] = emo_result.get("detected_emotion", "unknown")
            except Exception as e:
                logger.warning(f"Emotion error on frame {i}: {e}")

        frame_results.append(result_i)

    elapsed = time.time() - t0

    # Store aggregated scores in session
    session["scores"]["deepfake_frames"] = deepfake_scores
    session["scores"]["liveness_frames"] = liveness_scores
    session["scores"]["emotion_frames"] = emotion_scores

    logger.info(
        f"Stream processed: {len(frames)} frames in {elapsed:.1f}s "
        f"(deepfake={len(deepfake_scores)}, liveness={len(liveness_scores)}, emotion={len(emotion_scores)})"
    )

    return {
        "session_id": session_id,
        "frames_processed": len(frame_results),
        "inference_time_ms": round(elapsed * 1000, 1),
        "frame_results": frame_results,
        "averages": {
            "deepfake": float(np.mean(deepfake_scores)) if deepfake_scores else None,
            "liveness": float(np.mean(liveness_scores)) if liveness_scores else None,
            "emotion": float(np.mean(emotion_scores)) if emotion_scores else None,
        },
    }


# ══════════════════════════════════════════════════════════════
#  VERIFY/AUDIO — Process captured audio
# ══════════════════════════════════════════════════════════════

@app.post("/api/v1/verify/audio")
async def verify_audio(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
):
    """Run audio deepfake detection on the captured audio segment."""
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    audio_bytes = await audio.read()

    if audio_detector is None:
        return {"session_id": session_id, "audio_fake_probability": None, "detail": "Audio detector not loaded"}

    if len(audio_bytes) < 500:
        return {"session_id": session_id, "audio_fake_probability": None, "detail": "Audio too short"}

    t0 = time.time()
    try:
        result = audio_detector.predict_from_bytes(audio_bytes)
        fake_prob = result.get("fake_probability", 0.5)
    except Exception as e:
        logger.warning(f"Audio detection error: {e}")
        fake_prob = 0.5
    elapsed = time.time() - t0

    session["scores"]["audio_fake_prob"] = fake_prob

    return {
        "session_id": session_id,
        "audio_fake_probability": fake_prob,
        "inference_time_ms": round(elapsed * 1000, 1),
    }


# ══════════════════════════════════════════════════════════════
#  AGGREGATE — Compute final verification score
# ══════════════════════════════════════════════════════════════

class AggregateRequest(BaseModel):
    session_id: str
    wallet_address: str

@app.post("/api/v1/aggregate")
async def aggregate_scores(req: AggregateRequest):
    """
    Aggregate all per-frame and audio scores into a final verdict.
    Returns scores + verification_hash + passed boolean.
    """
    session = sessions.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    scores_data = session["scores"]

    # Average deepfake scores across frames (lower = more real)
    df_scores = scores_data["deepfake_frames"]
    avg_deepfake = float(np.mean(df_scores)) if df_scores else 0.5

    # Average liveness scores across frames (higher = more live)
    live_scores = scores_data["liveness_frames"]
    avg_liveness = float(np.mean(live_scores)) if live_scores else 0.0

    # Average emotion scores across frames (higher = more authentic)
    emo_scores = scores_data["emotion_frames"]
    avg_emotion = float(np.mean(emo_scores)) if emo_scores else 0.5

    # Audio fake probability (lower = more real)
    audio_fake = scores_data.get("audio_fake_prob") or 0.5

    # Face match score (1.0 if enrolled, simplified for lite mode)
    wallet = req.wallet_address.lower()
    face_match = 1.0 if wallet in enrolled_faces else 0.5

    # ── Final weighted score ──────────────────────────────────
    # Higher = better (more likely to be a real, live person)
    real_video = max(0.0, 1.0 - avg_deepfake)    # invert: low fake = high real
    real_audio = max(0.0, 1.0 - audio_fake)       # invert: low fake = high real

    final_score = (
        0.25 * face_match
        + 0.25 * avg_liveness
        + 0.20 * real_video
        + 0.15 * real_audio
        + 0.15 * avg_emotion
    )

    passed = final_score >= PASS_THRESHOLD

    # Generate verification hash
    hash_input = f"{req.session_id}:{req.wallet_address}:{final_score:.6f}:{session['nonce']}"
    verification_hash = "0x" + hashlib.sha256(hash_input.encode()).hexdigest()

    session["completed"] = True
    session["final_score"] = final_score
    session["verification_hash"] = verification_hash
    session["passed"] = passed

    logger.info(
        f"Aggregated: score={final_score:.3f} passed={passed} "
        f"(face={face_match:.2f} live={avg_liveness:.2f} deepfake={avg_deepfake:.2f} "
        f"audio={audio_fake:.2f} emotion={avg_emotion:.2f})"
    )

    return {
        "session_id": req.session_id,
        "face_match": face_match,
        "liveness": avg_liveness,
        "deepfake": avg_deepfake,
        "audio_fake": audio_fake,
        "emotion": avg_emotion,
        "final_score": final_score,
        "passed": passed,
        "verification_hash": verification_hash,
    }


# ══════════════════════════════════════════════════════════════
#  MINT — Issue proof-of-life (stub for lite mode)
# ══════════════════════════════════════════════════════════════

class MintRequest(BaseModel):
    session_id: str
    wallet_address: str
    verification_hash: str

@app.post("/api/v1/mint")
async def mint_proof(req: MintRequest):
    """
    Mint endpoint stub — returns a mock tx_hash.
    In production this would sign an EIP-712 message and submit on-chain.
    """
    session = sessions.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.get("passed"):
        raise HTTPException(status_code=403, detail="Verification did not pass")

    # Generate a mock transaction hash for lite mode
    mock_tx = "0x" + hashlib.sha256(
        f"mint:{req.session_id}:{req.wallet_address}:{time.time()}".encode()
    ).hexdigest()

    logger.info(f"Mint requested: wallet={req.wallet_address[:10]}… tx={mock_tx[:18]}…")

    return {
        "success": True,
        "tx_hash": mock_tx,
        "token_id": 1,
        "message": "Proof-of-Life token minted (lite mode — no on-chain tx)",
    }


# ══════════════════════════════════════════════════════════════
#  WebSocket — Live score updates
# ══════════════════════════════════════════════════════════════

@app.websocket("/api/v1/ws/scores")
async def ws_scores(ws: WebSocket):
    """WebSocket endpoint for live score streaming during verification."""
    await ws.accept()
    try:
        # Wait for client to send session_id
        data = await ws.receive_json()
        sid = data.get("session_id", "")
        logger.info(f"WS connected for session {sid[:8]}…")

        import asyncio
        # Poll session scores and push updates
        last_frame_count = 0
        while True:
            await asyncio.sleep(0.5)
            session = sessions.get(sid)
            if session is None:
                await ws.send_json({"type": "error", "message": "Session not found"})
                break

            scores_data = session["scores"]
            current_count = len(scores_data.get("deepfake_frames", []))

            if current_count > last_frame_count:
                df = scores_data["deepfake_frames"]
                lv = scores_data["liveness_frames"]
                em = scores_data["emotion_frames"]
                await ws.send_json({
                    "type": "score_update",
                    "scores": {
                        "deep": float(np.mean(df)) if df else 0.0,
                        "live": float(np.mean(lv)) if lv else 0.0,
                        "emot": float(np.mean(em)) if em else 0.0,
                    },
                    "frames": current_count,
                })
                last_frame_count = current_count

            if session.get("completed"):
                await ws.send_json({"type": "complete", "final_score": session.get("final_score", 0)})
                break

    except WebSocketDisconnect:
        logger.info("WS client disconnected")
    except Exception as e:
        logger.warning(f"WS error: {e}")


# ══════════════════════════════════════════════════════════════
#  Base64 Inference
# ══════════════════════════════════════════════════════════════

@app.post("/api/v1/predict/deepfake")
async def predict_deepfake_b64(payload: dict):
    """
    Run deepfake detection on a base64-encoded image.
    Body: { "image_b64": "<base64 string>" }
    """
    import base64

    if deepfake_detector is None:
        raise HTTPException(status_code=503, detail="Deepfake detector not loaded")

    b64 = payload.get("image_b64", "")
    if not b64:
        raise HTTPException(status_code=400, detail="Missing image_b64 field")

    try:
        raw = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 data")

    nparr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result = deepfake_detector.predict_frame(frame)
    return {"model_loaded": deepfake_detector.is_loaded, **result}


# ══════════════════════════════════════════════════════════════
#  PUBLIC ANALYZE — Upload image or video for deepfake analysis
# ══════════════════════════════════════════════════════════════

@app.post("/api/v1/analyze")
async def analyze_media(file: UploadFile = File(...)):
    """
    Public endpoint — upload an image or video and get a full deepfake analysis.
    Runs all available models: deepfake, liveness, emotion.
    Accepts: JPEG, PNG, WebP images; MP4, WebM, AVI videos.
    """
    file_bytes = await file.read()
    filename = (file.filename or "upload").lower()

    if len(file_bytes) < 100:
        raise HTTPException(status_code=400, detail="File too small")

    # Determine if image or video
    is_video = any(filename.endswith(ext) for ext in (".mp4", ".webm", ".avi", ".mov", ".mkv"))
    is_image = any(filename.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"))

    # If no extension, try to decode as image first
    if not is_video and not is_image:
        nparr = np.frombuffer(file_bytes, np.uint8)
        test_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        is_image = test_frame is not None
        is_video = not is_image

    t0 = time.time()

    if is_image:
        result = _analyze_image(file_bytes)
    elif is_video:
        result = _analyze_video(file_bytes, filename)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload an image or video.")

    elapsed = time.time() - t0
    result["total_time_ms"] = round(elapsed * 1000, 1)
    result["filename"] = file.filename

    return result


def _analyze_image(image_bytes: bytes) -> dict:
    """Run all models on a single image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result: dict = {"type": "image", "frames_analyzed": 1}

    def _sanitize(obj):
        """Recursively convert numpy types to native Python types for JSON."""
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Deepfake detection
    if deepfake_detector is not None:
        try:
            t = time.time()
            df = deepfake_detector.predict_frame(frame)
            result["deepfake"] = _sanitize({
                "fake_probability": df.get("fake_probability", 0.5),
                "confidence": abs(df.get("fake_probability", 0.5) - 0.5) * 2,
                "inference_ms": round((time.time() - t) * 1000, 1),
            })
        except Exception as e:
            result["deepfake"] = {"error": str(e)}
    else:
        result["deepfake"] = {"error": "Model not loaded"}

    # Liveness detection
    if liveness_engine is not None:
        try:
            t = time.time()
            lv = liveness_engine.process_frame(frame, "SMILE_NATURALLY")
            result["liveness"] = _sanitize({
                "score": lv.get("liveness_score", 0.0),
                "face_detected": "error" not in str(lv.get("layers", {}).get("active", {}).get("details", {})),
                "inference_ms": round((time.time() - t) * 1000, 1),
            })
        except Exception as e:
            result["liveness"] = {"error": str(e)}
    else:
        result["liveness"] = {"error": "Model not loaded"}

    # Emotion detection
    if emotion_scorer is not None:
        try:
            t = time.time()
            em = emotion_scorer.score_authenticity(frame, "SMILE_NATURALLY")
            result["emotion"] = _sanitize({
                "detected": em.get("detected_emotion", "unknown"),
                "confidence": em.get("details", {}).get("emotion_confidence", 0.0),
                "all_emotions": em.get("details", {}).get("all_emotions", {}),
                "duchenne": em.get("duchenne"),
                "inference_ms": round((time.time() - t) * 1000, 1),
            })
        except Exception as e:
            result["emotion"] = {"error": str(e)}
    else:
        result["emotion"] = {"error": "Model not loaded"}

    # Compute overall verdict
    fake_prob = float(result.get("deepfake", {}).get("fake_probability", 0.5))
    result["verdict"] = {
        "is_likely_fake": bool(fake_prob > 0.5),
        "fake_probability": float(fake_prob),
        "real_probability": float(1.0 - fake_prob),
        "label": "LIKELY FAKE" if fake_prob > 0.65 else "LIKELY REAL" if fake_prob < 0.35 else "UNCERTAIN",
    }

    return result


def _analyze_video(video_bytes: bytes, filename: str) -> dict:
    """Extract frames from video and run models on each."""
    import tempfile

    # Write to temp file for OpenCV
    suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        duration = total_frames / fps if fps > 0 else 0

        # Sample up to 20 frames evenly from the video
        max_sample = min(20, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, max_sample, dtype=int) if total_frames > 1 else [0]

        deepfake_scores = []
        liveness_scores = []
        emotion_results = []
        frame_details = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            detail: dict = {"frame_idx": int(idx), "timestamp_s": round(int(idx) / fps, 2)}

            # Deepfake
            if deepfake_detector is not None:
                try:
                    df = deepfake_detector.predict_frame(frame)
                    fp = df.get("fake_probability", 0.5)
                    deepfake_scores.append(fp)
                    detail["fake_probability"] = fp
                except Exception:
                    pass

            # Liveness
            if liveness_engine is not None:
                try:
                    lv = liveness_engine.process_frame(frame, "SMILE_NATURALLY")
                    ls = lv.get("liveness_score", 0.0)
                    liveness_scores.append(ls)
                    detail["liveness_score"] = ls
                except Exception:
                    pass

            # Emotion
            if emotion_scorer is not None:
                try:
                    em = emotion_scorer.score_authenticity(frame, "SMILE_NATURALLY")
                    emotion_results.append(em.get("detected_emotion", "unknown"))
                    detail["emotion"] = em.get("detected_emotion", "unknown")
                except Exception:
                    pass

            frame_details.append(detail)

        cap.release()

        avg_fake = float(np.mean(deepfake_scores)) if deepfake_scores else 0.5
        avg_liveness = float(np.mean(liveness_scores)) if liveness_scores else 0.0

        # Most common emotion
        top_emotion = "unknown"
        if emotion_results:
            from collections import Counter
            top_emotion = Counter(emotion_results).most_common(1)[0][0]

        result: dict = {
            "type": "video",
            "video_info": {
                "duration_s": round(duration, 2),
                "total_frames": total_frames,
                "fps": round(fps, 1),
            },
            "frames_analyzed": len(frame_details),
            "deepfake": {
                "avg_fake_probability": float(avg_fake),
                "min_fake": float(min(deepfake_scores)) if deepfake_scores else None,
                "max_fake": float(max(deepfake_scores)) if deepfake_scores else None,
                "per_frame": [float(x) for x in deepfake_scores],
            },
            "liveness": {
                "avg_score": float(avg_liveness),
            },
            "emotion": {
                "dominant": top_emotion,
                "per_frame": emotion_results,
            },
            "verdict": {
                "is_likely_fake": bool(avg_fake > 0.5),
                "fake_probability": float(avg_fake),
                "real_probability": float(1.0 - avg_fake),
                "label": "LIKELY FAKE" if avg_fake > 0.65 else "LIKELY REAL" if avg_fake < 0.35 else "UNCERTAIN",
            },
            "frame_details": frame_details,
        }

        return result

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server_lite:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
