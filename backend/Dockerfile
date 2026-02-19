# ── DeepFake Detector Backend ─────────────────────────────────
# Optimised for Hugging Face Spaces (Docker SDK)
#
# HF Spaces requirements:
#   - Must expose port 7860
#   - Runs as non-root user (uid 1000)
#   - Build context is the REPO ROOT (not backend/)
#     because server_lite.py resolves weights via ../ml/
#
# Local build/test:
#   docker build -f backend/Dockerfile -t deepfake-backend .
#   docker run -p 7860:7860 deepfake-backend
# ──────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps for OpenCV, MediaPipe, librosa (libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libsndfile1 \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user (HF Spaces requirement) ─────────────
RUN useradd -m -u 1000 appuser

# Mirror the repo structure so PROJECT_ROOT resolves correctly.
# server_lite.py uses: PROJECT_ROOT = Path(__file__).parent.parent
#   /opt/app/           ← PROJECT_ROOT
#   /opt/app/backend/   ← WORKDIR (where server_lite.py lives)
#   /opt/app/ml/        ← weight directories
WORKDIR /opt/app/backend

# ── Install Python deps (cached layer) ───────────────────────
COPY backend/requirements-deploy.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────
COPY backend/ ./
COPY ml/ /opt/app/ml/

# ── Download ML weights at build time ────────────────────────
# Weights are fetched from a Hugging Face model repo or any URL.
# Set HF_WEIGHTS_REPO (e.g. "JonSunilThomas/deepfake-weights")
# or WEIGHTS_BASE_URL for direct download links.
# If neither is set, models use random initialization.
ARG HF_WEIGHTS_REPO=""
ARG WEIGHTS_BASE_URL=""
ENV HF_WEIGHTS_REPO=${HF_WEIGHTS_REPO}
ENV WEIGHTS_BASE_URL=${WEIGHTS_BASE_URL}
RUN chmod +x download_weights.sh && ./download_weights.sh

# ── Fix permissions for HF Spaces ────────────────────────────
RUN chown -R appuser:appuser /opt/app

USER appuser

# ── Expose port 7860 (HF Spaces default) ─────────────────────
EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV JAX_PLATFORMS=cpu
ENV CUDA_VISIBLE_DEVICES=""

CMD ["python", "server_lite.py"]
