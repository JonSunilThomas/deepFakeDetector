# ── DeepFake Detector Backend ─────────────────────────────────
# Optimised for Render.com (Docker web service)
#
# IMPORTANT: Build context must be the REPO ROOT (not backend/),
# because server_lite.py resolves weights via ../ml/.
#
#   docker build -f backend/Dockerfile -t deepfake-backend .
#   docker run -p 8000:8000 deepfake-backend
# ──────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

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

# Mirror the repo structure so PROJECT_ROOT resolves correctly.
# server_lite.py uses: PROJECT_ROOT = Path(__file__).parent.parent
#   /opt/app/           ← PROJECT_ROOT
#   /opt/app/backend/   ← WORKDIR (where server_lite.py lives)
#   /opt/app/ml/        ← weight directories
WORKDIR /opt/app/backend

# ── Install Python deps (cached layer) ───────────────────────
COPY backend/requirements-render.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────
COPY backend/ ./
COPY ml/ /opt/app/ml/

# ── Download ML weights at build time ────────────────────────
# Set WEIGHTS_BASE_URL env var to your hosting location.
# If unset, models use random initialization (still functional).
ARG WEIGHTS_BASE_URL=""
ENV WEIGHTS_BASE_URL=${WEIGHTS_BASE_URL}
RUN chmod +x download_weights.sh && ./download_weights.sh

# ── Expose & run ─────────────────────────────────────────────
EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV JAX_PLATFORMS=cpu
ENV CUDA_VISIBLE_DEVICES=""

CMD ["python", "server_lite.py"]
