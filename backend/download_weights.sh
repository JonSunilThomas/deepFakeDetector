#!/usr/bin/env bash
# ── Download ML Weights ──────────────────────────────────────
# Called during Docker build to fetch model weights.
#
# Supports two modes:
#   1. HF_WEIGHTS_REPO  — Download from a Hugging Face model repo
#      e.g. HF_WEIGHTS_REPO=JonSunilThomas/deepfake-weights
#
#   2. WEIGHTS_BASE_URL — Download from any HTTP URL
#      e.g. WEIGHTS_BASE_URL=https://github.com/user/repo/releases/download/v1.0
#
# If neither is set, models use random initialization (still works).
# ──────────────────────────────────────────────────────────────
set -euo pipefail

HF_WEIGHTS_REPO="${HF_WEIGHTS_REPO:-}"
WEIGHTS_BASE_URL="${WEIGHTS_BASE_URL:-}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="$(cd "$(dirname "$0")" && pwd)"

# Target directories
mkdir -p "$PROJECT_ROOT/ml/deepfake/weights"
mkdir -p "$PROJECT_ROOT/ml/audio/weights"
mkdir -p "$PROJECT_ROOT/ml/emotion/weights"
mkdir -p "$BACKEND_DIR/weights"

echo "═══════════════════════════════════════════"
echo "  Downloading ML model weights..."
echo "═══════════════════════════════════════════"

# ── Mode 1: Hugging Face Hub ─────────────────────────────────
if [ -n "$HF_WEIGHTS_REPO" ]; then
    echo ""
    echo "  Using Hugging Face repo: $HF_WEIGHTS_REPO"
    echo ""

    python3 -c "
from huggingface_hub import hf_hub_download
import os, shutil

repo = os.environ['HF_WEIGHTS_REPO']
project = '$(echo $PROJECT_ROOT)'
backend = '$(echo $BACKEND_DIR)'

files = {
    'xception_celeb_df.pth':  f'{project}/ml/deepfake/weights/xception_celeb_df.pth',
    'audio_cnn_lstm.pth':     f'{project}/ml/audio/weights/audio_cnn_lstm.pth',
    'face_landmarker.task':   f'{backend}/weights/face_landmarker.task',
    # 'emotion_model.pth':    f'{project}/ml/emotion/weights/emotion_model.pth',
}

for filename, dest in files.items():
    if os.path.exists(dest):
        print(f'  ✓ Already exists: {dest}')
        continue
    try:
        print(f'  ↓ Downloading: {repo}/{filename}')
        path = hf_hub_download(repo_id=repo, filename=filename)
        shutil.copy2(path, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f'    ✅ {size_mb:.1f} MB → {dest}')
    except Exception as e:
        print(f'    ⚠ Failed: {e} (non-fatal)')
"

    echo ""
    echo "  ✅ Hugging Face download complete."
    echo "═══════════════════════════════════════════"
    exit 0
fi

# ── Mode 2: Direct URL download ──────────────────────────────
download() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        echo "  ✓ Already exists: $dest"
        return 0
    fi
    echo "  ↓ Downloading: $url"
    echo "    → $dest"
    wget -q --show-progress -O "$dest" "$url" || {
        echo "  ⚠ Failed to download $url (non-fatal)"
        return 0
    }
}

if [ -n "$WEIGHTS_BASE_URL" ]; then
    echo ""
    echo "  Using base URL: $WEIGHTS_BASE_URL"
    echo ""

    # XceptionNet deepfake detector (~80 MB)
    download "$WEIGHTS_BASE_URL/xception_celeb_df.pth" \
             "$PROJECT_ROOT/ml/deepfake/weights/xception_celeb_df.pth"

    # CNN-LSTM audio deepfake detector (~4.6 MB)
    download "$WEIGHTS_BASE_URL/audio_cnn_lstm.pth" \
             "$PROJECT_ROOT/ml/audio/weights/audio_cnn_lstm.pth"

    # MediaPipe face landmarker (~3.6 MB)
    download "$WEIGHTS_BASE_URL/face_landmarker.task" \
             "$BACKEND_DIR/weights/face_landmarker.task"

    echo ""
    echo "  ✅ URL download complete."
    echo "═══════════════════════════════════════════"
    exit 0
fi

# ── Neither set ──────────────────────────────────────────────
echo ""
echo "  ⚠ No weight source configured."
echo "    Models will use random initialization."
echo ""
echo "    Set one of:"
echo "      HF_WEIGHTS_REPO=JonSunilThomas/deepfake-weights"
echo "      WEIGHTS_BASE_URL=https://example.com/weights"
echo ""
echo "═══════════════════════════════════════════"
