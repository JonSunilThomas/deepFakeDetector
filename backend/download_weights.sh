#!/usr/bin/env bash
# ── Download ML Weights ──────────────────────────────────────
# Called during Docker build to fetch model weights that are
# gitignored. Set WEIGHTS_BASE_URL to your hosting location.
#
# Supported hosts (pick one):
#   • GitHub Releases:  https://github.com/<user>/<repo>/releases/download/v1.0
#   • Hugging Face Hub: https://huggingface.co/<user>/<repo>/resolve/main
#   • S3 / R2 bucket:   https://<bucket>.s3.amazonaws.com/weights
# ──────────────────────────────────────────────────────────────
set -euo pipefail

WEIGHTS_BASE_URL="${WEIGHTS_BASE_URL:-}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="$(cd "$(dirname "$0")" && pwd)"

# Target directories
mkdir -p "$PROJECT_ROOT/ml/deepfake/weights"
mkdir -p "$PROJECT_ROOT/ml/audio/weights"
mkdir -p "$PROJECT_ROOT/ml/emotion/weights"
mkdir -p "$BACKEND_DIR/weights"

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

echo "═══════════════════════════════════════════"
echo "  Downloading ML model weights..."
echo "═══════════════════════════════════════════"

if [ -z "$WEIGHTS_BASE_URL" ]; then
    echo ""
    echo "  ⚠ WEIGHTS_BASE_URL is not set."
    echo "    Models will use random initialization."
    echo "    To use trained weights, set WEIGHTS_BASE_URL"
    echo "    to the base URL where your .pth files are hosted."
    echo ""
    echo "    Example:"
    echo "      WEIGHTS_BASE_URL=https://github.com/you/repo/releases/download/v1.0"
    echo ""
    echo "    Expected files at that URL:"
    echo "      \$WEIGHTS_BASE_URL/xception_celeb_df.pth"
    echo "      \$WEIGHTS_BASE_URL/audio_cnn_lstm.pth"
    echo "      \$WEIGHTS_BASE_URL/face_landmarker.task"
    echo ""
    exit 0
fi

echo ""
echo "  Base URL: $WEIGHTS_BASE_URL"
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

# Emotion model (optional — uncomment when trained)
# download "$WEIGHTS_BASE_URL/emotion_model.pth" \
#          "$PROJECT_ROOT/ml/emotion/weights/emotion_model.pth"

echo ""
echo "  ✅ Weight download complete."
echo "═══════════════════════════════════════════"
