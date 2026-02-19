"""
Video Deepfake Detector — XceptionNet with temporal smoothing.
Fine-tuned on Celeb-DF dataset for frame-level deepfake classification.
"""
import logging
from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)

# ── Preprocessing ─────────────────────────────────────────────

_XCEPTION_INPUT_SIZE = 299

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((_XCEPTION_INPUT_SIZE, _XCEPTION_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ══════════════════════════════════════════════════════════════
#  XceptionNet — Depthwise Separable Convolution Architecture
# ══════════════════════════════════════════════════════════════

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reps, stride=1, start_with_relu=True, grow_first=True):
        super().__init__()

        # Residual connection
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

        layers = []
        channels = in_channels

        for i in range(reps):
            if grow_first:
                inc = in_channels if i == 0 else out_channels
                outc = out_channels
            else:
                inc = in_channels if i == 0 else in_channels
                outc = in_channels if i < reps - 1 else out_channels

            if start_with_relu or i > 0:
                layers.append(nn.ReLU(inplace=False))
            layers.append(SeparableConv2d(inc, outc, 3, 1, 1))
            layers.append(nn.BatchNorm2d(outc))

        if stride != 1:
            layers.append(nn.MaxPool2d(3, stride, 1))

        self.rep = nn.Sequential(*layers)

    def forward(self, x):
        return self.rep(x) + self.skip(x)


class XceptionNet(nn.Module):
    """
    Xception architecture adapted for binary deepfake detection.
    Based on "Xception: Deep Learning with Depthwise Separable Convolutions"
    (Chollet, 2017) — modified for binary classification.
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)

        self.block1 = XceptionBlock(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = XceptionBlock(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = XceptionBlock(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # Middle flow (8 blocks)
        self.middle_blocks = nn.Sequential(
            *[XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True) for _ in range(8)]
        )

        # Exit flow
        self.block_exit = XceptionBlock(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=False)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.relu4 = nn.ReLU(inplace=False)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Entry flow
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.middle_blocks(x)

        # Exit flow
        x = self.block_exit(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ══════════════════════════════════════════════════════════════
#  Video Deepfake Detector Wrapper
# ══════════════════════════════════════════════════════════════

class VideoDeepfakeDetector:
    """
    Wraps XceptionNet for frame-level deepfake detection with
    temporal smoothing across a sliding window.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        temporal_window: int = 15,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = XceptionNet(num_classes=1)
        self._loaded = False

        if weights_path:
            self._load_weights(weights_path)

        self.model.to(self.device)
        self.model.eval()

        self._score_history = deque(maxlen=temporal_window)

        # Face detector for cropping
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _load_weights(self, weights_path: str) -> None:
        """Robustly load XceptionNet weights from various checkpoint formats."""
        from pathlib import Path as _Path
        wp = _Path(weights_path)
        if not wp.exists():
            logger.warning(f"XceptionNet weights not found at {wp} — using random init")
            return
        try:
            checkpoint = torch.load(str(wp), map_location=self.device, weights_only=False)
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                state_dict = (
                    checkpoint.get("model_state_dict")
                    or checkpoint.get("state_dict")
                    or checkpoint
                )
            else:
                state_dict = checkpoint
            # Strip DataParallel 'module.' prefix if present
            cleaned = {
                (k[len("module."):] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }
            self.model.load_state_dict(cleaned, strict=False)
            self._loaded = True
            n_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Loaded XceptionNet weights from {wp} ({n_params:,} params)")
        except Exception as e:
            logger.warning(f"Could not load XceptionNet weights from {wp}: {e}. Using random init.")

    @property
    def is_loaded(self) -> bool:
        """Whether trained weights were successfully loaded."""
        return self._loaded

    def _extract_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect and crop the largest face from a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        if len(faces) == 0:
            return None

        # Take the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Add padding
        pad = int(0.2 * max(w, h))
        y1 = max(0, y - pad)
        y2 = min(frame.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(frame.shape[1], x + w + pad)

        return frame[y1:y2, x1:x2]

    @torch.no_grad()
    def predict_frame(self, frame: np.ndarray) -> dict:
        """
        Run deepfake detection on a single frame.

        Returns:
            dict with 'fake_probability', 'smoothed_probability', 'details'.
        """
        face = self._extract_face(frame)
        if face is None:
            return {
                "fake_probability": 0.5,
                "smoothed_probability": 0.5,
                "details": {"error": "No face detected"},
            }

        # Preprocess
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        tensor = _transform(rgb_face).unsqueeze(0).to(self.device)

        # Inference
        logit = self.model(tensor)
        fake_prob = float(torch.sigmoid(logit).cpu().item())

        self._score_history.append(fake_prob)

        # Temporal smoothing (weighted moving average — recent frames weighted more)
        weights = np.exp(np.linspace(-1, 0, len(self._score_history)))
        weights /= weights.sum()
        smoothed = float(np.dot(list(self._score_history), weights))

        return {
            "fake_probability": fake_prob,
            "smoothed_probability": smoothed,
            "details": {
                "raw_logit": float(logit.cpu().item()),
                "window_size": len(self._score_history),
            },
        }

    def reset(self):
        self._score_history.clear()
