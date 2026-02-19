"""
Audio Deepfake Detector — CNN-LSTM model on MFCC features.
Classifies audio as real or synthetically generated speech.
"""
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Audio Preprocessing ───────────────────────────────────────

# MFCC extraction parameters
SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
MAX_DURATION_SEC = 5  # clip to 5 seconds
MAX_FRAMES = int(SAMPLE_RATE * MAX_DURATION_SEC / HOP_LENGTH) + 1


def extract_mfcc(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract MFCC features from raw audio waveform.
    Uses manual implementation to avoid hard librosa dependency at inference.

    Args:
        audio: 1D numpy array of audio samples (float32, [-1, 1]).
        sr: Sample rate.

    Returns:
        MFCC features of shape (n_mfcc, time_frames).
    """
    try:
        import librosa
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
    except ImportError:
        # Fallback: compute via scipy
        from scipy.fft import dct
        from scipy.signal import stft

        # STFT
        _, _, Zxx = stft(audio, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)
        power_spectrum = np.abs(Zxx) ** 2

        # Mel filterbank
        n_mels = 128
        mel_basis = _mel_filterbank(sr, N_FFT, n_mels)
        mel_spec = mel_basis @ power_spectrum
        log_mel = np.log(mel_spec + 1e-9)

        # DCT to get MFCCs
        mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:N_MFCC]

    # Pad or truncate to fixed length
    if mfcc.shape[1] > MAX_FRAMES:
        mfcc = mfcc[:, :MAX_FRAMES]
    elif mfcc.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")

    return mfcc.astype(np.float32)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Create a Mel filterbank matrix."""
    f_min = 0.0
    f_max = sr / 2.0

    def _hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    n_freq = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freq))

    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        for j in range(left, center):
            if center > left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


def load_audio_file(file_path: str) -> np.ndarray:
    """Load an audio file and return as float32 array."""
    try:
        import librosa
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION_SEC)
        return audio
    except ImportError:
        import wave
        import struct
        with wave.open(file_path, "rb") as wf:
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            samples = struct.unpack(f"<{n_frames}h", raw)
            audio = np.array(samples, dtype=np.float32) / 32768.0
            return audio[:SAMPLE_RATE * MAX_DURATION_SEC]


def load_audio_bytes(data: bytes, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio from raw bytes (WAV format expected)."""
    import io
    import wave
    import struct

    with wave.open(io.BytesIO(data), "rb") as wf:
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()

        if sampwidth == 2:
            fmt = f"<{n_frames * n_channels}h"
            samples = np.array(struct.unpack(fmt, raw), dtype=np.float32) / 32768.0
        else:
            samples = np.frombuffer(raw, dtype=np.float32)

        # Convert to mono
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        return samples[:SAMPLE_RATE * MAX_DURATION_SEC]


# ══════════════════════════════════════════════════════════════
#  CNN-LSTM Audio Deepfake Detection Model
# ══════════════════════════════════════════════════════════════

class AudioCNNLSTM(nn.Module):
    """
    CNN-LSTM architecture for audio deepfake detection.
    Input: MFCC spectrogram (batch, 1, n_mfcc, time_frames)
    Output: Probability of synthetic speech (sigmoid).
    """

    def __init__(self, n_mfcc: int = N_MFCC, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),  # Collapse frequency dimension
        )

        # LSTM temporal processor
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 1, n_mfcc, time_frames)
        Returns:
            logit: (batch, 1)
        """
        # CNN
        cnn_out = self.cnn(x)  # (batch, 256, 1, T')
        cnn_out = cnn_out.squeeze(2)  # (batch, 256, T')
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch, T', 256)

        # LSTM
        lstm_out, _ = self.lstm(cnn_out)  # (batch, T', hidden*2)

        # Take last time step
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden*2)

        # Classify
        logit = self.classifier(last_hidden)
        return logit


# ══════════════════════════════════════════════════════════════
#  Audio Deepfake Detector Wrapper
# ══════════════════════════════════════════════════════════════

class AudioDeepfakeDetector:
    """
    High-level wrapper for audio deepfake detection.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = AudioCNNLSTM()
        self._loaded = False

        if weights_path:
            self._load_weights(weights_path)

        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, weights_path: str) -> None:
        """Robustly load CNN-LSTM weights from various checkpoint formats."""
        from pathlib import Path as _Path
        wp = _Path(weights_path)
        if not wp.exists():
            logger.warning(f"Audio weights not found at {wp} — using random init")
            return
        try:
            checkpoint = torch.load(str(wp), map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict):
                state_dict = (
                    checkpoint.get("model_state_dict")
                    or checkpoint.get("state_dict")
                    or checkpoint
                )
            else:
                state_dict = checkpoint
            cleaned = {
                (k[len("module."):] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }
            self.model.load_state_dict(cleaned, strict=False)
            self._loaded = True
            n_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Loaded AudioCNNLSTM weights from {wp} ({n_params:,} params)")
        except Exception as e:
            logger.warning(f"Could not load audio weights from {wp}: {e}. Using random init.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @torch.no_grad()
    def predict(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
        """
        Predict whether audio is synthetic.

        Args:
            audio: 1D float32 numpy array.
            sr: Sample rate.

        Returns:
            dict with 'fake_probability', 'details'.
        """
        if len(audio) < sr * 0.5:
            return {
                "fake_probability": 0.5,
                "details": {"error": "Audio too short (< 0.5s)"},
            }

        # Extract MFCC
        mfcc = extract_mfcc(audio, sr)

        # To tensor: (1, 1, n_mfcc, time_frames)
        tensor = torch.from_numpy(mfcc).unsqueeze(0).unsqueeze(0).to(self.device)

        # Inference
        logit = self.model(tensor)
        fake_prob = float(torch.sigmoid(logit).cpu().item())

        return {
            "fake_probability": fake_prob,
            "details": {
                "raw_logit": float(logit.cpu().item()),
                "audio_duration_sec": len(audio) / sr,
                "n_mfcc_frames": mfcc.shape[1],
            },
        }

    @torch.no_grad()
    def predict_from_bytes(self, audio_bytes: bytes) -> dict:
        """Predict from raw WAV bytes."""
        audio = load_audio_bytes(audio_bytes)
        return self.predict(audio)

    @torch.no_grad()
    def predict_from_file(self, file_path: str) -> dict:
        """Predict from an audio file path."""
        audio = load_audio_file(file_path)
        return self.predict(audio)
