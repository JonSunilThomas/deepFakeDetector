"""
Emotion Authenticity Scorer — CNN-based emotion classifier
with Duchenne smile detection (AU6 + AU12 analysis).
Trained on FER2013/AffectNet.

Updated for mediapipe >= 0.10.30 (tasks API, no mp.solutions).
"""
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ── MediaPipe Tasks API ──────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

logger = logging.getLogger(__name__)

# Default model path — same as face_liveness.py
_DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent / "weights" / "face_landmarker.task"
)

# ── Emotion Labels ────────────────────────────────────────────
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTION_TO_IDX = {label: i for i, label in enumerate(EMOTION_LABELS)}

_EMOTION_INPUT_SIZE = 48

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((_EMOTION_INPUT_SIZE, _EMOTION_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# ══════════════════════════════════════════════════════════════
#  Emotion CNN (FER2013-compatible architecture)
# ══════════════════════════════════════════════════════════════

class EmotionCNN(nn.Module):
    """
    Compact CNN for facial emotion recognition.
    Input: Grayscale face image (1, 48, 48)
    Output: 7-class probability distribution
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ══════════════════════════════════════════════════════════════
#  Duchenne Smile Detector (AU6 + AU12 via MediaPipe)
# ══════════════════════════════════════════════════════════════

# Key landmarks for Action Units
_AU6_LANDMARKS = {  # Cheek raiser (orbicularis oculi)
    "left_cheek_upper": 123,
    "left_eye_lower": 111,
    "right_cheek_upper": 352,
    "right_eye_lower": 340,
}

_AU12_LANDMARKS = {  # Lip corner puller (zygomaticus major)
    "mouth_left": 61,
    "mouth_right": 291,
    "nose_base": 2,
    "chin": 199,
}


def _create_face_landmarker(model_path: str = _DEFAULT_MODEL_PATH):
    """Create a FaceLandmarker using the tasks API."""
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


def _detect_landmarks(landmarker: FaceLandmarker, frame_bgr: np.ndarray):
    """Run detection and return the first face's landmark list, or None."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if result.face_landmarks:
        return result.face_landmarks[0]  # List of NormalizedLandmark
    return None


def _lm_point(landmarks, idx, w, h):
    """Extract a 2D point from the tasks-API landmark list."""
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def detect_duchenne_smile(landmarks, img_w: int, img_h: int) -> dict:
    """
    Detect a Duchenne (genuine) smile by analyzing:
    - AU6: Cheek raiser — the eyes narrow and crinkle
    - AU12: Lip corner puller — mouth corners move up

    A Duchenne smile activates both AU6 and AU12.
    A non-Duchenne (social/fake) smile only activates AU12.

    Returns:
        dict with 'is_duchenne', 'au6_score', 'au12_score', 'details'.
    """
    # AU6: Measure cheek raise (distance between lower eye and upper cheek)
    left_cheek = _lm_point(landmarks, _AU6_LANDMARKS["left_cheek_upper"], img_w, img_h)
    left_eye_low = _lm_point(landmarks, _AU6_LANDMARKS["left_eye_lower"], img_w, img_h)
    right_cheek = _lm_point(landmarks, _AU6_LANDMARKS["right_cheek_upper"], img_w, img_h)
    right_eye_low = _lm_point(landmarks, _AU6_LANDMARKS["right_eye_lower"], img_w, img_h)

    left_au6_dist = np.linalg.norm(left_cheek - left_eye_low)
    right_au6_dist = np.linalg.norm(right_cheek - right_eye_low)
    au6_avg = (left_au6_dist + right_au6_dist) / 2.0

    # Normalize by face height
    nose = _lm_point(landmarks, _AU12_LANDMARKS["nose_base"], img_w, img_h)
    chin = _lm_point(landmarks, _AU12_LANDMARKS["chin"], img_w, img_h)
    face_height = np.linalg.norm(nose - chin) + 1e-6

    au6_normalized = au6_avg / face_height
    # Lower value = more cheek raise (eye narrows)
    au6_active = au6_normalized < 0.35
    au6_score = max(0.0, min(1.0, 1.0 - (au6_normalized / 0.5)))

    # AU12: Measure lip corner elevation
    mouth_left = _lm_point(landmarks, _AU12_LANDMARKS["mouth_left"], img_w, img_h)
    mouth_right = _lm_point(landmarks, _AU12_LANDMARKS["mouth_right"], img_w, img_h)

    mouth_width = np.linalg.norm(mouth_left - mouth_right)
    mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2.0
    nose_y = nose[1]

    # Mouth-to-nose vertical ratio — smile pulls corners up
    mouth_nose_ratio = (nose_y - mouth_center_y) / face_height
    au12_active = mouth_nose_ratio > -0.15 and mouth_width / face_height > 0.45
    au12_score = min(1.0, max(0.0, (mouth_width / face_height - 0.3) / 0.3))

    is_duchenne = au6_active and au12_active

    return {
        "is_duchenne": is_duchenne,
        "au6_score": float(au6_score),
        "au12_score": float(au12_score),
        "details": {
            "au6_normalized": float(au6_normalized),
            "mouth_nose_ratio": float(mouth_nose_ratio),
            "mouth_width_ratio": float(mouth_width / face_height),
        },
    }


# ══════════════════════════════════════════════════════════════
#  Emotion Authenticity Engine
# ══════════════════════════════════════════════════════════════

class EmotionAuthenticityScorer:
    """
    Combined emotion classification + Duchenne analysis.
    Ensures emotional response is coherent with the challenge prompt.
    """

    # Challenge-to-expected-emotion mapping
    CHALLENGE_EMOTIONS = {
        "SMILE_NATURALLY": "happy",
        "LEFT": None,
        "RIGHT": None,
        "UP": None,
        "DOWN": None,
        "BLINK_TWICE": None,
        "RAISE_RIGHT_EYEBROW": "surprise",
    }

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        model_path: str = _DEFAULT_MODEL_PATH,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = EmotionCNN()

        if weights_path:
            try:
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded emotion model from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load emotion weights: {e}. Using random init.")

        self.model.to(self.device)
        self.model.eval()

        # MediaPipe tasks API — replaces deprecated mp.solutions.face_mesh
        self._landmarker = _create_face_landmarker(model_path)

        # Face detector for cropping
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    @torch.no_grad()
    def classify_emotion(self, frame: np.ndarray) -> dict:
        """
        Classify the dominant emotion in a face image.

        Returns:
            dict with 'emotion', 'confidence', 'all_scores'.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))

        if len(faces) == 0:
            return {"emotion": "neutral", "confidence": 0.0, "all_scores": {}}

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = frame[y:y+h, x:x+w]

        tensor = _transform(face_roi).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        all_scores = {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}

        return {
            "emotion": EMOTION_LABELS[top_idx],
            "confidence": float(probs[top_idx]),
            "all_scores": all_scores,
        }

    def score_authenticity(
        self, frame: np.ndarray, challenge_type: str
    ) -> dict:
        """
        Score the emotional authenticity of a frame against the challenge.

        Returns:
            dict with 'score', 'emotion_match', 'duchenne', 'details'.
        """
        # Emotion classification
        emotion_result = self.classify_emotion(frame)

        # Duchenne smile detection via mediapipe tasks API
        img_h, img_w = frame.shape[:2]
        landmarks = _detect_landmarks(self._landmarker, frame)

        duchenne_result = None
        if landmarks is not None:
            duchenne_result = detect_duchenne_smile(landmarks, img_w, img_h)

        # Check emotion coherence with challenge
        expected = self.CHALLENGE_EMOTIONS.get(challenge_type.upper())
        emotion_match_score = 1.0

        if expected is not None:
            detected_emotion = emotion_result["emotion"]
            if detected_emotion == expected:
                emotion_match_score = emotion_result["confidence"]
            else:
                # Partial credit if expected emotion has some probability
                emotion_match_score = emotion_result["all_scores"].get(expected, 0.0)

        # Duchenne bonus for smile challenges
        duchenne_bonus = 0.0
        if challenge_type.upper() == "SMILE_NATURALLY" and duchenne_result:
            if duchenne_result["is_duchenne"]:
                duchenne_bonus = 0.2  # Genuine smile bonus
            else:
                emotion_match_score *= 0.7  # Penalty for fake smile

        # Mechanical expression detection: high confidence + low variance = suspicious
        mechanical_penalty = 0.0
        if emotion_result["confidence"] > 0.98:
            # Overly confident = possibly unnatural
            mechanical_penalty = 0.1

        # Final score
        score = min(1.0, max(0.0, emotion_match_score + duchenne_bonus - mechanical_penalty))

        # If no expected emotion, just check that face shows any natural expression
        if expected is None:
            # For non-emotion challenges, base score on whether the face looks natural
            # (not frozen, not mechanical)
            neutral_prob = emotion_result["all_scores"].get("neutral", 0.0)
            score = max(0.5, 1.0 - mechanical_penalty)

        return {
            "score": float(score),
            "emotion_match": expected == emotion_result["emotion"] if expected else True,
            "detected_emotion": emotion_result["emotion"],
            "expected_emotion": expected,
            "duchenne": duchenne_result,
            "details": {
                "emotion_confidence": emotion_result["confidence"],
                "all_emotions": emotion_result["all_scores"],
                "duchenne_bonus": duchenne_bonus,
                "mechanical_penalty": mechanical_penalty,
            },
        }

    def close(self):
        self._landmarker.close()
