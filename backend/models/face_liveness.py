"""
3-Layer Liveness Detection System:
  Layer 1 — Active Challenge (head pose, blink, eyebrow, smile via MediaPipe)
  Layer 2 — Depth / Optical Flow (Lucas-Kanade for replay detection)
  Layer 3 — rPPG (remote photoplethysmography for heart-rate detection)

Updated for mediapipe >= 0.10.30 (tasks API, no mp.solutions).
"""
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

# ── MediaPipe Tasks API ──────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

# Default model path — downloaded from Google
_DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent / "weights" / "face_landmarker.task"
)

# Landmark indices (same as old mp.solutions — 478 landmarks)
_NOSE_TIP = 1
_CHIN = 199
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263
_LEFT_MOUTH = 61
_RIGHT_MOUTH = 291
_POSE_LANDMARKS = [_LEFT_EYE_OUTER, _RIGHT_EYE_OUTER, _NOSE_TIP, _LEFT_MOUTH, _RIGHT_MOUTH, _CHIN]

# Blink detection landmarks (EAR — Eye Aspect Ratio)
_LEFT_EYE_TOP = 159
_LEFT_EYE_BOTTOM = 145
_LEFT_EYE_LEFT = 33
_LEFT_EYE_RIGHT = 133
_RIGHT_EYE_TOP = 386
_RIGHT_EYE_BOTTOM = 374
_RIGHT_EYE_LEFT = 362
_RIGHT_EYE_RIGHT = 263

# Eyebrow landmarks
_LEFT_EYEBROW_TOP = 105
_LEFT_EYEBROW_BOTTOM = 65
_RIGHT_EYEBROW_TOP = 334
_RIGHT_EYEBROW_BOTTOM = 295

# Mouth landmarks for smile detection
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291
_MOUTH_TOP = 13
_MOUTH_BOTTOM = 14
_LEFT_CHEEK = 234
_RIGHT_CHEEK = 454

# rPPG ROI landmarks (forehead region)
_FOREHEAD_POINTS = [10, 67, 69, 104, 108, 151, 299, 337, 338, 297]


def _get_landmark_point(landmarks_list, idx, img_w, img_h):
    """Extract a 2D point from the new tasks-API landmark list."""
    lm = landmarks_list[idx]
    return np.array([lm.x * img_w, lm.y * img_h])


def _create_face_landmarker(model_path: str = _DEFAULT_MODEL_PATH, running_mode=RunningMode.IMAGE):
    """Create a FaceLandmarker using the tasks API."""
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=running_mode,
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


# ══════════════════════════════════════════════════════════════
#  LAYER 1: Active Challenge Detection
# ══════════════════════════════════════════════════════════════

def get_head_pose(landmarks, img_w: int, img_h: int) -> tuple[float, float, float]:
    """Estimate head pose (pitch, yaw, roll) from face landmarks via PnP."""
    face_2d = []
    face_3d = []

    for idx in _POSE_LANDMARKS:
        lm = landmarks[idx]
        x, y = lm.x * img_w, lm.y * img_h
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = img_w
    cam_matrix = np.array(
        [[focal_length, 0, img_w / 2],
         [0, focal_length, img_h / 2],
         [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs)
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch = angles[0] * 360
    yaw = angles[1] * 360
    roll = angles[2] * 360
    return pitch, yaw, roll


def compute_eye_aspect_ratio(landmarks, img_w: int, img_h: int) -> float:
    """Compute average EAR (Eye Aspect Ratio) for blink detection."""
    def _ear(top, bottom, left, right):
        p_top = _get_landmark_point(landmarks, top, img_w, img_h)
        p_bot = _get_landmark_point(landmarks, bottom, img_w, img_h)
        p_lft = _get_landmark_point(landmarks, left, img_w, img_h)
        p_rgt = _get_landmark_point(landmarks, right, img_w, img_h)
        vertical = np.linalg.norm(p_top - p_bot)
        horizontal = np.linalg.norm(p_lft - p_rgt)
        return vertical / (horizontal + 1e-6)

    left_ear = _ear(_LEFT_EYE_TOP, _LEFT_EYE_BOTTOM, _LEFT_EYE_LEFT, _LEFT_EYE_RIGHT)
    right_ear = _ear(_RIGHT_EYE_TOP, _RIGHT_EYE_BOTTOM, _RIGHT_EYE_LEFT, _RIGHT_EYE_RIGHT)
    return (left_ear + right_ear) / 2.0


def compute_eyebrow_raise(landmarks, img_w: int, img_h: int) -> dict:
    """Detect eyebrow raise by measuring distance from brow to eye."""
    left_brow_top = _get_landmark_point(landmarks, _LEFT_EYEBROW_TOP, img_w, img_h)
    left_brow_bot = _get_landmark_point(landmarks, _LEFT_EYEBROW_BOTTOM, img_w, img_h)
    right_brow_top = _get_landmark_point(landmarks, _RIGHT_EYEBROW_TOP, img_w, img_h)
    right_brow_bot = _get_landmark_point(landmarks, _RIGHT_EYEBROW_BOTTOM, img_w, img_h)

    left_dist = np.linalg.norm(left_brow_top - left_brow_bot)
    right_dist = np.linalg.norm(right_brow_top - right_brow_bot)

    left_eye = _get_landmark_point(landmarks, _LEFT_EYE_OUTER, img_w, img_h)
    right_eye = _get_landmark_point(landmarks, _RIGHT_EYE_OUTER, img_w, img_h)
    inter_eye = np.linalg.norm(left_eye - right_eye) + 1e-6

    return {
        "left_normalized": float(left_dist / inter_eye),
        "right_normalized": float(right_dist / inter_eye),
    }


def compute_smile_ratio(landmarks, img_w: int, img_h: int) -> float:
    """Detect smile by mouth width-to-height ratio."""
    m_left = _get_landmark_point(landmarks, _MOUTH_LEFT, img_w, img_h)
    m_right = _get_landmark_point(landmarks, _MOUTH_RIGHT, img_w, img_h)
    m_top = _get_landmark_point(landmarks, _MOUTH_TOP, img_w, img_h)
    m_bottom = _get_landmark_point(landmarks, _MOUTH_BOTTOM, img_w, img_h)

    width = np.linalg.norm(m_left - m_right)
    height = np.linalg.norm(m_top - m_bottom) + 1e-6
    return float(width / height)


class ActiveChallengeDetector:
    """Verify that the user's face matches a given challenge prompt."""

    YAW_THRESHOLD = 15.0
    PITCH_THRESHOLD = 15.0
    EAR_BLINK_THRESHOLD = 0.18
    EYEBROW_RAISE_THRESHOLD = 0.28
    SMILE_RATIO_THRESHOLD = 3.0

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH):
        self._landmarker = _create_face_landmarker(model_path)
        self.blink_count = 0
        self._prev_ear = 1.0
        self._blink_state = False

    def process_frame(self, frame: np.ndarray, challenge_type: str) -> dict:
        img_h, img_w = frame.shape[:2]
        landmarks = _detect_landmarks(self._landmarker, frame)

        if landmarks is None:
            return {"verified": False, "score": 0.0, "details": {"error": "No face detected"}}

        pitch, yaw, roll = get_head_pose(landmarks, img_w, img_h)

        verified = False
        score = 0.0
        details = {"pitch": pitch, "yaw": yaw, "roll": roll}

        challenge = challenge_type.upper()

        if challenge == "LEFT":
            score = min(1.0, max(0.0, (-yaw - self.YAW_THRESHOLD) / 15.0)) if yaw < -self.YAW_THRESHOLD else 0.0
            verified = yaw < -self.YAW_THRESHOLD
        elif challenge == "RIGHT":
            score = min(1.0, max(0.0, (yaw - self.YAW_THRESHOLD) / 15.0)) if yaw > self.YAW_THRESHOLD else 0.0
            verified = yaw > self.YAW_THRESHOLD
        elif challenge == "UP":
            score = min(1.0, max(0.0, (pitch - self.PITCH_THRESHOLD) / 15.0)) if pitch > self.PITCH_THRESHOLD else 0.0
            verified = pitch > self.PITCH_THRESHOLD
        elif challenge == "DOWN":
            score = min(1.0, max(0.0, (-pitch - self.PITCH_THRESHOLD) / 15.0)) if pitch < -self.PITCH_THRESHOLD else 0.0
            verified = pitch < -self.PITCH_THRESHOLD
        elif challenge == "BLINK_TWICE":
            ear = compute_eye_aspect_ratio(landmarks, img_w, img_h)
            details["ear"] = ear
            if self._prev_ear > self.EAR_BLINK_THRESHOLD and ear < self.EAR_BLINK_THRESHOLD:
                if not self._blink_state:
                    self.blink_count += 1
                    self._blink_state = True
            if ear > self.EAR_BLINK_THRESHOLD:
                self._blink_state = False
            self._prev_ear = ear
            details["blink_count"] = self.blink_count
            score = min(1.0, self.blink_count / 2.0)
            verified = self.blink_count >= 2
        elif challenge == "RAISE_RIGHT_EYEBROW":
            brow_data = compute_eyebrow_raise(landmarks, img_w, img_h)
            details.update(brow_data)
            right_raised = brow_data["right_normalized"] > self.EYEBROW_RAISE_THRESHOLD
            left_neutral = brow_data["left_normalized"] < self.EYEBROW_RAISE_THRESHOLD
            verified = right_raised and left_neutral
            score = 1.0 if verified else brow_data["right_normalized"] / self.EYEBROW_RAISE_THRESHOLD
        elif challenge == "SMILE_NATURALLY":
            smile_ratio = compute_smile_ratio(landmarks, img_w, img_h)
            details["smile_ratio"] = smile_ratio
            verified = smile_ratio > self.SMILE_RATIO_THRESHOLD
            score = min(1.0, smile_ratio / self.SMILE_RATIO_THRESHOLD)

        if verified:
            score = max(score, 1.0)

        return {"verified": verified, "score": float(score), "details": details}

    def close(self):
        self._landmarker.close()


# ══════════════════════════════════════════════════════════════
#  LAYER 2: Optical Flow Depth Analysis (Replay Detection)
# ══════════════════════════════════════════════════════════════

class OpticalFlowAnalyzer:
    """
    Detect replay attacks by analyzing optical flow.
    A real face exhibits non-uniform, parallax-consistent motion.
    A replayed video (screen/photo) shows uniform, planar flow.
    """

    def __init__(self, history_size: int = 30):
        self._prev_gray: Optional[np.ndarray] = None
        self._flow_magnitudes = deque(maxlen=history_size)
        self._flow_variances = deque(maxlen=history_size)

    def process_frame(self, frame: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return {"is_live": False, "score": 0.0, "details": {"status": "initializing"}}

        prev_pts = cv2.goodFeaturesToTrack(
            self._prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
        )

        if prev_pts is None or len(prev_pts) < 10:
            self._prev_gray = gray
            return {"is_live": False, "score": 0.0, "details": {"status": "insufficient_features"}}

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, prev_pts, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        good_prev = prev_pts[status.flatten() == 1]
        good_next = next_pts[status.flatten() == 1]

        if len(good_prev) < 5:
            self._prev_gray = gray
            return {"is_live": False, "score": 0.0, "details": {"status": "tracking_lost"}}

        flow_vectors = good_next - good_prev
        magnitudes = np.linalg.norm(flow_vectors, axis=1)
        mean_mag = float(np.mean(magnitudes))
        var_mag = float(np.var(magnitudes))
        angles = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0])
        angle_var = float(np.var(angles))

        self._flow_magnitudes.append(mean_mag)
        self._flow_variances.append(var_mag)

        has_motion = mean_mag > 0.3
        has_depth_cues = var_mag > 0.5
        has_direction_diversity = angle_var > 0.3
        temporal_var = float(np.var(list(self._flow_magnitudes))) if len(self._flow_magnitudes) > 5 else 0.0
        has_temporal_variation = temporal_var > 0.1

        live_indicators = sum([has_motion, has_depth_cues, has_direction_diversity, has_temporal_variation])
        score = live_indicators / 4.0
        is_live = score >= 0.5

        self._prev_gray = gray

        return {
            "is_live": is_live,
            "score": float(score),
            "details": {
                "mean_magnitude": mean_mag,
                "magnitude_variance": var_mag,
                "angle_variance": angle_var,
                "temporal_variance": temporal_var,
            },
        }

    def reset(self):
        self._prev_gray = None
        self._flow_magnitudes.clear()
        self._flow_variances.clear()


# ══════════════════════════════════════════════════════════════
#  LAYER 3: rPPG — Remote Photoplethysmography
# ══════════════════════════════════════════════════════════════

class RPPGDetector:
    """
    Detect biological liveness by extracting heart rate from subtle
    RGB variations in the forehead region.
    """

    LOW_FREQ = 0.75
    HIGH_FREQ = 2.0
    MIN_HR = 50
    MAX_HR = 120
    BUFFER_SECONDS = 6
    TARGET_FPS = 30

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH):
        self._landmarker = _create_face_landmarker(model_path)
        buffer_size = self.BUFFER_SECONDS * self.TARGET_FPS
        self._green_signal = deque(maxlen=buffer_size)
        self._timestamps = deque(maxlen=buffer_size)
        self._last_hr = 0.0

    def _get_forehead_roi(self, landmarks, img_w: int, img_h: int) -> Optional[np.ndarray]:
        points = []
        for idx in _FOREHEAD_POINTS:
            lm = landmarks[idx]
            points.append([int(lm.x * img_w), int(lm.y * img_h)])
        return np.array(points, dtype=np.int32)

    def process_frame(self, frame: np.ndarray) -> dict:
        img_h, img_w = frame.shape[:2]
        landmarks = _detect_landmarks(self._landmarker, frame)

        if landmarks is None:
            return {"has_pulse": False, "heart_rate": 0.0, "score": 0.0, "details": {"error": "No face"}}

        roi_points = self._get_forehead_roi(landmarks, img_w, img_h)

        if roi_points is None:
            return {"has_pulse": False, "heart_rate": 0.0, "score": 0.0, "details": {"error": "No ROI"}}

        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, roi_points, 255)
        green_channel = frame[:, :, 1]
        roi_pixels = green_channel[mask == 255]

        if len(roi_pixels) < 50:
            return {"has_pulse": False, "heart_rate": 0.0, "score": 0.0, "details": {"error": "ROI too small"}}

        mean_green = float(np.mean(roi_pixels))
        self._green_signal.append(mean_green)
        self._timestamps.append(time.time())

        min_samples = self.TARGET_FPS * 3
        if len(self._green_signal) < min_samples:
            return {
                "has_pulse": False,
                "heart_rate": 0.0,
                "score": 0.0,
                "details": {"status": "buffering", "samples": len(self._green_signal)},
            }

        timestamps = np.array(list(self._timestamps))
        duration = timestamps[-1] - timestamps[0]
        actual_fps = len(timestamps) / duration if duration > 0 else self.TARGET_FPS

        raw = np.array(list(self._green_signal), dtype=np.float64)
        detrended = signal.detrend(raw)
        normalized = (detrended - np.mean(detrended)) / (np.std(detrended) + 1e-8)

        nyquist = actual_fps / 2.0
        low = self.LOW_FREQ / nyquist
        high = min(self.HIGH_FREQ / nyquist, 0.99)

        if low >= high or low <= 0:
            return {"has_pulse": False, "heart_rate": 0.0, "score": 0.0, "details": {"error": "Bad FPS"}}

        b, a = signal.butter(3, [low, high], btype="band")
        filtered = signal.filtfilt(b, a, normalized)

        n = len(filtered)
        fft_vals = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(n, d=1.0 / actual_fps)

        valid_mask = (freqs >= self.LOW_FREQ) & (freqs <= self.HIGH_FREQ)
        if not np.any(valid_mask):
            return {"has_pulse": False, "heart_rate": 0.0, "score": 0.0, "details": {"error": "No valid freq"}}

        valid_fft = fft_vals[valid_mask]
        valid_freqs = freqs[valid_mask]

        peak_idx = np.argmax(valid_fft)
        peak_freq = valid_freqs[peak_idx]
        heart_rate = peak_freq * 60.0

        peak_power = valid_fft[peak_idx] ** 2
        mean_power = np.mean(valid_fft ** 2)
        snr = peak_power / (mean_power + 1e-8)

        hr_valid = self.MIN_HR <= heart_rate <= self.MAX_HR
        signal_strong = snr > 3.0
        has_pulse = hr_valid and signal_strong

        hr_score = 1.0 if hr_valid else 0.0
        snr_score = min(1.0, snr / 10.0)
        score = 0.6 * hr_score + 0.4 * snr_score

        self._last_hr = heart_rate

        return {
            "has_pulse": has_pulse,
            "heart_rate": float(heart_rate),
            "score": float(score),
            "details": {
                "snr": float(snr),
                "peak_frequency": float(peak_freq),
                "actual_fps": float(actual_fps),
                "buffer_size": len(self._green_signal),
            },
        }

    def reset(self):
        self._green_signal.clear()
        self._timestamps.clear()
        self._last_hr = 0.0

    def close(self):
        self._landmarker.close()


# ══════════════════════════════════════════════════════════════
#  Combined Liveness Score
# ══════════════════════════════════════════════════════════════

class LivenessEngine:
    """
    Orchestrates all three liveness detection layers
    and produces a unified liveness score.
    """

    ACTIVE_WEIGHT = 0.50
    OPTICAL_WEIGHT = 0.25
    RPPG_WEIGHT = 0.25

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH):
        self.active_detector = ActiveChallengeDetector(model_path)
        self.optical_flow = OpticalFlowAnalyzer()
        self.rppg = RPPGDetector(model_path)

    def process_frame(self, frame: np.ndarray, challenge_type: str) -> dict:
        active_result = self.active_detector.process_frame(frame, challenge_type)
        flow_result = self.optical_flow.process_frame(frame)
        rppg_result = self.rppg.process_frame(frame)

        combined_score = (
            self.ACTIVE_WEIGHT * active_result["score"]
            + self.OPTICAL_WEIGHT * flow_result["score"]
            + self.RPPG_WEIGHT * rppg_result["score"]
        )

        return {
            "liveness_score": float(combined_score),
            "challenge_verified": active_result["verified"],
            "layers": {
                "active": active_result,
                "optical_flow": flow_result,
                "rppg": rppg_result,
            },
        }

    def reset(self):
        self.active_detector.blink_count = 0
        self.optical_flow.reset()
        self.rppg.reset()

    def close(self):
        self.active_detector.close()
        self.rppg.close()