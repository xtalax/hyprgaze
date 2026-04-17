"""MediaPipe Tasks FaceLandmarker based gaze estimation.

Produces a (yaw, pitch) estimate in radians by combining:
  1. Head pose via cv2.solvePnP on 6 anchor landmarks against a canonical 3D face.
  2. Iris-in-eye offset (eye-in-head), scaled to an angular contribution.

Sign convention:
  yaw   > 0 → user looking to their right (camera's left)
  pitch > 0 → user looking up
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions


# Cached model file. Downloaded on first use.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = Path.home() / ".cache" / "hyprgaze" / "face_landmarker.task"


# --- Canonical 3D face model (mm, origin at nose tip). ---
# Y is DOWN (matches OpenCV camera frame convention). Z is out-of-face (+Z
# points toward the camera when the user is looking at it).
#
# Using Y-up here is tempting but wrong: combined with OpenCV's Y-down camera
# frame, the face-at-camera pose would require a reflection (det=-1),
# which solvePnP (restricted to proper rotations) can't return. Y-down
# makes face-at-camera a clean Rot_Y(π).
#
# Correspondence note: MediaPipe LM 33 is subject's RIGHT eye outer corner
# (appears on image's LEFT side when the subject faces the camera), and
# LM 263 is subject's LEFT. Same for mouth: LM 61 = subject's right
# mouth corner, LM 291 = subject's left. Index order below pairs each
# LM with its correct-side 3D point.
_ANCHOR_IDX = [1, 152, 263, 33, 291, 61]
_ANCHOR_3D = np.array(
    [
        [  0.0,   0.0,   0.0],   # LM 1   — nose tip
        [  0.0,  63.6, -12.5],   # LM 152 — chin  (below nose → +Y)
        [-43.3, -32.7, -26.0],   # LM 263 — subject's LEFT eye outer (−X, above)
        [ 43.3, -32.7, -26.0],   # LM 33  — subject's RIGHT eye outer (+X)
        [-28.9,  28.9, -24.1],   # LM 291 — subject's LEFT mouth corner
        [ 28.9,  28.9, -24.1],   # LM 61  — subject's RIGHT mouth corner
    ],
    dtype=np.float64,
)

# Eye / iris landmark indices. The Tasks FaceLandmarker model emits 478 points
# including iris landmarks (468-477).
_L_EYE_OUT, _L_EYE_IN = 33, 133
_R_EYE_IN, _R_EYE_OUT = 362, 263
_L_EYE_TOP, _L_EYE_BOT = 159, 145
_R_EYE_TOP, _R_EYE_BOT = 386, 374
_L_IRIS, _R_IRIS = 468, 473


def _ensure_model() -> str:
    if not _MODEL_PATH.exists():
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face landmarker model to {_MODEL_PATH}...", flush=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return str(_MODEL_PATH)


@dataclass
class GazeSample:
    yaw: float          # total (head + eye), radians, + = user's right
    pitch: float        # + = up
    head_yaw: float
    head_pitch: float
    iris_x: float       # average normalized iris offset, image coords (+ = image-right)
    iris_y: float


class GazeTracker:
    def __init__(
        self,
        iris_yaw_gain: float = 0.35,
        iris_pitch_gain: float = 0.20,
        iris_x_sign: float = -1.0,
        iris_y_sign: float = -1.0,
    ):
        self.iris_yaw_gain = iris_yaw_gain
        self.iris_pitch_gain = iris_pitch_gain
        self.iris_x_sign = iris_x_sign
        self.iris_y_sign = iris_y_sign

        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure_model()),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        self._cam_mat: np.ndarray | None = None
        self._cam_mat_for: tuple[int, int] | None = None
        self._last_ts_ms = 0

    def config(self) -> dict:
        """Serializable tracker parameters (for saving with a calibration)."""
        return dict(
            iris_yaw_gain=self.iris_yaw_gain,
            iris_pitch_gain=self.iris_pitch_gain,
            iris_x_sign=self.iris_x_sign,
            iris_y_sign=self.iris_y_sign,
        )

    def _camera_matrix(self, w: int, h: int) -> np.ndarray:
        if self._cam_mat_for != (w, h):
            # Assume horizontal FOV ~60° (C920-class).
            f = w / (2 * np.tan(np.deg2rad(60) / 2))
            self._cam_mat = np.array(
                [[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]],
                dtype=np.float64,
            )
            self._cam_mat_for = (w, h)
        return self._cam_mat  # type: ignore[return-value]

    def process(self, frame_bgr: np.ndarray, t_seconds: float) -> GazeSample | None:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Task API requires monotonically increasing millisecond timestamps.
        ts_ms = int(t_seconds * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        res = self.landmarker.detect_for_video(mp_image, ts_ms)
        if not res.face_landmarks:
            return None
        lm = res.face_landmarks[0]

        def pt(i: int) -> tuple[float, float]:
            return (lm[i].x * w, lm[i].y * h)

        # --- Head pose via solvePnP. ---
        # SQPNP is stable for this near-planar 6-point config; ITERATIVE
        # flips to the mirror solution ~30% of the time under landmark noise,
        # which is what was producing baseline angles near ±π.
        img_pts = np.array([pt(i) for i in _ANCHOR_IDX], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            _ANCHOR_3D,
            img_pts,
            self._camera_matrix(w, h),
            np.zeros(4),
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok:
            return None
        # Safety net: if the solver still placed the face behind the camera
        # (tvec[2] < 0), we've hit the mirror solution. Flip it.
        if float(tvec[2, 0]) < 0:
            rvec = -rvec

        R, _ = cv2.Rodrigues(rvec)
        # Object-space forward = +Z (out of face). Camera frame: +X right,
        # +Y down, +Z into scene. For a face looking at the camera, fwd in
        # camera frame ≈ [0, 0, -1] (pointing toward camera).
        fwd = R @ np.array([0.0, 0.0, 1.0])
        head_yaw = float(np.arctan2(fwd[0], -fwd[2]))    # + = user's right
        head_pitch = float(np.arctan2(-fwd[1], -fwd[2]))  # + = user looks up

        # --- Iris-in-eye offset. ---
        def iris_offset(out_i, in_i, top_i, bot_i, iris_i):
            ox, oy = pt(out_i)
            ix, iy = pt(in_i)
            _, ty = pt(top_i)
            _, by = pt(bot_i)
            irx, iry = pt(iris_i)
            cx = (ox + ix) / 2
            cy = (ty + by) / 2
            ew = float(np.hypot(ix - ox, iy - oy))
            eh = abs(by - ty)
            if ew < 2 or eh < 2:
                return 0.0, 0.0
            return (irx - cx) / (ew / 2), (iry - cy) / (eh / 2)

        l_dx, l_dy = iris_offset(_L_EYE_OUT, _L_EYE_IN, _L_EYE_TOP, _L_EYE_BOT, _L_IRIS)
        r_dx, r_dy = iris_offset(_R_EYE_IN, _R_EYE_OUT, _R_EYE_TOP, _R_EYE_BOT, _R_IRIS)
        iris_x = (l_dx + r_dx) / 2
        iris_y = (l_dy + r_dy) / 2

        iris_yaw = self.iris_x_sign * self.iris_yaw_gain * iris_x
        iris_pitch = self.iris_y_sign * self.iris_pitch_gain * iris_y

        return GazeSample(
            yaw=head_yaw + iris_yaw,
            pitch=head_pitch + iris_pitch,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            iris_x=iris_x,
            iris_y=iris_y,
        )
