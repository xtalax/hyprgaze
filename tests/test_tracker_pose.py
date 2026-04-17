"""Head-pose math regression tests.

We don't test MediaPipe itself — only the `solvePnP + rotation → yaw/pitch`
pipeline that caused the longest debug cycle in this project. Feed
synthetic image points projected from known (R, t), recover the pose,
compare.

Two regressions these tests pin down:
  • SOLVEPNP_ITERATIVE used to converge on the mirror solution ~30 % of
    the time under landmark noise → head angles clustered near ±π.
    Requires SOLVEPNP_SQPNP.
  • Y-up face model + OpenCV Y-down camera frame make face-at-camera
    a reflection (det=−1), which solvePnP can't return. Requires Y-down
    face model (chin at +Y, eyes at −Y).
"""
from __future__ import annotations

import numpy as np
import cv2

from hyprgaze.tracker import _ANCHOR_3D


def _face_at_camera_R():
    """Rotation that takes the canonical face pose to a user facing camera.

    With Y-down face model + OpenCV camera frame, this is Rot_Y(π).
    """
    return np.array([[-1.0, 0, 0], [0, 1, 0], [0, 0, -1]])


def _synth_project(R, t, cam_mat, focal):
    """Project the 6 anchors through (R, t) onto the image plane."""
    pts_cam = (_ANCHOR_3D @ R.T) + t
    cx, cy = cam_mat[0, 2], cam_mat[1, 2]
    img = (pts_cam[:, :2] / pts_cam[:, 2:3]) * focal + np.array([cx, cy])
    return img


def _recover_yaw_pitch(rvec, tvec):
    if float(tvec[2, 0]) < 0:
        rvec = -rvec                     # the explicit mirror check
    R, _ = cv2.Rodrigues(rvec)
    fwd = R @ np.array([0.0, 0.0, 1.0])
    yaw = float(np.arctan2(fwd[0], -fwd[2]))
    pitch = float(np.arctan2(-fwd[1], -fwd[2]))
    return yaw, pitch


def _cam(w=640, h=480, fov_deg=60.0):
    f = w / (2 * np.tan(np.deg2rad(fov_deg) / 2))
    cam = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    return cam, f


def test_face_at_camera_recovers_zero_yaw_pitch_noise_free():
    cam, f = _cam()
    R = _face_at_camera_R()
    t = np.array([0, 0, 600.0])
    img_pts = _synth_project(R, t, cam, f)
    ok, rvec, tvec = cv2.solvePnP(_ANCHOR_3D, img_pts, cam, np.zeros(4),
                                   flags=cv2.SOLVEPNP_SQPNP)
    assert ok
    yaw, pitch = _recover_yaw_pitch(rvec, tvec)
    assert abs(yaw) < np.deg2rad(0.5)
    assert abs(pitch) < np.deg2rad(0.5)


def test_sqpnp_does_not_flip_under_realistic_noise():
    """ITERATIVE flips ~30 % of the time at 1.5 px landmark noise; SQPNP 0 %."""
    cam, f = _cam()
    R = _face_at_camera_R()
    t = np.array([0, 0, 600.0])
    np.random.seed(0)

    yaws, pitches = [], []
    for _ in range(50):
        noise = np.random.randn(6, 2) * 1.5
        img_pts = _synth_project(R, t, cam, f) + noise
        ok, rvec, tvec = cv2.solvePnP(_ANCHOR_3D, img_pts, cam, np.zeros(4),
                                       flags=cv2.SOLVEPNP_SQPNP)
        assert ok
        yaw, pitch = _recover_yaw_pitch(rvec, tvec)
        yaws.append(yaw)
        pitches.append(pitch)
    # The damning failure mode was mean-near-±π. Tight cluster near 0 ⇒ no flip.
    assert abs(np.mean(yaws)) < np.deg2rad(3)
    assert abs(np.mean(pitches)) < np.deg2rad(3)
    # And individual samples don't wander to π territory either.
    assert max(abs(y) for y in yaws) < np.deg2rad(20)


def test_yaw_sign_is_positive_for_user_right_turn():
    """User turns head to their right by ~15°; recovered yaw should be +~15°.

    User-right = rotation about the up axis (which, with Y-down model, is −Y).
    We compose Rot_{−Y}(15°) in FACE coords, then the face-at-camera rotation.
    """
    cam, f = _cam()
    theta = np.deg2rad(15)
    # Rotation about -Y axis (the "up" axis for a Y-down face model)
    R_head = cv2.Rodrigues(np.array([0.0, -theta, 0.0]))[0]
    R_face = _face_at_camera_R() @ R_head
    t = np.array([0, 0, 600.0])
    img_pts = _synth_project(R_face, t, cam, f)

    ok, rvec, tvec = cv2.solvePnP(_ANCHOR_3D, img_pts, cam, np.zeros(4),
                                   flags=cv2.SOLVEPNP_SQPNP)
    yaw, pitch = _recover_yaw_pitch(rvec, tvec)
    # Positive because +yaw = user's right by convention.
    assert yaw == _approx_deg(15, tol_deg=1.0)
    assert abs(pitch) < np.deg2rad(1.5)


def test_pitch_sign_is_positive_for_looking_up():
    cam, f = _cam()
    theta = np.deg2rad(10)
    # Looking up = head tilts back = rotation about +X axis (in Y-down face coords).
    R_head = cv2.Rodrigues(np.array([theta, 0.0, 0.0]))[0]
    R_face = _face_at_camera_R() @ R_head
    t = np.array([0, 0, 600.0])
    img_pts = _synth_project(R_face, t, cam, f)

    ok, rvec, tvec = cv2.solvePnP(_ANCHOR_3D, img_pts, cam, np.zeros(4),
                                   flags=cv2.SOLVEPNP_SQPNP)
    yaw, pitch = _recover_yaw_pitch(rvec, tvec)
    assert pitch == _approx_deg(10, tol_deg=1.5)
    assert abs(yaw) < np.deg2rad(2)


# ------ tiny helper ------

class _approx_deg:
    def __init__(self, deg, tol_deg=1.0):
        self.rad = np.deg2rad(deg)
        self.tol = np.deg2rad(tol_deg)
    def __eq__(self, other):
        return abs(float(other) - self.rad) < self.tol
    def __repr__(self):
        return f"≈{np.rad2deg(self.rad):+.1f}°±{np.rad2deg(self.tol):.1f}°"
