"""Unit tests for the device-independent IMU fusion (no hardware needed)."""
import numpy as np

from hyprgaze.imu import ComplementaryFilter, ImuAxes


# Gravity steady on the +Y (up) axis → neutral pitch (forward axis ⟂ gravity).
GRAVITY_UP = np.array([0.0, 1.0, 0.0])


def _run(f, gyro, acc, n, dt=0.01):
    yaw = pitch = 0.0
    for _ in range(n):
        yaw, pitch = f.update(np.asarray(gyro, float), np.asarray(acc, float), dt)
    return yaw, pitch


def test_yaw_integrates_gyro():
    f = ComplementaryFilter()
    yaw, pitch = _run(f, [0.0, 0.5, 0.0], GRAVITY_UP, 100, dt=0.01)  # 0.5 rad/s, 1 s
    assert abs(yaw - 0.5) < 1e-6
    assert abs(pitch) < 1e-6


def test_pitch_seeds_from_gravity_on_first_sample():
    f = ComplementaryFilter()
    # acc up=cos θ, forward=sin θ  →  accel pitch = atan2(fwd, up) = θ
    theta = 0.3
    acc = [0.0, np.cos(theta), np.sin(theta)]
    yaw, pitch = f.update(np.zeros(3), np.asarray(acc, float), 0.01)
    assert abs(pitch - theta) < 1e-9      # seeded immediately, no settle
    assert abs(yaw) < 1e-12


def test_pitch_tracks_gravity_with_zero_gyro():
    f = ComplementaryFilter()
    theta = -0.4
    acc = [0.0, np.cos(theta), np.sin(theta)]
    _, pitch = _run(f, [0.0, 0.0, 0.0], acc, 200, dt=0.01)
    assert abs(pitch - theta) < 1e-3


def test_recenter_zeros_current_pose():
    f = ComplementaryFilter()
    _run(f, [0.0, 0.3, 0.0], GRAVITY_UP, 100)   # build up some yaw
    f.recenter()
    yaw, _ = f.update(np.zeros(3), GRAVITY_UP, 0.01)
    assert abs(yaw) < 1e-9


def test_axis_sign_inverts_yaw():
    f = ComplementaryFilter(ImuAxes(gyro_yaw_sign=-1.0))
    yaw, _ = _run(f, [0.0, 0.5, 0.0], GRAVITY_UP, 100)
    assert abs(yaw + 0.5) < 1e-6           # inverted


def test_zero_dt_is_noop():
    f = ComplementaryFilter()
    _run(f, [0.0, 0.5, 0.0], GRAVITY_UP, 10)
    before = (f.yaw, f.pitch)
    f.update(np.array([9.0, 9.0, 9.0]), GRAVITY_UP, 0.0)
    assert (f.yaw, f.pitch) == before      # dt<=0 must not advance state
