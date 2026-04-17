"""Calibration math: angle unwrap, nearest-monitor selection, round-trip I/O,
legacy-format upgrade.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from hyprgaze.calibration import (
    Calibration,
    MonitorCalibration,
    _angle_diff,
    _robust_median,
)


# ----------------------- _angle_diff -----------------------

def test_angle_diff_small_values():
    assert _angle_diff(0.1, 0.0) == pytest.approx(0.1)
    assert _angle_diff(0.0, 0.1) == pytest.approx(-0.1)


def test_angle_diff_zero():
    assert _angle_diff(1.23, 1.23) == pytest.approx(0.0)


def test_angle_diff_wraps_across_pi():
    # If baseline sits near +π and raw sits near −π, the *signed short path*
    # is small — unwrap must produce that, not a 2π jump.
    a = -math.pi + 0.1
    b = math.pi - 0.1
    assert _angle_diff(a, b) == pytest.approx(0.2)
    # Symmetric case.
    assert _angle_diff(b, a) == pytest.approx(-0.2)


def test_angle_diff_output_range_is_pi_bounded():
    """For any two angles, the unwrapped difference is in (-π, π]."""
    import random
    random.seed(0)
    for _ in range(1000):
        a = random.uniform(-10, 10)
        b = random.uniform(-10, 10)
        d = _angle_diff(a, b)
        assert -math.pi < d <= math.pi


# ----------------------- _robust_median -----------------------

def test_robust_median_center_of_cluster():
    samples = [(0.1, 0.2)] * 10 + [(10.0, -10.0)]  # one huge outlier
    yaw, pitch = _robust_median(samples)
    assert yaw == pytest.approx(0.1, abs=0.01)
    assert pitch == pytest.approx(0.2, abs=0.01)


def test_robust_median_handles_tiny_set():
    samples = [(0.1, 0.0), (0.2, 0.1), (0.3, 0.2)]
    yaw, pitch = _robust_median(samples)
    assert yaw == pytest.approx(0.2, abs=0.01)
    assert pitch == pytest.approx(0.1, abs=0.01)


# --------------------- Calibration.apply ---------------------

def _mc(name, A, cx, cy, mon_x=0, mon_y=0, mon_w=3840, mon_h=2160):
    return MonitorCalibration(
        name=name, A=np.array(A, dtype=np.float64),
        center_yaw=cx, center_pitch=cy,
        mon_x=mon_x, mon_y=mon_y, mon_w=mon_w, mon_h=mon_h,
        fit_error_px=0.0, cond=1.0,
    )


def test_apply_picks_nearest_monitor_by_angle():
    A1 = [[1000.0, 0, 1920], [0, 1000.0, 1080]]
    A2 = [[1000.0, 0, 5760], [0, 1000.0, 0]]
    cal = Calibration(
        monitors=[
            _mc("M1", A1, cx=0.0, cy=0.0),
            _mc("M2", A2, cx=0.5, cy=-0.1, mon_x=3840, mon_y=-1080),
        ],
        tracker_config={},
        baseline_yaw=0.1, baseline_pitch=0.05,
    )
    # Neutral gaze (at baseline) → M1 (its center is at adj=0).
    x, y, mc = cal.apply_with_monitor(0.1, 0.05)
    assert mc.name == "M1"
    assert (x, y) == pytest.approx((1920, 1080))

    # Gaze shifted toward M2's center angular offset → M2.
    x, y, mc = cal.apply_with_monitor(0.1 + 0.5, 0.05 - 0.1)
    assert mc.name == "M2"


def test_apply_uses_angle_unwrap():
    """Baseline near −π: raw near +π should produce a small Δ, not a 2π jump."""
    A = [[1000.0, 0, 1920], [0, 1000.0, 1080]]
    cal = Calibration(
        monitors=[_mc("M1", A, 0.0, 0.0)],
        tracker_config={},
        baseline_yaw=-math.pi + 0.05,
        baseline_pitch=0.0,
    )
    # Raw at +π - 0.05: short path is −0.1, not +2π-0.1.
    x, y = cal.apply(math.pi - 0.05, 0.0)
    # adj_yaw ≈ -0.1; x = A[0,0]*-0.1 + A[0,2] = -100 + 1920 = 1820
    assert x == pytest.approx(1820, abs=1)
    assert y == pytest.approx(1080, abs=1)


# ----------------------- save / load -----------------------

def test_roundtrip_v2(tmp_path: Path):
    A1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    A2 = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    cal = Calibration(
        monitors=[
            _mc("DP-1", A1, 0.0, 0.0),
            _mc("DP-2", A2, 0.3, -0.2, mon_x=3840, mon_y=-1080, mon_w=2160, mon_h=3840),
        ],
        tracker_config={"iris_yaw_gain": 0.21, "iris_x_sign": -1.0},
        baseline_yaw=0.15, baseline_pitch=-0.08,
    )
    p = tmp_path / "cal.json"
    cal.save(p)
    loaded = Calibration.load(p)
    assert loaded is not None
    assert loaded.baseline_yaw == pytest.approx(0.15)
    assert loaded.baseline_pitch == pytest.approx(-0.08)
    assert [m.name for m in loaded.monitors] == ["DP-1", "DP-2"]
    assert np.allclose(loaded.monitors[0].A, A1)
    assert loaded.tracker_config["iris_yaw_gain"] == pytest.approx(0.21)


def test_load_missing_returns_none(tmp_path: Path):
    assert Calibration.load(tmp_path / "absent.json") is None


def test_load_upgrades_legacy_single_A_format(tmp_path: Path, capsys, monkeypatch):
    """A v0/v1 file (no `monitors` list, single `A`) must auto-upgrade to a
    one-monitor v2, using the first live monitor as the placeholder."""
    # Stub get_monitors so the upgrade path has something to work with.
    from hyprgaze import calibration, warp
    fake_mon = warp.Monitor(
        name="DP-1", x=0, y=0, w=3840, h=2160, scale=1.0, transform=0,
        active_workspace_id=1,
    )
    monkeypatch.setattr(calibration, "get_monitors", lambda: [fake_mon])

    legacy = {
        "A": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "tracker": {"iris_yaw_gain": 0.21},
        "baseline_yaw": 0.0,
        "baseline_pitch": 0.0,
    }
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(legacy))
    cal = Calibration.load(p)
    assert cal is not None
    assert len(cal.monitors) == 1
    assert cal.monitors[0].name == "DP-1"
    assert np.allclose(cal.monitors[0].A, legacy["A"])
    # A warning is printed.
    out = capsys.readouterr().out
    assert "older single-monitor" in out or "single-monitor" in out


def test_apply_empty_monitors_returns_zero():
    cal = Calibration(monitors=[], tracker_config={})
    assert cal.apply(0.0, 0.0) == (0.0, 0.0)
