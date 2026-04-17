"""Calibration flow: show targets across monitors, collect gaze, fit per-monitor affines, save.

Each monitor gets its own affine:
    [screen_x, screen_y] = A_mon · [yaw - baseline_yaw, pitch - baseline_pitch, 1]

At runtime, Calibration.apply picks the monitor whose center is nearest (in
angular space) to the current baseline-subtracted gaze, then applies that
monitor's A. This handles monitors in different planes and rotations — each
monitor's own (angle → logical screen coord) relationship is learned
independently.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .tracker import GazeTracker
from .warp import Monitor, get_monitors


def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference (a - b) wrapped to (-π, π].

    Essential when the baseline head pose sits near the atan2 branch cut
    (e.g. the camera is far off-center from a monitor, so 'neutral gaze'
    has large yaw). Without wrapping, adjusted angles for the OTHER
    monitor cross ±π and the fit explodes.
    """
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return float(d)


CALIB_PATH = Path(
    os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
) / "hyprgaze" / "calibration.json"

_WIN = "hyprgaze calibration"


@dataclass
class MonitorCalibration:
    name: str
    A: np.ndarray              # (2, 3), maps (Δyaw, Δpitch, 1) → (x, y) logical
    center_yaw: float          # baseline-subtracted yaw at this monitor's on-screen center
    center_pitch: float
    mon_x: int                 # monitor bounds in logical coords (snapshot at calibration)
    mon_y: int
    mon_w: int
    mon_h: int
    fit_error_px: float
    cond: float


@dataclass
class Calibration:
    monitors: list[MonitorCalibration]
    tracker_config: dict
    baseline_yaw: float = 0.0
    baseline_pitch: float = 0.0

    @property
    def fit_error_px(self) -> float:
        if not self.monitors:
            return 0.0
        return float(np.mean([m.fit_error_px for m in self.monitors]))

    def nearest_monitor(self, adj_yaw: float, adj_pitch: float) -> MonitorCalibration:
        return min(
            self.monitors,
            key=lambda m: (adj_yaw - m.center_yaw) ** 2
            + (adj_pitch - m.center_pitch) ** 2,
        )

    def apply(self, yaw: float, pitch: float) -> tuple[float, float]:
        if not self.monitors:
            return 0.0, 0.0
        a = _angle_diff(yaw, self.baseline_yaw)
        b = _angle_diff(pitch, self.baseline_pitch)
        mc = self.nearest_monitor(a, b)
        v = mc.A @ np.array([a, b, 1.0])
        return float(v[0]), float(v[1])

    def apply_with_monitor(
        self, yaw: float, pitch: float
    ) -> tuple[float, float, MonitorCalibration]:
        a = _angle_diff(yaw, self.baseline_yaw)
        b = _angle_diff(pitch, self.baseline_pitch)
        mc = self.nearest_monitor(a, b)
        v = mc.A @ np.array([a, b, 1.0])
        return float(v[0]), float(v[1]), mc

    def save(self, path: Path = CALIB_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "tracker": self.tracker_config,
                    "baseline_yaw": self.baseline_yaw,
                    "baseline_pitch": self.baseline_pitch,
                    "monitors": [
                        {
                            "name": m.name,
                            "A": m.A.tolist(),
                            "center_yaw": m.center_yaw,
                            "center_pitch": m.center_pitch,
                            "mon_x": m.mon_x,
                            "mon_y": m.mon_y,
                            "mon_w": m.mon_w,
                            "mon_h": m.mon_h,
                            "fit_error_px": m.fit_error_px,
                            "cond": m.cond,
                        }
                        for m in self.monitors
                    ],
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path = CALIB_PATH) -> "Calibration | None":
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        if d.get("version") == 2:
            return cls(
                monitors=[
                    MonitorCalibration(
                        name=m["name"],
                        A=np.array(m["A"], dtype=np.float64),
                        center_yaw=float(m["center_yaw"]),
                        center_pitch=float(m["center_pitch"]),
                        mon_x=int(m["mon_x"]),
                        mon_y=int(m["mon_y"]),
                        mon_w=int(m["mon_w"]),
                        mon_h=int(m["mon_h"]),
                        fit_error_px=float(m.get("fit_error_px", 0.0)),
                        cond=float(m.get("cond", 0.0)),
                    )
                    for m in d["monitors"]
                ],
                tracker_config=d.get("tracker", {}),
                baseline_yaw=float(d.get("baseline_yaw") or 0.0),
                baseline_pitch=float(d.get("baseline_pitch") or 0.0),
            )
        # Auto-upgrade a v0/v1 single-monitor file into a v2 with one entry.
        print(
            f"⚠ {path} is an older single-monitor calibration format — "
            "please re-run `hyprgaze calibrate` for multi-monitor support.",
            flush=True,
        )
        mons_hw = get_monitors()
        if not mons_hw or "A" not in d:
            return None
        mon = mons_hw[0]
        mc = MonitorCalibration(
            name=mon.name,
            A=np.array(d["A"], dtype=np.float64),
            center_yaw=0.0,
            center_pitch=0.0,
            mon_x=mon.x,
            mon_y=mon.y,
            mon_w=int(mon.w / mon.scale),
            mon_h=int(mon.h / mon.scale),
            fit_error_px=float(d.get("fit_error_px", 0.0)),
            cond=0.0,
        )
        return cls(
            monitors=[mc],
            tracker_config=d.get("tracker", {}),
            baseline_yaw=float(d.get("baseline_yaw") or 0.0),
            baseline_pitch=float(d.get("baseline_pitch") or 0.0),
        )


# ------------------------- UI helpers -------------------------

def _put_centered(img, text, cx, cy, scale, color, thick=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(
        img, text, (cx - tw // 2, cy + th // 2),
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA,
    )


def _target(img, x, y, r, ring=(60, 140, 255), dot=(80, 180, 255)):
    cv2.circle(img, (x, y), r + 6, (0, 0, 0), -1)
    cv2.circle(img, (x, y), r, ring, 3, cv2.LINE_AA)
    cv2.circle(img, (x, y), max(3, r // 4), dot, -1, cv2.LINE_AA)


def _blank(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# --------------------- Monitor / window setup ---------------------

def _pick_monitor(mons: list[Monitor], name: str | None) -> Monitor:
    """Single-monitor chooser — used by run_zero."""
    if name:
        for m in mons:
            if m.name == name:
                return m
        print(f"warning: monitor {name!r} not found; falling back to {mons[0].name}")
    for m in mons:
        if m.x == 0 and m.y == 0:
            return m
    return mons[0]


def _resolve_monitors(
    all_mons: list[Monitor], requested: list[str] | None
) -> list[Monitor]:
    if requested:
        out: list[Monitor] = []
        for name in requested:
            m = next((m for m in all_mons if m.name == name), None)
            if m is None:
                print(f"warning: monitor {name!r} not found; skipping.", flush=True)
                continue
            out.append(m)
        return out
    # Default: origin monitor first, then by (x, y).
    return sorted(
        all_mons,
        key=lambda m: (not (m.x == 0 and m.y == 0), m.x, m.y),
    )


def _query_window_size() -> tuple[int, int] | None:
    """Ask Hyprland what size our fullscreen window actually is.

    This is authoritative for rotated monitors — the logical width/height
    after any transform is applied.
    """
    try:
        out = subprocess.check_output(["hyprctl", "clients", "-j"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    for c in json.loads(out):
        if _WIN in c.get("title", ""):
            s = c.get("size") or [0, 0]
            if s[0] > 0 and s[1] > 0:
                return int(s[0]), int(s[1])
    return None


def _relocate_fullscreen(mon: Monitor) -> tuple[int, int]:
    """Open or relocate the calibration window fullscreen on `mon`.

    Returns the actual logical (width, height) of the window after transition.
    For rotated outputs (e.g. Hyprland transform=1), this may differ from
    mon.w/mon.h, so we query hyprctl for the true post-transform size.
    """
    # Rough initial guess (will be corrected below).
    guess_w = int(mon.w / mon.scale)
    guess_h = int(mon.h / mon.scale)
    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
    cv2.imshow(_WIN, _blank(guess_w, guess_h))
    cv2.waitKey(50)
    try:
        subprocess.run(
            ["hyprctl", "dispatch", "fullscreen", "0"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["hyprctl", "dispatch", "focuswindow", f"title:^({_WIN})$"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["hyprctl", "dispatch", "movewindow", f"mon:{mon.name}"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["hyprctl", "dispatch", "fullscreen", "1"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass
    time.sleep(0.35)
    cv2.setWindowProperty(_WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # Query actual size — authoritative for rotated monitors.
    actual = _query_window_size()
    if actual is not None:
        return actual
    return guess_w, guess_h


def _teardown(cam):
    cam.release()
    try:
        subprocess.run(
            ["hyprctl", "dispatch", "fullscreen", "0"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass
    cv2.destroyAllWindows()
    cv2.waitKey(1)


# ----------------------- Core capture -----------------------

def _positions(w: int, h: int, margin: float, n: int) -> list[tuple[int, int]]:
    mx, my = int(w * margin), int(h * margin)
    if n == 5:
        return [
            (w // 2, h // 2),     # center first — provides baseline on monitor 0
            (mx, h // 2),
            (w - mx, h // 2),
            (w // 2, my),
            (w // 2, h - my),
        ]
    if n == 9:
        xs = [mx, w // 2, w - mx]
        ys = [my, h // 2, h - my]
        return [(x, y) for y in ys for x in xs]
    raise ValueError(f"points must be 5 or 9, got {n}")


def _capture_point(
    cam,
    tracker: GazeTracker,
    w: int,
    h: int,
    px: int,
    py: int,
    header: str,
    get_ready_sec: float,
    capture_sec: float,
    flash_sec: float,
) -> list[tuple[float, float]] | None:
    # Phase 1: get ready (pulsing small dot).
    t0 = time.monotonic()
    while time.monotonic() - t0 < get_ready_sec:
        cam.read()
        canvas = _blank(w, h)
        _put_centered(canvas, header, w // 2, 36, 0.8, (130, 130, 130), 1)
        _put_centered(canvas, "get ready…", w // 2, h - 40, 0.7, (100, 100, 100), 1)
        pulse = 0.5 + 0.5 * float(np.sin((time.monotonic() - t0) * 6))
        r = int(10 + pulse * 10)
        _target(canvas, px, py, r, (60, 90, 140), (60, 90, 140))
        cv2.imshow(_WIN, canvas)
        if (cv2.waitKey(20) & 0xFF) == ord("q"):
            return None

    # Phase 2: capture.
    samples: list[tuple[float, float]] = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < capture_sec:
        ok, frame = cam.read()
        if not ok:
            continue
        s = tracker.process(frame, time.monotonic())
        if s is not None:
            samples.append((s.yaw, s.pitch))
        canvas = _blank(w, h)
        _put_centered(canvas, header, w // 2, 36, 0.8, (130, 130, 130), 1)
        progress = min(1.0, (time.monotonic() - t0) / capture_sec)
        _target(canvas, px, py, 36, (80, 200, 255), (80, 220, 255))
        cv2.ellipse(
            canvas, (px, py), (52, 52), -90, 0,
            int(360 * progress), (0, 255, 120), 4, cv2.LINE_AA,
        )
        _put_centered(canvas, f"captured {len(samples)}", w // 2, h - 40,
                      0.7, (0, 220, 120), 1)
        cv2.imshow(_WIN, canvas)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            return None

    # Phase 3: flash.
    canvas = _blank(w, h)
    _target(canvas, px, py, 36, (0, 255, 120), (0, 255, 120))
    _put_centered(canvas, f"{len(samples)} samples",
                  w // 2, h // 2 + 80, 1.0, (0, 255, 120))
    cv2.imshow(_WIN, canvas)
    cv2.waitKey(int(flash_sec * 1000))
    return samples


def _wait_with_message(
    cam,
    w: int,
    h: int,
    draw: Callable[[np.ndarray, float], None],
    duration: float,
) -> bool:
    t0 = time.monotonic()
    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= duration:
            return True
        cam.read()
        canvas = _blank(w, h)
        draw(canvas, elapsed)
        cv2.imshow(_WIN, canvas)
        if (cv2.waitKey(20) & 0xFF) == ord("q"):
            return False


def _robust_median(samples: list[tuple[float, float]]) -> tuple[float, float]:
    arr = np.asarray(samples, dtype=np.float64)
    lo, hi = np.percentile(arr, [10, 90], axis=0)
    mask = np.all((arr >= lo) & (arr <= hi), axis=1)
    kept = arr[mask] if int(mask.sum()) >= 3 else arr
    return float(np.median(kept[:, 0])), float(np.median(kept[:, 1]))


# ----------------------- Main entry: calibrate -----------------------

def run_calibration(
    camera_index: int = 0,
    monitor_names: list[str] | None = None,
    n_points: int = 9,
    margin: float = 0.1,
    get_ready_sec: float = 1.2,
    capture_sec: float = 1.5,
    flash_sec: float = 0.25,
    banner_sec: float = 1.2,
    welcome_sec: float = 3.0,
    iris_gain: float = 0.6,
    flip_iris_x: bool = False,
    flip_iris_y: bool = False,
) -> Calibration | None:
    mons = _resolve_monitors(get_monitors(), monitor_names)
    if not mons:
        print("no valid monitors; aborting.")
        return None

    print(
        f"calibrating across {len(mons)} monitor(s): "
        f"{', '.join(m.name for m in mons)}",
        flush=True,
    )

    cam = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)
    if not cam.isOpened():
        print(f"Failed to open camera {camera_index}.")
        return None

    tracker = GazeTracker(
        iris_yaw_gain=iris_gain * float(np.deg2rad(20)),
        iris_pitch_gain=iris_gain * float(np.deg2rad(15)),
        iris_x_sign=1.0 if flip_iris_x else -1.0,
        iris_y_sign=1.0 if flip_iris_y else -1.0,
    )

    w, h = _relocate_fullscreen(mons[0])
    mon_list_text = " → ".join(m.name for m in mons)

    def _draw_welcome(canvas, elapsed):
        remaining = max(0, welcome_sec - elapsed)
        _put_centered(canvas, "GAZE CALIBRATION",
                      w // 2, h // 2 - 120, 2.0, (230, 230, 230))
        _put_centered(canvas, "Targets appear on each monitor, in order:",
                      w // 2, h // 2 - 55, 0.8, (180, 220, 255), 1)
        _put_centered(canvas, mon_list_text,
                      w // 2, h // 2 - 15, 0.9, (200, 240, 255), 2)
        _put_centered(canvas, "Face each dot fully — move your head AND eyes.",
                      w // 2, h // 2 + 30, 0.8, (180, 180, 180), 1)
        _put_centered(canvas, f"starting in {int(np.ceil(remaining))}...",
                      w // 2, h // 2 + 80, 1.0, (120, 180, 255))
        _put_centered(canvas, "Q to abort", w // 2, h - 40, 0.6, (80, 80, 80), 1)

    if not _wait_with_message(cam, w, h, _draw_welcome, welcome_sec):
        _teardown(cam)
        return None

    # Collect per-monitor.
    by_monitor: list[tuple[Monitor, int, int, list[tuple[tuple[int, int], list[tuple[float, float]]]]]] = []

    for mon_idx, mon in enumerate(mons):
        if mon_idx > 0:
            w, h = _relocate_fullscreen(mon)

        banner_text = f"MONITOR {mon_idx + 1} of {len(mons)}: {mon.name}"
        print(f"\n→ {banner_text} ({w}x{h} @ {mon.x},{mon.y})", flush=True)

        def _draw_banner(canvas, _, bt=banner_text, _w=w, _h=h):
            _put_centered(canvas, bt, _w // 2, _h // 2, 1.4, (180, 220, 255))

        if not _wait_with_message(cam, w, h, _draw_banner, banner_sec):
            _teardown(cam)
            return None

        positions = _positions(w, h, margin, n_points)
        mon_points: list[tuple[tuple[int, int], list[tuple[float, float]]]] = []
        for i, (px, py) in enumerate(positions):
            header = f"{mon.name}  —  point {i + 1}/{len(positions)}"
            samples = _capture_point(
                cam, tracker, w, h, px, py, header,
                get_ready_sec, capture_sec, flash_sec,
            )
            if samples is None:
                _teardown(cam)
                return None
            mon_points.append(((mon.x + px, mon.y + py), samples))
        by_monitor.append((mon, w, h, mon_points))

    _teardown(cam)

    # --- Baseline from first monitor's first (center) point ---
    _, _, _, first_points = by_monitor[0]
    if not first_points or len(first_points[0][1]) < 5:
        print("no usable samples on first monitor's center point; aborting.")
        return None
    baseline_yaw, baseline_pitch = _robust_median(first_points[0][1])
    print(
        "\nbaseline head pose (raw, at first monitor's center):\n"
        f"  yaw {np.rad2deg(baseline_yaw):+7.2f}°  "
        f"pitch {np.rad2deg(baseline_pitch):+7.2f}°"
    )
    if abs(baseline_yaw) > np.deg2rad(60) or abs(baseline_pitch) > np.deg2rad(60):
        print(
            "  ⚠ baseline > ±60° from camera axis — either the camera is "
            "very off-center from this monitor, or head pose math is off."
        )

    # --- Per-monitor fits ---
    mon_cals: list[MonitorCalibration] = []
    print("\nper-monitor fits:")
    for mon, mw, mh, points in by_monitor:
        X_rows: list[list[float]] = []
        Y_rows: list[list[float]] = []
        center_adj: tuple[float, float] | None = None
        for i, ((abs_x, abs_y), samples) in enumerate(points):
            if len(samples) < 5:
                print(f"  skipping point {i} on {mon.name} (only {len(samples)} samples)")
                continue
            y_m, p_m = _robust_median(samples)
            adj_y = _angle_diff(y_m, baseline_yaw)
            adj_p = _angle_diff(p_m, baseline_pitch)
            if i == 0:
                center_adj = (adj_y, adj_p)
            X_rows.append([adj_y, adj_p, 1.0])
            Y_rows.append([float(abs_x), float(abs_y)])

        if len(X_rows) < 3 or center_adj is None:
            print(f"  skipping {mon.name}: not enough usable points")
            continue

        X = np.array(X_rows)
        Y = np.array(Y_rows)
        M, *_ = np.linalg.lstsq(X, Y, rcond=None)    # (3, 2)
        A = M.T                                      # (2, 3)
        residuals = np.linalg.norm(Y - (X @ M), axis=1)
        fit_err = float(residuals.mean())
        cond = float(np.linalg.cond(X))

        mon_cals.append(
            MonitorCalibration(
                name=mon.name,
                A=A,
                center_yaw=center_adj[0],
                center_pitch=center_adj[1],
                mon_x=mon.x,
                mon_y=mon.y,
                mon_w=mw,
                mon_h=mh,
                fit_error_px=fit_err,
                cond=cond,
            )
        )
        center_map = A @ np.array([center_adj[0], center_adj[1], 1.0])    # this mon's center
        print(
            f"  {mon.name}: fit={fit_err:6.0f} px, cond={cond:5.1f}, "
            f"center angle (Δ={np.rad2deg(center_adj[0]):+.1f}°, "
            f"{np.rad2deg(center_adj[1]):+.1f}°) → "
            f"({center_map[0]:.0f}, {center_map[1]:.0f})"
        )
        if cond > 50:
            print(f"    ⚠ cond > 50; move your head more when re-running.")
        if fit_err > 400:
            print(f"    ⚠ residual > 400 px.")

    if not mon_cals:
        print("no valid monitor calibrations; aborting.")
        return None

    cal = Calibration(
        monitors=mon_cals,
        tracker_config=tracker.config(),
        baseline_yaw=baseline_yaw,
        baseline_pitch=baseline_pitch,
    )
    cal.save()
    print(f"\nsaved to {CALIB_PATH}")
    return cal


# ----------------------- Main entry: zero -----------------------

def run_zero(
    camera_index: int = 0,
    monitor_name: str | None = None,
    capture_sec: float = 1.5,
    get_ready_sec: float = 1.5,
) -> Calibration | None:
    """Quick head-pose re-baseline: stare at one screen's center, update baseline only.

    Per-monitor affines are unaffected; updating the shared baseline shifts
    inputs uniformly, and each affine translates that into a corresponding
    screen-space shift. Only effective for drifts that are uniform across
    gaze directions (chair moved, new posture).
    """
    cal = Calibration.load()
    if cal is None:
        print("no calibration found — run `hyprgaze calibrate` first.")
        return None

    mons = get_monitors()
    mon = _pick_monitor(mons, monitor_name)

    cam = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)
    if not cam.isOpened():
        print(f"Failed to open camera {camera_index}.")
        return None

    tracker = GazeTracker(**cal.tracker_config)
    w, h = _relocate_fullscreen(mon)

    def _draw_ready(canvas, elapsed):
        _put_centered(canvas, "RE-ZERO HEAD POSE",
                      w // 2, h // 2 - 120, 1.4, (230, 230, 230))
        _put_centered(canvas, "Stare at the center dot, head neutral.",
                      w // 2, h // 2 - 70, 0.8, (180, 180, 180), 1)
        pulse = 0.5 + 0.5 * float(np.sin(elapsed * 6))
        r = int(10 + pulse * 10)
        _target(canvas, w // 2, h // 2, r, (60, 90, 140), (60, 90, 140))

    if not _wait_with_message(cam, w, h, _draw_ready, get_ready_sec):
        _teardown(cam)
        return None

    samples: list[tuple[float, float]] = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < capture_sec:
        ok, frame = cam.read()
        if not ok:
            continue
        s = tracker.process(frame, time.monotonic())
        if s is not None:
            samples.append((s.yaw, s.pitch))
        canvas = _blank(w, h)
        _put_centered(canvas, "capturing…", w // 2, h // 2 - 120, 1.0, (0, 220, 120))
        progress = min(1.0, (time.monotonic() - t0) / capture_sec)
        _target(canvas, w // 2, h // 2, 36, (80, 200, 255), (80, 220, 255))
        cv2.ellipse(
            canvas, (w // 2, h // 2), (52, 52), -90, 0,
            int(360 * progress), (0, 255, 120), 4, cv2.LINE_AA,
        )
        _put_centered(canvas, f"captured {len(samples)}", w // 2, h - 40,
                      0.7, (0, 220, 120), 1)
        cv2.imshow(_WIN, canvas)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            _teardown(cam)
            return None

    _teardown(cam)

    if len(samples) < 5:
        print(f"Too few samples ({len(samples)}). Aborting.")
        return None

    new_yaw, new_pitch = _robust_median(samples)
    old_yaw, old_pitch = cal.baseline_yaw, cal.baseline_pitch
    cal.baseline_yaw = new_yaw
    cal.baseline_pitch = new_pitch
    cal.save()

    print(
        f"baseline updated: "
        f"yaw {np.rad2deg(old_yaw):+.2f}° → {np.rad2deg(new_yaw):+.2f}°, "
        f"pitch {np.rad2deg(old_pitch):+.2f}° → {np.rad2deg(new_pitch):+.2f}°"
    )
    return cal
