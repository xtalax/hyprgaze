"""hyprgaze entrypoint — window-focus dwell loop + CLI."""
from __future__ import annotations

import argparse
import signal
import sys
import time

import cv2
import numpy as np

from .calibration import CALIB_PATH, Calibration, run_calibration, run_zero
from .filter import OneEuroFilter
from .tracker import GazeTracker
from .warp import (
    Monitor,
    ScreenBox,
    bounding_box,
    get_active_window,
    get_clients,
    get_monitors,
    move_cursor,
    window_at,
    window_center,
)


# ------------------------------ run ------------------------------

def _gaze_to_screen(
    cal: Calibration | None,
    box: ScreenBox,
    yaw_range_rad: float,
    pitch_range_rad: float,
):
    """Return a callable (yaw, pitch) → (sx, sy) in absolute logical coords."""
    if cal is not None:
        return lambda yaw, pitch: cal.apply(yaw, pitch)

    yaw_scale = (box.w / 2) / yaw_range_rad
    pitch_scale = (box.h / 2) / pitch_range_rad
    cx, cy = box.cx, box.cy
    return lambda yaw, pitch: (cx + yaw_scale * yaw, cy - pitch_scale * pitch)


def _cmd_run(args: argparse.Namespace) -> int:
    mons = get_monitors()
    if not mons:
        print("No monitors reported by hyprctl.", file=sys.stderr)
        return 1
    box = bounding_box(mons)
    print(
        f"Screen bounding box: x {box.x0}..{box.x1} ({box.w}px),"
        f" y {box.y0}..{box.y1} ({box.h}px)",
        flush=True,
    )

    cal = Calibration.load() if not args.no_calibration else None
    if cal is not None:
        print(
            f"Using calibration from {CALIB_PATH} "
            f"(fit ~{cal.fit_error_px:.0f}px mean error, "
            f"baseline yaw {np.rad2deg(cal.baseline_yaw):+.1f}° "
            f"pitch {np.rad2deg(cal.baseline_pitch):+.1f}°)",
            flush=True,
        )
        tracker_cfg = dict(cal.tracker_config)
    else:
        print("No calibration — using linear default mapping.", flush=True)
        tracker_cfg = dict(
            iris_yaw_gain=args.iris_gain * float(np.deg2rad(20)),
            iris_pitch_gain=args.iris_gain * float(np.deg2rad(15)),
            iris_x_sign=1.0 if args.flip_iris_x else -1.0,
            iris_y_sign=1.0 if args.flip_iris_y else -1.0,
        )

    map_to_screen = _gaze_to_screen(
        cal,
        box,
        float(np.deg2rad(args.yaw_range_deg)),
        float(np.deg2rad(args.pitch_range_deg)),
    )

    cam = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)
    if not cam.isOpened():
        print(f"Failed to open camera index {args.camera}.", file=sys.stderr)
        return 1

    tracker = GazeTracker(**tracker_cfg)
    fx = OneEuroFilter(min_cutoff=1.0, beta=0.02)
    fy = OneEuroFilter(min_cutoff=1.0, beta=0.02)

    running = True

    def on_sig(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    debug_win: str | None = None
    if args.debug:
        debug_win = "hyprgaze"
        cv2.namedWindow(debug_win, cv2.WINDOW_NORMAL)

    dwell_sec = args.dwell_ms / 1000.0
    state_refresh_sec = 0.25

    # Cached hyprctl state.
    last_state_t = 0.0
    clients: list[dict] = []
    monitors_live: list[Monitor] = []
    active_addr: str | None = None

    # Dwell tracker.
    candidate_addr: str | None = None
    candidate_since = 0.0
    candidate_class = ""

    print(
        f"Running (focus-dwell mode, {args.dwell_ms}ms). "
        f"Ctrl+C (or 'q' in debug window) to stop.",
        flush=True,
    )

    sx_i = sy_i = 0
    stat_t = time.monotonic()
    stat_frames = 0
    stat_faces = 0
    stat_warps = 0
    while running:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.01)
            continue

        t = time.monotonic()
        stat_frames += 1
        sample = tracker.process(frame, t)

        win_under: dict | None = None

        if sample is not None:
            stat_faces += 1
            sx, sy = map_to_screen(sample.yaw, sample.pitch)
            sx_f = fx(t, sx)
            sy_f = fy(t, sy)
            sx_i = int(sx_f)
            sy_i = int(sy_f)

            # Refresh hyprctl state periodically.
            if t - last_state_t > state_refresh_sec:
                try:
                    monitors_live = get_monitors()
                    clients = get_clients()
                    aw = get_active_window()
                    active_addr = aw["address"] if aw else None
                    last_state_t = t
                except Exception:
                    pass  # keep stale data

            win_under = window_at(
                sx_f, sy_f, clients, monitors_live,
                exclude_titles=("hyprgaze",),
            )

            if win_under is None or win_under.get("address") == active_addr:
                candidate_addr = None
                candidate_class = ""
            else:
                addr = win_under.get("address")
                if addr != candidate_addr:
                    candidate_addr = addr
                    candidate_since = t
                    candidate_class = win_under.get("class", "")
                elif t - candidate_since >= dwell_sec:
                    cx, cy = window_center(win_under)
                    if not args.dry_run:
                        move_cursor(cx, cy)
                    stat_warps += 1
                    # Assume our warp will take effect; suppress re-dwell until
                    # the state refresh confirms the new active.
                    active_addr = addr
                    candidate_addr = None
                    candidate_class = ""

        if debug_win is not None:
            _draw_debug(
                frame, sample, sx_i, sy_i, box,
                win_under, candidate_class,
                max(0.0, t - candidate_since) if candidate_addr else 0.0,
                dwell_sec,
            )
            cv2.imshow(debug_win, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if t - stat_t >= 2.0:
            dt = t - stat_t
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"{stat_frames/dt:4.1f} fps, "
                f"face {100*stat_faces/max(1,stat_frames):3.0f}%, "
                f"focus-changes {stat_warps}"
                + ("  (dry-run)" if args.dry_run else ""),
                flush=True,
            )
            stat_t = t
            stat_frames = stat_faces = stat_warps = 0

    cam.release()
    if debug_win is not None:
        cv2.destroyAllWindows()
    return 0


def _draw_debug(
    frame, sample, sx_i, sy_i, box,
    win_under, candidate_class, dwell_elapsed, dwell_sec,
) -> None:
    w = frame.shape[1]
    if sample is not None:
        lines = [
            f"yaw  {np.rad2deg(sample.yaw):+6.1f}  (head {np.rad2deg(sample.head_yaw):+5.1f})",
            f"pitch{np.rad2deg(sample.pitch):+6.1f}  (head {np.rad2deg(sample.head_pitch):+5.1f})",
            f"iris  x={sample.iris_x:+.2f}  y={sample.iris_y:+.2f}",
            f"screen @ {sx_i}, {sy_i}",
        ]
        if win_under is not None:
            klass = win_under.get("class", "?")
            title = win_under.get("title", "")[:40]
            lines.append(f"under: {klass}  \"{title}\"")
        else:
            lines.append("under: -")
        if candidate_class:
            pct = min(1.0, dwell_elapsed / max(0.001, dwell_sec))
            bar = "█" * int(pct * 20) + "·" * (20 - int(pct * 20))
            lines.append(f"dwell: {candidate_class}  [{bar}] {int(pct*100)}%")
        for i, ln in enumerate(lines):
            cv2.putText(frame, ln, (10, 26 + 24 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        rw, rh = 220, int(220 * box.h / max(1, box.w))
        rx0, ry0 = w - rw - 10, 10
        cv2.rectangle(frame, (rx0, ry0), (rx0 + rw, ry0 + rh), (120, 120, 120), 1)
        fx_ = (sx_i - box.x0) / max(1, box.w)
        fy_ = (sy_i - box.y0) / max(1, box.h)
        cv2.circle(frame, (int(rx0 + fx_ * rw), int(ry0 + fy_ * rh)),
                   4, (0, 255, 255), -1)
    else:
        cv2.putText(frame, "no face", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


# --------------------------- calibrate ---------------------------

def _cmd_calibrate(args: argparse.Namespace) -> int:
    monitor_names: list[str] | None = None
    if args.monitors:
        monitor_names = [n.strip() for n in args.monitors.split(",") if n.strip()]
    cal = run_calibration(
        camera_index=args.camera,
        monitor_names=monitor_names,
        n_points=args.points,
        margin=args.margin,
        iris_gain=args.iris_gain,
        flip_iris_x=args.flip_iris_x,
        flip_iris_y=args.flip_iris_y,
    )
    return 0 if cal is not None else 1


def _cmd_zero(args: argparse.Namespace) -> int:
    cal = run_zero(
        camera_index=args.camera,
        monitor_name=args.monitor,
    )
    return 0 if cal is not None else 1


# ------------------------------ CLI ------------------------------

def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--debug", action="store_true",
                   help="show webcam preview with gaze + dwell overlay")
    p.add_argument("--dry-run", action="store_true",
                   help="don't actually move the cursor (observe only)")
    p.add_argument("--dwell-ms", type=int, default=400,
                   help="gaze must stay on a new window this long before focusing")
    p.add_argument("--no-calibration", action="store_true",
                   help="ignore ~/.config/hyprgaze/calibration.json")
    # Fallback linear-mapping args (only used if no calibration).
    p.add_argument("--yaw-range-deg", type=float, default=15.0)
    p.add_argument("--pitch-range-deg", type=float, default=10.0)
    p.add_argument("--iris-gain", type=float, default=0.6)
    p.add_argument("--flip-iris-x", action="store_true")
    p.add_argument("--flip-iris-y", action="store_true")


def _add_calibrate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--monitors", type=str, default=None,
                   help="comma-separated monitor names (default: all, origin first)")
    p.add_argument("--points", type=int, default=9, choices=[5, 9],
                   help="points per monitor (default 9 = 3×3 grid)")
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--iris-gain", type=float, default=0.6)
    p.add_argument("--flip-iris-x", action="store_true")
    p.add_argument("--flip-iris-y", action="store_true")


def _add_zero_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--monitor", type=str, default=None)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or (raw[0].startswith("-") and raw[0] not in ("-h", "--help")):
        raw = ["run"] + raw

    p = argparse.ArgumentParser(prog="hyprgaze")
    sub = p.add_subparsers(dest="cmd", required=True)
    _add_run_args(sub.add_parser("run", help="run gaze → window focus daemon"))
    _add_calibrate_args(sub.add_parser("calibrate", help="full affine calibration"))
    _add_zero_args(sub.add_parser("zero", help="re-baseline head pose only"))

    args = p.parse_args(raw)
    if args.cmd == "calibrate":
        return _cmd_calibrate(args)
    if args.cmd == "zero":
        return _cmd_zero(args)
    return _cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
