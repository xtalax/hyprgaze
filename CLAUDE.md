# hyprgaze — Claude notes

Webcam gaze tracking → Hyprland window focus. Python daemon, no compiled
plugin. MediaPipe FaceLandmarker for landmarks + head pose, per-monitor
affine to map gaze angles to logical screen coords, dwell-based focus
(cursor only moves when gaze stably lands on a *new* window).

## Data flow

```
VideoCapture ─▶ GazeTracker.process(frame, t)
                    │  • MediaPipe FaceLandmarker (VIDEO mode, .task model)
                    │  • solvePnP (SQPNP!) against _ANCHOR_3D → head yaw/pitch
                    │  • iris-in-eye offset → eye-in-head correction
                    └─▶ GazeSample(yaw, pitch, head_yaw, head_pitch, iris_x, iris_y)

Calibration.apply(yaw, pitch)
    │  • angle_diff subtract baseline  (handles ±π wrap)
    │  • pick nearest monitor by angular distance to its center
    │  • apply that monitor's 2×3 affine  →  logical (sx, sy)

(sx, sy) ─▶ OneEuroFilter per axis ─▶ window_at(sx, sy, clients, monitors)
                                         │  (filters by monitor's activeWorkspace)
                                         └─▶ dwell gate ─▶ hyprctl movecursor
```

## Module map

- `src/hyprgaze/tracker.py` — MediaPipe + solvePnP + iris math.
- `src/hyprgaze/calibration.py` — `Calibration`/`MonitorCalibration`, fullscreen UI,
  `run_calibration`, `run_zero`. Stores to `~/.config/hyprgaze/calibration.json`.
- `src/hyprgaze/warp.py` — all Hyprland IPC: `get_monitors`, `get_clients`,
  `get_active_window`, `window_at`, `move_cursor`. The runtime path never
  computes monitor geometry of its own — see gotcha #5.
- `src/hyprgaze/filter.py` — One-Euro filter.
- `src/hyprgaze/__main__.py` — CLI (`run` / `calibrate` / `zero`) + focus-dwell loop.
- `bin/hyprgaze-{toggle,status,recalibrate}` — waybar / keybind adapters
  (shell wrappers around `systemctl --user`).
- `systemd/hyprgaze.service.template` — user service; `install.sh`
  substitutes `__REPO__` with the absolute repo path.
- `install.sh` — idempotent installer. Regenerates service unit, migrates
  legacy `~/.config/gazefocus/` config if present.

## Critical gotchas (hard-won)

These aren't obvious from the code. Each one cost at least one recalibration
cycle to discover.

### 1. Face model must use Y-down, not Y-up

OpenCV's camera frame is Y-down. If the canonical face model uses Y-up,
then face-at-camera requires flipping X, Y, and Z — det=−1, a *reflection*,
which `solvePnP` (proper rotations only) cannot return. Symptom: head
yaw/pitch cluster near ±π instead of 0 at neutral pose. `_ANCHOR_3D` in
`tracker.py` has chin at `+63.6` and eyes at `-32.7` for this reason.

### 2. Use `SOLVEPNP_SQPNP`, not `SOLVEPNP_ITERATIVE`

ITERATIVE has a mirror ambiguity for near-planar 6-point configs — it
converges on "face behind camera, upside-down" ~30 % of the time under
realistic landmark noise (verified via simulation). SQPNP never flips.
Also keep the `if tvec[2, 0] < 0: rvec = -rvec` safety check after
solvePnP.

### 3. MediaPipe LM 33 is subject's RIGHT eye

Looking at the canonical face diagram, LM 33 is on the *viewer's* left,
which is the *subject's* right eye (image is not mirrored). Same for
LM 61 (subject's right mouth corner). The `_ANCHOR_IDX` order is
`[1, 152, 263, 33, 291, 61]` to pair each landmark with its correct-side
3D point. Swapping indices (as I did originally) produces near-±π
baselines even with SQPNP.

### 4. `_angle_diff` is mandatory, not defensive

The baseline head pose can legitimately be far from zero (camera is
often off-center from the primary monitor). If baseline is near ±π,
naïve subtraction of a monitor's center yaw wraps through the branch
cut and the affine fit blows up (e.g. `center_yaw` stored as 5.27 rad
in an old `calibration.json`). Always use
`_angle_diff(a, b) = ((a - b + π) mod 2π) - π` when subtracting baseline —
both during the fit and in `Calibration.apply`.

### 5. Runtime path doesn't compute monitor geometry — hyprctl does

`hyprctl monitors -j` reports panel width/height (pre-transform on most
Hyprland versions) plus a `transform` integer (0–7; 1/3/5/7 swap w/h).
Our `Monitor` dataclass applies transform + scale when populating `w/h`,
but **the runtime `window_at` path doesn't use those dims at all**. It
only uses `Monitor.active_workspace_id` (which hyprctl reports directly)
to filter clients to the visible workspaces; containment is then tested
against each client's own absolute `at`/`size` — also from hyprctl.

So the "which monitor is this point on" question never gets asked at
runtime. Windows live in one shared logical coord system and the
compositor is the source of truth for their positions. If a Hyprland
version ever changes how `width`/`height`/`transform`/`scale` interact,
the runtime path is unaffected.

For the **calibration fullscreen window size**, don't trust `Monitor.w/h`
either — it's an initial canvas guess only. Query the window's actual
size via `hyprctl clients -j` right after fullscreen takes effect
(`_query_window_size` in `calibration.py`). That's the authoritative
logical size: post-transform, post-scale, post-Hyprland-gaps.

Don't re-introduce a `_monitor_at(x, y, monitors)` helper. It's a
tempting abstraction ("find the monitor for this point") but it
requires us to compute logical bounds, which is exactly the thing the
compositor should own. The old version caused a real bug: on
`transform=1` monitors it mis-bounded the monitor and `window_at`
returned None for valid points.

### 6. Per-monitor affines, not one global affine

A single global affine fits two monitors in different planes/orientations
badly (large residuals). Each `MonitorCalibration` owns its own 2×3 `A`.
Monitor selection at runtime must happen in **angular space** (gaze
pre-affine), not screen space (chicken-and-egg: you need the right
affine to decide which monitor → which affine to apply).
`Calibration.nearest_monitor(adj_yaw, adj_pitch)` does it by Euclidean
distance to each monitor's stored `center_yaw/pitch`.

### 7. Hyprland `movewindow mon:NAME` needs an un-fullscreen first

To relocate the calibration window between monitors, issue
`fullscreen 0 → movewindow mon:X → fullscreen 1`. Fullscreen windows
resist `movewindow`. `_relocate_fullscreen` in `calibration.py` handles it.

### 8. Logical ≠ physical monitor layout

The user's DP-2 is at Hyprland `x=3840` (logically "right" of DP-1) but
physically on their *left*. The system handles this transparently
because per-monitor affines map gaze angles to logical coords, and
window lookup uses logical coords. Never assume logical x order
corresponds to physical left-to-right.

## Calibration JSON format (v2)

```jsonc
{
  "version": 2,
  "tracker": { iris_yaw_gain, iris_pitch_gain, iris_x_sign, iris_y_sign },
  "baseline_yaw":   <radians>,   // neutral-pose head yaw  (not always ≈ 0)
  "baseline_pitch": <radians>,
  "monitors": [
    {
      "name": "DP-1",
      "A": [[a00,a01,a02], [a10,a11,a12]],          // (2,3), maps [Δyaw, Δpitch, 1] → (x, y)
      "center_yaw": 0.0, "center_pitch": 0.0,       // monitor 0 is baseline anchor
      "mon_x, mon_y, mon_w, mon_h": …,              // logical bounds snapshot
      "fit_error_px": …,
      "cond": …                                     // cond(X); under 30 is good
    },
    ...
  ]
}
```

Old single-`A` formats auto-upgrade into a one-monitor v2 on load
(`Calibration.load` handles it). Users recalibrate to gain multi-monitor
support.

## Hyprland interaction

All through `hyprctl <verb> -j` subprocess calls (cached at 4 Hz in the
main loop to keep overhead low). Not using the IPC socket directly — if
we ever need <25 ms latency on state refreshes, that would be the upgrade.

Subprocesses used:
- `hyprctl monitors -j` — layouts + `transform` + `activeWorkspace`.
- `hyprctl clients -j` — window at/size/workspace/class/title/focusHistoryID.
- `hyprctl activewindow -j` — currently focused.
- `hyprctl dispatch movecursor X Y` — absolute cursor warp.
- `hyprctl dispatch focuswindow / movewindow / fullscreen` — only during
  calibration, for the fullscreen target window.

`window_at` excludes windows whose title contains "hyprgaze" (debug and
calibration windows) so we don't try to focus ourselves.

The runtime refresh cadence is 250 ms (4 Hz). It's not the per-frame hot
path — the per-frame work is MediaPipe + solvePnP + the filter/dwell
checks. Staleness up to 250 ms is imperceptible because focus-dwell
itself requires ≥400 ms.

## Commands I actually use

```sh
./install.sh                          # setup / regenerate systemd unit
./install.sh --with-path              # + symlink bin/* to ~/.local/bin

.venv/bin/hyprgaze calibrate          # 9 points × each monitor (default)
.venv/bin/hyprgaze zero               # re-baseline only (~3s)
.venv/bin/hyprgaze --debug            # live preview + dwell bar
.venv/bin/hyprgaze --debug --dry-run  # observe without warping cursor

systemctl --user enable --now hyprgaze.service
journalctl --user -u hyprgaze -f

cat ~/.config/hyprgaze/calibration.json     # sanity-check a fresh calibration
```

**Sanity check a calibration:** `baseline_yaw/pitch` should be within
±0.5 rad (±30°) of zero (the residual is camera-off-center, not a bug).
`cond` should be < 30 per monitor. `fit_error_px` < 200 is good on 4K.

## Don'ts

- **Don't** revert `SOLVEPNP_SQPNP` to `SOLVEPNP_ITERATIVE` for speed —
  SQPNP is closed-form and fast enough; ITERATIVE is a regression.
- **Don't** compute monitor selection on the predicted `(sx, sy)` —
  you'd need the per-monitor affine to *get* that prediction. Selection
  must be in angular space.
- **Don't** drop the angle unwrap "because baseline should be small" —
  we've seen legitimate large baselines (camera on a different monitor
  than the origin one).
- **Don't** add `hyprctl` polling to the per-frame hot loop without a
  TTL cache. Subprocess spawn is ~5–10 ms; 30× per second eats the
  whole frame budget.
- **Don't** re-introduce `_monitor_at(x, y)` or any other "which monitor
  owns this point" helper. The runtime path filters clients by active
  workspace and tests each client's own `at`/`size`. Let hyprctl own
  the geometry.
- **Don't** introduce continuous cursor warping. Window-focus-dwell is
  the interaction model; continuous warp was tried and felt awful.
- **Don't** assume the camera is centered on any monitor, or that
  Hyprland's logical monitor order matches physical left-to-right.
  Per-monitor affines + baseline absorb arbitrary camera placement.
- **Don't** edit the user's waybar or Hyprland configs from `install.sh`.
  Print snippets; let the user paste. (Their configs often have complex
  include structures we shouldn't touch.)
