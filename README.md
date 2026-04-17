# hyprgaze

Webcam gaze tracking picks a target window, the cursor warps to that
window's center, and Hyprland's `follow_mouse` focuses it. The cursor
only moves on a **focus change** — no continuous jitter.

## How it works

```
webcam ─▶ MediaPipe FaceLandmarker ─▶ head pose + iris offset
                                                │
                                      (yaw, pitch) in radians
                                                │
                         ─▶ subtract shared baseline
                         ─▶ pick nearest calibrated monitor (angular space)
                         ─▶ apply that monitor's affine
                         ─▶ One-Euro filter (low-lag smoothing)
                         ─▶ find window under gaze point (hyprctl)
                         ─▶ dwell N ms on a new window
                         ─▶ warp cursor to window center
```

**Per-monitor affines.** Each monitor is calibrated independently and its
own 2×3 affine is stored. At runtime, the monitor whose calibrated center
is closest (in angular space) to the current gaze wins, and its affine
maps gaze to logical screen coords. This handles monitors in **different
planes** (angled / swiveled) and **different orientations** (Hyprland
`transform=1` / portrait) — the linear relationship between gaze angle
and *logical* screen coord is learned separately for each, so whatever
the compositor's output transform or the monitor's physical angle does
is baked in.

## Install

Requires `uv`, `hyprctl`, and `systemctl` on PATH. MediaPipe only ships
wheels up to Python 3.12, so the project pins 3.12 via `.python-version`.

```sh
./install.sh               # sync venv, install systemd --user service, print next-steps
./install.sh --with-path   # same, plus symlink bin/hyprgaze-* into ~/.local/bin
```

Idempotent. Re-run after pulling updates. The installer also migrates
any legacy `~/.config/gazefocus/` config to `~/.config/hyprgaze/` and
removes any older `gazefocus.service` unit.

After install, do these in order:

1. **Calibrate** (mandatory, ~30 s):
   ```sh
   .venv/bin/hyprgaze calibrate
   ```
2. **Enable the daemon:**
   ```sh
   systemctl --user enable --now hyprgaze.service
   journalctl --user -u hyprgaze -f
   ```
3. **Wire up the waybar widget + Hyprland keybind** — see sections below.

## Run

```sh
uv run hyprgaze                 # (implicit `run`)
uv run hyprgaze --debug         # preview window, shows dwell progress bar
uv run hyprgaze --dry-run       # observe only, don't warp the cursor
uv run hyprgaze --dwell-ms 600  # longer dwell (default 400 ms)
```

Ctrl+C to stop. In `--debug` you can also press `q` in the preview window.

Under the systemd service, stop/start with `systemctl --user stop/start
hyprgaze.service` (or use the waybar widget / keybind).

## Calibrate

```sh
uv run hyprgaze calibrate                          # 9 points × every monitor (default)
uv run hyprgaze calibrate --points 5               # faster, less robust
uv run hyprgaze calibrate --monitors DP-1          # one monitor only
uv run hyprgaze calibrate --monitors DP-2,DP-1     # explicit order
```

By default, calibration visits every monitor Hyprland reports (origin
monitor first) and captures a 3×3 grid of targets per monitor. The
window relocates between monitors automatically; a brief "MONITOR N of M"
banner gives your gaze time to follow.

A fullscreen window shows red targets, auto-captures ~1.5 s per target —
no keypresses. **Face each dot fully. Move your head.** Pure eye-only
movement produces tiny angular excursions that give a badly-conditioned
affine fit (symptom: cursor ends up at some screen edge regardless of
where you actually look).

Per-monitor results print at the end:

```
per-monitor fits:
  DP-1: fit=   85 px, cond= 7.2, center angle (Δ=+0.0°, +0.0°) → (1920, 1080)
  DP-2: fit=  120 px, cond= 8.5, center angle (Δ=+24.3°, -5.1°) → (5760, 0)
```

- **fit** — mean residual in logical pixels. Under ~200 px is good on 4K.
- **cond** — condition number. Over 50 means your angular excursion was too
  small on that monitor; re-run and actually face the dots.
- **center angle (Δ)** — how much you turned (yaw, pitch) to look at that
  monitor's center relative to your neutral pose. For the first monitor
  this is (0, 0) by construction; for others it's what the monitor's
  physical position *and* orientation require. These values drive the
  nearest-monitor selection at runtime.

## Re-zero the head pose

Calibration captures your head-at-neutral as a baseline. If you shift
posture mid-session (slouch, move chair), gaze estimates drift. Quick fix:

```sh
uv run hyprgaze zero
```

1.5 s of "stare at center, don't move" — updates just the baseline in
your existing calibration. Takes ~3 s total.

### Useful flags (`run`)

| flag | default | meaning |
|---|---|---|
| `--camera N` | 0 | `/dev/videoN` |
| `--dwell-ms MS` | 400 | gaze must stay on a new window this long before it gets focused |
| `--debug` | off | preview window with gaze + dwell bar |
| `--dry-run` | off | compute everything but don't warp the cursor |
| `--no-calibration` | off | ignore saved calibration (fall back to linear) |
| `--yaw-range-deg D` | 15 | uncalibrated only: ±D° → full screen width |
| `--pitch-range-deg D` | 10 | uncalibrated only: ±D° → full screen height |
| `--iris-gain G` | 0.6 | uncalibrated only |
| `--flip-iris-x/y` | off | uncalibrated only |

## Known limitations

- **Head pose dominates gaze estimate.** Eye-in-head contribution is
  smaller (~1/3 by default). This is deliberate for stability — eye-only
  gaze from a webcam is very noisy at screen distances.
- **Monitor selection is nearest-center.** If two monitors' calibrated
  centers are close in angular space (e.g. stacked with small vertical
  spread), selection can flip at the boundary. Focus-dwell absorbs some
  of that, but unusual layouts may want smarter selection later.
- **Existing manual cursor use competes with us.** If you move the mouse
  yourself, it will fight gaze-driven focus. No pause keybind yet —
  `pkill -f hyprgaze` is the current answer.

## Waybar widget

An eye icon in your tray — left-click toggles the daemon on/off,
right-click runs a full recalibration (daemon is stopped while
calibration runs, then restarted if it was running).

`bin/hyprgaze-status` emits a waybar JSON payload with
`class: "active"` / `"inactive"` for CSS styling.

Paths below assume the repo lives at `~/src/hyprgaze`. Substitute your
actual path (or run `./install.sh` — it prints the correct snippets with
your absolute paths filled in).

In `~/.config/waybar/config` (or `config.jsonc`), add `"custom/hyprgaze"`
to your `modules-right` (or wherever), and define the module:

```jsonc
"custom/hyprgaze": {
    "exec": "$HOME/src/hyprgaze/bin/hyprgaze-status",
    "return-type": "json",
    "interval": 2,
    "on-click":       "$HOME/src/hyprgaze/bin/hyprgaze-toggle",
    "on-click-right": "$HOME/src/hyprgaze/bin/hyprgaze-recalibrate"
}
```

Optional styling in `~/.config/waybar/style.css`:

```css
#custom-hyprgaze.active   { color: #a6e3a1; }
#custom-hyprgaze.inactive { color: #6c7086; }
```

Reload waybar with `pkill -SIGUSR2 waybar`.

## Hyprland keybind

```conf
# ~/.config/hypr/hyprland.conf
bind = SUPER ALT, F, exec, $HOME/src/hyprgaze/bin/hyprgaze-toggle
```

Reload with `hyprctl reload`.

## Systemd service environment

The `hyprgaze` daemon runs as a user service and needs access to
Hyprland's session environment (`HYPRLAND_INSTANCE_SIGNATURE`, etc.)
so `hyprctl` can talk to the compositor. If you haven't already, add
this to your Hyprland config so the session exports its environment
into user-systemd at login:

```
exec-once = systemctl --user import-environment DISPLAY WAYLAND_DISPLAY HYPRLAND_INSTANCE_SIGNATURE XDG_RUNTIME_DIR
```

Most omarchy setups already do this. Check with
`journalctl --user -u hyprgaze` — if the daemon exits immediately with
`hyprctl` errors, this is the cause.
