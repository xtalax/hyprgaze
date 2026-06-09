"""niri compositor integration via `niri msg --json`.

niri is a scrollable-tiling compositor that OWNS window layout and does not
expose per-window pixel rectangles over IPC. So gazefocus on niri uses a
different focus model than the Hyprland path (warp.py):

    map gaze → screen point → dwell until it settles → move the cursor there
    (via ydotool) → niri's `focus-follows-mouse` focuses whatever is under it.

No `window_at`, no client polling, no explicit focus call. Requires in
~/.config/niri/config.kdl:

    input {
        focus-follows-mouse max-scroll-amount="0%"   // focus w/o auto-scrolling
    }

Cursor movement uses ydotool (uinput, compositor-agnostic; ydotoold must run).

TODO (verify on the target, can't run niri offline): the exact shape of
`niri msg --json outputs` and ydotool's absolute-coord units. The parser below
follows the documented niri-ipc Output/LogicalOutput schema and is defensive;
recalibrate on niri so the affine absorbs any coordinate-space quirk.
"""
from __future__ import annotations

import json
import math
import os
import subprocess

from .warp import Monitor, ScreenBox, bounding_box  # reuse generic geometry

__all__ = ["detect_compositor", "get_outputs", "move_cursor", "NiriFocuser",
           "ScreenBox", "bounding_box"]


def detect_compositor() -> str:
    """'niri' or 'hyprland' from the session env, defaulting to hyprland."""
    if os.environ.get("NIRI_SOCKET"):
        return "niri"
    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        return "hyprland"
    return "hyprland"


def _niri_json(*args: str):
    try:
        out = subprocess.check_output(["niri", "msg", "--json", *args])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return json.loads(out)


def parse_outputs(data) -> list[Monitor]:
    """Parse `niri msg --json outputs` into Monitors. Pure (unit-tested).

    niri reports a connector→Output map (older builds: a list). Each enabled
    Output has a `logical` block with final logical x/y/width/height/scale —
    already post-transform, so no w/h swap needed (unlike Hyprland).
    """
    if not data:
        return []
    entries = data.values() if isinstance(data, dict) else data
    mons: list[Monitor] = []
    for o in entries:
        logical = o.get("logical")
        if not logical:          # output disabled / not mapped
            continue
        mons.append(
            Monitor(
                name=o.get("name", "?"),
                x=int(logical["x"]),
                y=int(logical["y"]),
                w=int(logical["width"]),
                h=int(logical["height"]),
                scale=float(logical.get("scale", 1.0)) or 1.0,
                transform=0,                 # logical dims already applied it
                active_workspace_id=None,    # niri focus model doesn't need it
            )
        )
    return mons


def get_outputs() -> list[Monitor]:
    return parse_outputs(_niri_json("outputs"))


def move_cursor(x: int, y: int) -> None:
    """Absolute pointer warp via ydotool (uinput). Requires ydotoold running."""
    subprocess.run(
        ["ydotool", "mousemove", "--absolute", "-x", str(int(x)), "-y", str(int(y))],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class NiriFocuser:
    """Decide *when* to warp the cursor, so niri's focus-follows-mouse focuses.

    Policy (no continuous warp — respects gazefocus's interaction model): warp
    only after the gaze point has held still (within `stable_radius_px`) for
    `dwell_sec`, and only if the new target is `move_threshold_px` away from the
    last warp. After a warp, the user must leave and re-dwell to warp again.

    Pure / no IO — unit-tested in tests/test_niri.py.
    """

    def __init__(self, dwell_sec: float = 0.4, stable_radius_px: float = 60.0,
                 move_threshold_px: float = 80.0):
        self.dwell_sec = dwell_sec
        self.stable_radius_px = stable_radius_px
        self.move_threshold_px = move_threshold_px
        self._cand: tuple[float, float] | None = None
        self._cand_since = 0.0
        self._last_warp: tuple[float, float] | None = None
        self._armed = True   # must re-dwell after a warp before warping again

    def update(self, t: float, sx: float, sy: float) -> tuple[int, int] | None:
        """Feed a (filtered) gaze point; return (x, y) to warp to, or None."""
        if self._cand is None or math.hypot(sx - self._cand[0], sy - self._cand[1]) > self.stable_radius_px:
            # Moved off the candidate → start a fresh dwell here, re-arm.
            self._cand = (sx, sy)
            self._cand_since = t
            self._armed = True
            return None

        # Still dwelling near the candidate.
        if not self._armed or (t - self._cand_since) < self.dwell_sec:
            return None

        target = self._cand
        if self._last_warp is not None and \
                math.hypot(target[0] - self._last_warp[0], target[1] - self._last_warp[1]) <= self.move_threshold_px:
            self._armed = False   # too close to last warp; wait for a real move
            return None

        self._last_warp = target
        self._armed = False       # disarm until the user looks away and back
        return int(target[0]), int(target[1])
