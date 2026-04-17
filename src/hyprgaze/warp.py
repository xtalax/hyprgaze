"""Hyprland integration: pure hyprctl IPC, no geometry of our own at runtime.

Design principle: Hyprland is the source of truth for window/monitor
geometry. We never compute "which monitor does this point belong to"
from logical bounds ourselves — instead we filter clients by active
workspace (which hyprctl tells us) and check the client's own
absolute `at`/`size` against the point.

The only place we still need logical monitor dimensions is as an
initial canvas guess for the calibration fullscreen window, and even
that gets immediately overridden by an authoritative size query
(`_query_window_size` in calibration.py). So the transform/scale math
below is best-effort metadata on the Monitor dataclass, not a hot-path
calculation.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Monitor:
    name: str
    x: int                         # logical top-left
    y: int
    w: int                         # logical width (transform/scale applied)
    h: int                         # logical height
    scale: float
    transform: int = 0             # wl_output transform (1/3/5/7 swap w/h)
    active_workspace_id: int | None = None


@dataclass(frozen=True)
class ScreenBox:
    """Union bounding box over all monitors, in Hyprland logical coords."""
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int: return self.x1 - self.x0
    @property
    def h(self) -> int: return self.y1 - self.y0
    @property
    def cx(self) -> float: return (self.x0 + self.x1) / 2
    @property
    def cy(self) -> float: return (self.y0 + self.y1) / 2


# ---------- hyprctl calls ----------

def _hyprctl_json(*args: str):
    """Call hyprctl with -j and parse the JSON reply. Returns None on error."""
    try:
        out = subprocess.check_output(["hyprctl", *args, "-j"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return json.loads(out)


def get_monitors() -> list[Monitor]:
    data = _hyprctl_json("monitors") or []
    mons: list[Monitor] = []
    for m in data:
        transform = int(m.get("transform", 0))
        scale = float(m.get("scale", 1.0)) or 1.0
        w_panel = int(m["width"])
        h_panel = int(m["height"])
        # wl_output transforms 1/3/5/7 are 90°/270° rotations (with
        # optional flip); the logical footprint swaps w/h.
        if transform in (1, 3, 5, 7):
            w_panel, h_panel = h_panel, w_panel
        w_logical = int(w_panel / scale)
        h_logical = int(h_panel / scale)
        ws = m.get("activeWorkspace") or {}
        active_ws = ws.get("id")
        mons.append(
            Monitor(
                name=m["name"],
                x=int(m["x"]),
                y=int(m["y"]),
                w=w_logical,
                h=h_logical,
                scale=scale,
                transform=transform,
                active_workspace_id=active_ws,
            )
        )
    return mons


def get_clients() -> list[dict]:
    return _hyprctl_json("clients") or []


def get_active_window() -> dict | None:
    data = _hyprctl_json("activewindow")
    return data if data else None


def move_cursor(x: int, y: int) -> None:
    subprocess.run(
        ["hyprctl", "dispatch", "movecursor", str(x), str(y)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------- geometry (only uses hyprctl-reported window bounds) ----------

def bounding_box(mons: list[Monitor]) -> ScreenBox:
    """Union of all monitors' logical rectangles.

    Used only for debug/info printing; the runtime path doesn't depend
    on this, so any Hyprland quirk in per-monitor logical-size reporting
    at worst produces a slightly-off info line.
    """
    if not mons:
        return ScreenBox(0, 0, 0, 0)
    xs0 = [m.x for m in mons]
    ys0 = [m.y for m in mons]
    xs1 = [m.x + m.w for m in mons]
    ys1 = [m.y + m.h for m in mons]
    return ScreenBox(min(xs0), min(ys0), max(xs1), max(ys1))


def window_at(
    x: float,
    y: float,
    clients: list[dict],
    monitors: list[Monitor],
    exclude_titles: tuple[str, ...] = (),
) -> dict | None:
    """Topmost *visible* window containing (x, y) in logical coords.

    Visible = `mapped`, not `hidden`, and on an active workspace on
    SOME monitor. We don't care which monitor the point is technically
    "on" — the client's own `at`/`size` test handles that. Two monitors
    only overlap in logical space under mirroring, which this still
    treats sensibly (overlapping windows are tie-broken by focus history).
    """
    active_ws = {m.active_workspace_id for m in monitors}
    active_ws.discard(None)

    hits: list[dict] = []
    for c in clients:
        if not c.get("mapped", True) or c.get("hidden", False):
            continue
        if c.get("workspace", {}).get("id") not in active_ws:
            continue
        title = c.get("title", "")
        if any(t in title for t in exclude_titles):
            continue
        cx, cy = c.get("at", (0, 0))
        cw, ch = c.get("size", (0, 0))
        if cx <= x < cx + cw and cy <= y < cy + ch:
            hits.append(c)
    if not hits:
        return None
    # Prefer most-recently-focused (lowest focusHistoryID).
    hits.sort(key=lambda c: c.get("focusHistoryID", 1 << 30))
    return hits[0]


def window_center(w: dict) -> tuple[int, int]:
    cx, cy = w["at"]
    cw, ch = w["size"]
    return int(cx + cw / 2), int(cy + ch / 2)
