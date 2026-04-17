"""warp.py: parsing hyprctl output, window_at active-workspace filter,
bounding_box with rotated monitors.
"""
from __future__ import annotations

import json
from unittest import mock

from hyprgaze.warp import (
    Monitor,
    ScreenBox,
    bounding_box,
    get_monitors,
    window_at,
    window_center,
)


# ------------------------ get_monitors ------------------------

_DP1 = {
    "name": "DP-1", "x": 0, "y": 0, "width": 3840, "height": 2160,
    "scale": 1.0, "transform": 0,
    "activeWorkspace": {"id": 1, "name": "1"},
}
_DP2_ROT = {
    "name": "DP-2", "x": 3840, "y": -1080,
    "width": 3840, "height": 2160,             # PRE-transform dims
    "scale": 1.0, "transform": 1,              # 90° rotation
    "activeWorkspace": {"id": 11, "name": "11"},
}


def _with_hyprctl(monitors=None, clients=None):
    """Return a fake subprocess.check_output that responds to hyprctl args."""
    def fake(argv, **_):
        if argv[:2] == ["hyprctl", "monitors"]:
            return json.dumps(monitors or []).encode()
        if argv[:2] == ["hyprctl", "clients"]:
            return json.dumps(clients or []).encode()
        if argv[:2] == ["hyprctl", "activewindow"]:
            return b"{}"
        raise AssertionError(f"unexpected hyprctl call: {argv}")
    return fake


def test_get_monitors_applies_transform_swap():
    with mock.patch("subprocess.check_output", side_effect=_with_hyprctl(monitors=[_DP1, _DP2_ROT])):
        mons = get_monitors()
    assert [m.name for m in mons] == ["DP-1", "DP-2"]
    # DP-1: unrotated, logical == panel.
    assert (mons[0].w, mons[0].h) == (3840, 2160)
    # DP-2: transform=1 swaps w/h.
    assert (mons[1].w, mons[1].h) == (2160, 3840)
    assert mons[1].transform == 1
    assert mons[0].active_workspace_id == 1
    assert mons[1].active_workspace_id == 11


def test_get_monitors_applies_scale():
    m = dict(_DP1, scale=1.5)
    with mock.patch("subprocess.check_output", side_effect=_with_hyprctl(monitors=[m])):
        mons = get_monitors()
    # 3840 / 1.5 = 2560
    assert mons[0].w == 2560 and mons[0].h == 1440


def test_get_monitors_handles_missing_fields_gracefully():
    minimal = {"name": "MM-1", "x": 0, "y": 0, "width": 1920, "height": 1080}
    with mock.patch("subprocess.check_output", side_effect=_with_hyprctl(monitors=[minimal])):
        mons = get_monitors()
    assert mons[0].w == 1920 and mons[0].h == 1080
    assert mons[0].scale == 1.0
    assert mons[0].transform == 0
    assert mons[0].active_workspace_id is None


# ------------------------ bounding_box ------------------------

def test_bounding_box_single_monitor():
    box = bounding_box([Monitor("DP-1", 0, 0, 3840, 2160, 1.0)])
    assert box == ScreenBox(0, 0, 3840, 2160)
    assert box.cx == 1920 and box.cy == 1080


def test_bounding_box_rotated_and_offset():
    mons = [
        Monitor("DP-1", 0, 0, 3840, 2160, 1.0, transform=0),
        Monitor("DP-2", 3840, -1080, 2160, 3840, 1.0, transform=1),
    ]
    box = bounding_box(mons)
    assert box == ScreenBox(0, -1080, 6000, 2760)


def test_bounding_box_empty():
    assert bounding_box([]) == ScreenBox(0, 0, 0, 0)


# ------------------------ window_at ------------------------

def _mon(name, x, y, w, h, active_ws):
    return Monitor(name=name, x=x, y=y, w=w, h=h, scale=1.0, transform=0,
                   active_workspace_id=active_ws)


def _client(addr, ws, at, size, *, title="", klass="app", mapped=True,
            hidden=False, fh=99):
    return {
        "address": addr, "workspace": {"id": ws}, "at": list(at),
        "size": list(size), "title": title, "class": klass,
        "mapped": mapped, "hidden": hidden, "focusHistoryID": fh,
    }


def test_window_at_basic_hit():
    mons = [_mon("M1", 0, 0, 1000, 1000, active_ws=1)]
    clients = [_client("a", ws=1, at=(100, 100), size=(400, 400))]
    assert window_at(200, 200, clients, mons)["address"] == "a"


def test_window_at_outside_any_window_returns_none():
    mons = [_mon("M1", 0, 0, 1000, 1000, active_ws=1)]
    clients = [_client("a", ws=1, at=(100, 100), size=(400, 400))]
    assert window_at(600, 600, clients, mons) is None


def test_window_at_filters_inactive_workspace():
    """A window whose workspace isn't active on ANY monitor must be ignored,
    even if its `at`/`size` happen to contain the point."""
    mons = [_mon("M1", 0, 0, 1000, 1000, active_ws=1)]
    clients = [
        _client("visible", ws=1, at=(0, 0), size=(500, 500), fh=1),
        _client("hidden",  ws=2, at=(0, 0), size=(500, 500), fh=0),  # lower fh but wrong ws
    ]
    assert window_at(250, 250, clients, mons)["address"] == "visible"


def test_window_at_respects_mapped_and_hidden():
    mons = [_mon("M1", 0, 0, 1000, 1000, active_ws=1)]
    clients = [
        _client("a", ws=1, at=(0, 0), size=(500, 500), mapped=False),
        _client("b", ws=1, at=(0, 0), size=(500, 500), hidden=True),
    ]
    assert window_at(100, 100, clients, mons) is None


def test_window_at_title_exclusion():
    mons = [_mon("M1", 0, 0, 1000, 1000, active_ws=1)]
    clients = [_client("a", ws=1, at=(0, 0), size=(500, 500),
                       title="hyprgaze calibration")]
    assert window_at(100, 100, clients, mons, exclude_titles=("hyprgaze",)) is None


def test_window_at_prefers_most_recently_focused_when_stacked():
    mons = [_mon("M1", 0, 0, 1000, 1000, active_ws=1)]
    clients = [
        _client("old", ws=1, at=(0, 0), size=(500, 500), fh=5),
        _client("recent", ws=1, at=(0, 0), size=(500, 500), fh=0),
    ]
    assert window_at(100, 100, clients, mons)["address"] == "recent"


def test_window_at_accepts_point_on_rotated_monitor():
    """Regression: old _monitor_at using untransformed dims thought
    `(4500, 2500)` wasn't on DP-2. With per-client filtering that check is
    gone; this must now hit a window at those coords."""
    mons = [
        _mon("DP-1", 0, 0, 3840, 2160, active_ws=1),
        _mon("DP-2", 3840, -1080, 2160, 3840, active_ws=11),  # post-transform
    ]
    clients = [
        _client("on-dp2", ws=11, at=(3900, 2000), size=(2000, 700), klass="vivaldi"),
    ]
    w = window_at(4500, 2500, clients, mons)
    assert w is not None and w["class"] == "vivaldi"


# ------------------------ window_center ------------------------

def test_window_center():
    c = {"at": [100, 200], "size": [400, 300]}
    assert window_center(c) == (300, 350)
