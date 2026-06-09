"""Unit tests for the niri IPC layer (no niri process needed)."""
from hyprgaze.niri import NiriFocuser, bounding_box, detect_compositor, parse_outputs

# Representative `niri msg --json outputs` payload (connector→Output map). One
# enabled glasses output + one disabled internal panel (logical = null).
SAMPLE_OUTPUTS = {
    "DP-1": {
        "name": "DP-1",
        "make": "Viture", "model": "Luma Ultra",
        "logical": {"x": 0, "y": 0, "width": 1920, "height": 1080,
                    "scale": 1.0, "transform": "normal"},
        "current_mode": 0,
        "modes": [{"width": 1920, "height": 1080, "refresh_rate": 120000,
                   "is_preferred": True}],
    },
    "eDP-1": {"name": "eDP-1", "logical": None},  # disabled → skipped
}


def test_parse_outputs_dict_form():
    mons = parse_outputs(SAMPLE_OUTPUTS)
    assert len(mons) == 1
    m = mons[0]
    assert (m.name, m.x, m.y, m.w, m.h, m.scale) == ("DP-1", 0, 0, 1920, 1080, 1.0)


def test_parse_outputs_list_form():
    mons = parse_outputs(list(SAMPLE_OUTPUTS.values()))
    assert [m.name for m in mons] == ["DP-1"]   # disabled output still skipped


def test_parse_outputs_empty():
    assert parse_outputs(None) == []
    assert parse_outputs({}) == []


def test_bounding_box_from_outputs():
    box = bounding_box(parse_outputs(SAMPLE_OUTPUTS))
    assert (box.x0, box.y0, box.w, box.h) == (0, 0, 1920, 1080)


def test_detect_compositor(monkeypatch):
    monkeypatch.delenv("NIRI_SOCKET", raising=False)
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    assert detect_compositor() == "hyprland"
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "abc")
    assert detect_compositor() == "hyprland"
    monkeypatch.setenv("NIRI_SOCKET", "/run/niri.sock")
    assert detect_compositor() == "niri"   # niri wins when both set


# --- NiriFocuser dwell/threshold policy ---

def test_focuser_warps_only_after_dwell():
    f = NiriFocuser(dwell_sec=0.4, stable_radius_px=60, move_threshold_px=80)
    assert f.update(0.0, 100, 100) is None         # dwell starts
    assert f.update(0.3, 105, 105) is None          # jitter within radius, <dwell
    assert f.update(0.45, 105, 105) == (100, 100)   # dwell elapsed → warp to anchor


def test_focuser_no_repeat_until_look_away():
    f = NiriFocuser(dwell_sec=0.4, stable_radius_px=60, move_threshold_px=80)
    f.update(0.0, 100, 100)
    assert f.update(0.45, 100, 100) == (100, 100)   # first warp
    assert f.update(0.6, 100, 100) is None           # disarmed — no repeat warp
    # Look away (beyond radius) then re-dwell far enough → warps again.
    assert f.update(0.7, 600, 600) is None           # resets candidate, re-arms
    assert f.update(1.2, 600, 600) == (600, 600)


def test_focuser_suppresses_tiny_moves():
    f = NiriFocuser(dwell_sec=0.1, stable_radius_px=60, move_threshold_px=80)
    f.update(0.0, 200, 200)
    assert f.update(0.2, 200, 200) == (200, 200)     # first warp at (200,200)
    f.update(0.3, 400, 400)                           # look away (re-arm)
    # Re-dwell only ~42px from the last warp (< 80 threshold) → suppressed.
    assert f.update(0.6, 230, 230) is None
