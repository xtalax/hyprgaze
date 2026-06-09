"""6-axis IMU head tracking from AR glasses (e.g. RayNeo Air 4 Pro).

An alternative to the webcam `GazeTracker`: instead of estimating gaze from a
camera, read the glasses' built-in gyro + accelerometer over hidraw and produce
head yaw/pitch — the same (yaw, pitch) signal the rest of the pipeline already
consumes (Calibration.apply → filter → dwell → focus).

Why this is the better modality for a reclined / bedbound user:
  • Works lying down — no webcam needing line-of-sight to the face.
  • Lower latency than MediaPipe + solvePnP, no lighting dependence.
  • No dependency on XRLinuxDriver (which does NOT support the Air 4 Pro).

Sign convention matches tracker.py:  yaw > 0 = head turned to user's RIGHT,
pitch > 0 = looking UP. Output is head-only (no iris term); iris_* = 0.

--------------------------------------------------------------------------
TWO BACKENDS (both hardware-gated; the glasses are not here yet):

  1. XRDriverHeadTracker — PREFERRED once XRLinuxDriver supports the Air 4 Pro.
     Plan: add Air 4 Pro support upstream to wheaney/XRLinuxDriver when the
     glasses arrive. The driver then publishes a *fused* orientation we read
     directly — no protocol decode or fusion duplicated here. Confirm the
     driver's IMU output path/format during that work (TODO in the class).

  2. ImuHeadTracker (hidraw + ComplementaryFilter) — independent fallback that
     reads raw gyro/accel and fuses them here. To finish the raw decode:
       a. Plug in the Air 4 Pro, find its hidraw node:
            grep -l HID_ID /sys/class/hidraw/*/device/uevent   # note vid:pid
       b. Capture raw reports:  sudo cat /dev/hidrawN | xxd | head
       c. Decode the gyro/accel layout. Reference for a sibling device:
            github.com/verncat/RayNeo-Air-3S-Pro-OpenVR (Air 4 Pro may match).

  The fusion math (ComplementaryFilter) is device-independent and unit-tested;
  only the byte decode + VID/PID (backend 2) and the driver IPC path
  (backend 1) need the hardware.
--------------------------------------------------------------------------
"""
from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass

import numpy as np

from .sample import GazeSample

# (idVendor, idProduct) of the glasses' IMU HID interface. PLACEHOLDER — fill in
# from step 1 above, or override at runtime via HYPRGAZE_IMU_IDS="1bbb:af50".
RAYNEO_IDS: tuple[tuple[int, int], ...] = (
    # (0x35ca, 0x1011),  # RayNeo Air 4 Pro — TODO verify
)


@dataclass
class ImuAxes:
    """How the glasses' sensor axes map to head yaw/pitch.

    Device-/mounting-specific; defaults are a reasonable starting guess for a
    glasses IMU with +X right, +Y up, +Z out-of-lens. Tune signs during the
    first calibration (a wrong sign just inverts an axis — easy to spot).
    """
    gyro_yaw_axis: int = 1     # which gyro component is yaw rate (about vertical)
    gyro_pitch_axis: int = 0   # which gyro component is pitch rate
    gyro_yaw_sign: float = 1.0
    gyro_pitch_sign: float = 1.0
    # Accelerometer axis indices for gravity-referenced pitch.
    acc_forward_axis: int = 2  # +Z out of lens
    acc_up_axis: int = 1       # +Y up
    acc_pitch_sign: float = 1.0


class ComplementaryFilter:
    """Fuse gyro (rate) + accelerometer (gravity) into pitch & yaw.

    PITCH is gravity-referenced and absolute: the accelerometer term anchors it
    so integration drift is bounded. YAW has no absolute reference (no
    magnetometer), so it is pure gyro integration and *will* drift slowly —
    `recenter()` (bound to a key, like the camera path's `zero`) resets it.

    Pure / no IO — unit-tested in tests/test_imu.py.
    """

    def __init__(self, axes: ImuAxes | None = None, tau: float = 0.5):
        # tau = complementary time constant (s). alpha = tau / (tau + dt):
        # high alpha trusts the gyro short-term, the accel pulls pitch back long-term.
        self.axes = axes or ImuAxes()
        self.tau = tau
        self.pitch = 0.0   # radians, gravity-referenced
        self.yaw = 0.0     # radians, gyro-integrated (drifts)
        self._yaw0 = 0.0
        self._pitch0 = 0.0
        self._initialized = False
        self._g_ema: float | None = None   # running |accel|, for the near-rest gate

    def _accel_pitch(self, acc: np.ndarray) -> float:
        # Gravity-referenced pitch: 0 at neutral (forward axis ⟂ gravity), grows
        # as the forward axis tilts toward/away from gravity. Exact axis + sign
        # are mounting-specific (refined at calibration); a wrong sign just
        # inverts the axis, which is obvious on screen.
        a = self.axes
        fwd = float(acc[a.acc_forward_axis])
        up = float(acc[a.acc_up_axis])
        return a.acc_pitch_sign * math.atan2(fwd, up if abs(up) > 1e-9 else 1e-9)

    def update(self, gyro: np.ndarray, acc: np.ndarray, dt: float) -> tuple[float, float]:
        """Advance the filter. gyro in rad/s, acc in any consistent unit, dt in s.

        Returns (yaw, pitch) already offset by the last recenter().
        """
        a = self.axes
        if dt <= 0:
            return self.yaw - self._yaw0, self.pitch - self._pitch0

        # Track gravity magnitude (unit-agnostic: g-units ~1, m/s^2 ~9.81) so we
        # can distinguish "near rest" (accel ≈ gravity) from a "head jerk" (extra
        # linear accel) and only trust the accel for pitch when near rest.
        mag = float(np.linalg.norm(acc))
        self._g_ema = mag if self._g_ema is None else 0.98 * self._g_ema + 0.02 * mag

        acc_pitch = self._accel_pitch(acc)
        if not self._initialized:
            self.pitch = acc_pitch          # seed from gravity, skip long settle
            self._initialized = True

        # Integrate gyro on both axes.
        self.yaw += a.gyro_yaw_sign * float(gyro[a.gyro_yaw_axis]) * dt
        pitch_gyro = self.pitch + a.gyro_pitch_sign * float(gyro[a.gyro_pitch_axis]) * dt

        # Complementary blend: gyro short-term, accel pulls pitch back long-term.
        alpha = self.tau / (self.tau + dt)
        near_rest = self._g_ema is not None and abs(mag - self._g_ema) < 0.15 * self._g_ema
        self.pitch = alpha * pitch_gyro + (1 - alpha) * acc_pitch if near_rest else pitch_gyro

        return self.yaw - self._yaw0, self.pitch - self._pitch0

    def recenter(self) -> None:
        """Make the current head pose the (0, 0) screen-center reference."""
        self._yaw0 = self.yaw
        self._pitch0 = self.pitch


def find_rayneo_hidraw() -> str | None:
    """Return the /dev/hidrawN path for the glasses IMU, or None.

    Honors HYPRGAZE_IMU_IDS='vid:pid[,vid:pid]' and HYPRGAZE_IMU_HIDRAW=/dev/hidrawN.
    """
    forced = os.environ.get("HYPRGAZE_IMU_HIDRAW")
    if forced and os.path.exists(forced):
        return forced

    ids = list(RAYNEO_IDS)
    env_ids = os.environ.get("HYPRGAZE_IMU_IDS")
    if env_ids:
        for tok in env_ids.split(","):
            v, _, p = tok.strip().partition(":")
            try:
                ids.append((int(v, 16), int(p, 16)))
            except ValueError:
                pass
    if not ids:
        return None

    for node in glob.glob("/sys/class/hidraw/hidraw*"):
        uevent = os.path.join(node, "device", "uevent")
        try:
            with open(uevent) as f:
                text = f.read()
        except OSError:
            continue
        # HID_ID lines look like: HID_ID=0003:000035CA:00001011
        for line in text.splitlines():
            if not line.startswith("HID_ID="):
                continue
            parts = line.split(":")
            if len(parts) != 3:
                continue
            try:
                vid = int(parts[1], 16) & 0xFFFF
                pid = int(parts[2], 16) & 0xFFFF
            except ValueError:
                continue
            if (vid, pid) in ids:
                return "/dev/" + os.path.basename(node)
    return None


class RayNeoReport:
    """Decode a raw hidraw report into (gyro rad/s, accel) arrays.

    PLACEHOLDER decode — see the HARDWARE-GATED TODO at module top. Replace
    `_decode` with the real byte layout once captured from the device.
    """

    REPORT_LEN = 64  # TODO verify

    @staticmethod
    def _decode(buf: bytes) -> tuple[np.ndarray, np.ndarray] | None:
        if len(buf) < 12:        # placeholder guard; set the real REPORT_LEN when decoded
            return None
        # TODO: real layout. Typical pattern: int16 LE triplets for gyro & accel
        # at known offsets, plus a scale factor. Example skeleton (DO NOT trust
        # offsets/scales — placeholders):
        #   gyro_raw  = struct.unpack_from('<3h', buf, GYRO_OFF)
        #   accel_raw = struct.unpack_from('<3h', buf, ACCEL_OFF)
        #   gyro  = np.array(gyro_raw)  * GYRO_SCALE_RAD_S
        #   accel = np.array(accel_raw) * ACCEL_SCALE
        #   return gyro, accel
        return None


class ImuHeadTracker:
    """Webcam-free head tracker: reads the glasses IMU, outputs GazeSample.

    Mirrors the GazeTracker interface enough for sources.py: `.poll(t)` returns
    a GazeSample or None, `.recenter()`, `.close()`.
    """

    def __init__(self, axes: ImuAxes | None = None, hidraw_path: str | None = None):
        self.filter = ComplementaryFilter(axes)
        self.path = hidraw_path or find_rayneo_hidraw()
        self._fd: int | None = None
        self._last_t: float | None = None
        if self.path:
            try:
                self._fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError as e:
                raise RuntimeError(
                    f"Cannot open IMU hidraw {self.path}: {e}. "
                    f"Add a udev rule granting read access, or run via the user service."
                ) from e

    def available(self) -> bool:
        return self._fd is not None

    def poll(self, t: float) -> GazeSample | None:
        if self._fd is None:
            return None
        try:
            buf = os.read(self._fd, RayNeoReport.REPORT_LEN)
        except BlockingIOError:
            return None
        except OSError:
            return None
        decoded = RayNeoReport._decode(buf)
        if decoded is None:
            return None
        gyro, acc = decoded
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        yaw, pitch = self.filter.update(gyro, acc, dt)
        return GazeSample(yaw=yaw, pitch=pitch, head_yaw=yaw, head_pitch=pitch,
                          iris_x=0.0, iris_y=0.0)

    def recenter(self) -> None:
        self.filter.recenter()

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


# Where XRLinuxDriver publishes its fused IMU orientation for clients like
# breezy-desktop. TODO: confirm the exact path/format when we add Air 4 Pro
# support upstream (driver builds have used /dev/shm and control files).
XRDRIVER_IMU_PATHS: tuple[str, ...] = (
    "/dev/shm/xr_driver_imu_data",
)


class XRDriverHeadTracker:
    """Head tracker that consumes XRLinuxDriver's already-fused orientation.

    Preferred over ImuHeadTracker once XRLinuxDriver supports the Air 4 Pro: the
    driver does the device decode + sensor fusion (and yaw drift handling), and
    we just read euler angles. recenter() is applied as a yaw/pitch offset here.

    DRIVER-GATED: `_read_orientation` is a stub until the driver's IMU output
    path + format are confirmed during the upstream Air 4 Pro support work.
    Same `.poll/.recenter/.close/.available` shape as ImuHeadTracker.
    """

    def __init__(self, path: str | None = None):
        self.path = path or next((p for p in XRDRIVER_IMU_PATHS if os.path.exists(p)), None)
        self._yaw0 = 0.0
        self._pitch0 = 0.0
        self._last = (0.0, 0.0)

    def available(self) -> bool:
        return self.path is not None

    def _read_orientation(self) -> tuple[float, float] | None:
        """Return (yaw, pitch) in radians from the driver, or None. TODO: real read."""
        return None

    def poll(self, _t: float) -> GazeSample | None:
        del _t  # driver gives absolute orientation; no timestamp needed
        ori = self._read_orientation()
        if ori is None:
            return None
        self._last = ori
        ry, rp = ori[0] - self._yaw0, ori[1] - self._pitch0
        return GazeSample(yaw=ry, pitch=rp, head_yaw=ry, head_pitch=rp,
                          iris_x=0.0, iris_y=0.0)

    def recenter(self) -> None:
        self._yaw0, self._pitch0 = self._last

    def close(self) -> None:
        pass
