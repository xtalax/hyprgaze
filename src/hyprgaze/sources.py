"""Sample sources for the focus loop: webcam gaze or glasses IMU.

Each source yields (GazeSample | None, debug_frame | None) per tick, so the
dwell/focus loop in __main__ is identical regardless of input device.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .sample import GazeSample


class Source(Protocol):
    def read(self, t: float) -> tuple[GazeSample | None, np.ndarray | None]: ...
    def recenter(self) -> None: ...
    def close(self) -> None: ...


class CameraSource:
    """Webcam + MediaPipe gaze (the original path), wrapped as a Source."""

    def __init__(self, camera_index: int, tracker_cfg: dict):
        import cv2  # local import: IMU-only users needn't have OpenCV loaded
        from .tracker import GazeTracker
        cam = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_FPS, 30)
        self.cam = cam
        self.opened = cam.isOpened()
        self.tracker = GazeTracker(**tracker_cfg)

    def read(self, t: float) -> tuple[GazeSample | None, np.ndarray | None]:
        ok, frame = self.cam.read()
        if not ok:
            return None, None
        return self.tracker.process(frame, t), frame

    def recenter(self) -> None:
        # Camera baseline is set by calibration / `zero`, not a live recenter.
        pass

    def close(self) -> None:
        self.cam.release()


class ImuSource:
    """Glasses IMU head tracking, wrapped as a Source.

    Prefers XRLinuxDriver's fused orientation when available, else raw hidraw +
    our ComplementaryFilter. No debug frame (returns None); __main__ draws on a
    synthetic canvas when --debug is set.
    """

    def __init__(self, prefer_driver: bool = True):
        from .imu import ImuHeadTracker, XRDriverHeadTracker
        tracker: object | None = None
        if prefer_driver:
            xr = XRDriverHeadTracker()
            if xr.available():
                tracker = xr
        if tracker is None:
            tracker = ImuHeadTracker()
        self.tracker = tracker
        self.kind = type(tracker).__name__
        self.ready: bool = tracker.available()  # type: ignore[attr-defined]

    def read(self, t: float) -> tuple[GazeSample | None, np.ndarray | None]:
        return self.tracker.poll(t), None  # type: ignore[attr-defined]

    def recenter(self) -> None:
        self.tracker.recenter()  # type: ignore[attr-defined]

    def close(self) -> None:
        self.tracker.close()  # type: ignore[attr-defined]
