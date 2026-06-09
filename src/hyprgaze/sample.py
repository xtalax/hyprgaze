"""The angle sample shared by every tracker backend.

Lives in its own module so lightweight backends (e.g. imu.py) and their tests
don't have to import cv2/mediapipe via tracker.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GazeSample:
    yaw: float          # total (head + eye), radians, + = user's right
    pitch: float        # + = up
    head_yaw: float
    head_pitch: float
    iris_x: float       # average normalized iris offset, image coords (+ = image-right)
    iris_y: float
