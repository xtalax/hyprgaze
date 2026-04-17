"""One-Euro filter — low-lag smoothing for noisy pointer signals.

Casiez, Roussel & Vogel, 2012. The cutoff frequency adapts to signal
speed: quiet signal → aggressive smoothing, fast motion → light smoothing.
"""
from __future__ import annotations

import math


class OneEuroFilter:
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._t_prev: float | None = None
        self._x_prev: float = 0.0
        self._dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1 / (2 * math.pi * cutoff)
        return 1 / (1 + tau / dt)

    def __call__(self, t: float, x: float) -> float:
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = x
            return x
        dt = t - self._t_prev
        if dt <= 0:
            return self._x_prev
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev
        self._t_prev = t
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat
