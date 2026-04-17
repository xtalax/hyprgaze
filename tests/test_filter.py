"""One-Euro filter: should smooth noise but track steady-state signals."""
from __future__ import annotations

from hyprgaze.filter import OneEuroFilter


def test_first_sample_passthrough():
    f = OneEuroFilter()
    assert f(0.0, 42.0) == 42.0


def test_tracks_steady_state_exactly():
    f = OneEuroFilter()
    # Feed a constant signal for a while; filter should converge to it.
    out = 0.0
    for i in range(20):
        out = f(i * 0.03, 100.0)
    assert abs(out - 100.0) < 1e-6


def test_smooths_noise_at_steady_input():
    # Noise around 50; filtered output should sit near 50 with small spread.
    import random
    random.seed(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.01)
    outputs = []
    for i in range(200):
        t = i * 0.033
        noisy = 50.0 + random.gauss(0, 5.0)
        outputs.append(f(t, noisy))
    tail = outputs[100:]
    mean = sum(tail) / len(tail)
    var = sum((x - mean) ** 2 for x in tail) / len(tail)
    assert abs(mean - 50.0) < 2.0
    # Filter should attenuate noise variance (input std=5 → var=25).
    assert var < 25.0 / 4


def test_reacts_to_step_change():
    """After a step, filter should converge to the new value within ~1 s."""
    f = OneEuroFilter()
    for i in range(30):
        f(i * 0.03, 0.0)
    # Step to 100.
    out = 0.0
    for i in range(30, 60):
        out = f(i * 0.03, 100.0)
    assert out > 90.0


def test_nonmonotonic_timestamp_is_safe():
    """Time going backwards shouldn't crash — just reuses previous estimate."""
    f = OneEuroFilter()
    f(1.0, 10.0)
    # dt <= 0 returns previous x.
    assert f(0.5, 99.0) == 10.0
    # Then forward again works.
    assert f(2.0, 10.0) == 10.0
