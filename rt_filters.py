"""Real-time filter chain (stdlib only, works on both the Pi and the desktop).

Shared by web_capture.py / draw_app.py / denoise.py so the same filter code
isn't copied three times with parameters drifting apart. When deploying to the
Pi, copy this file together with mgc3130.py.

Chain: 3-point median (MedianWin) -> spike gate (SpikeGate) -> 1€ smoothing (OneEuro)
Pen up/down: ZPenHysteresis (dual-threshold hysteresis + consecutive-frame debounce)
"""
from __future__ import annotations

import math
from collections import deque

# ---- Defaults tuned on real data (calibrated in web_capture, reused by draw_app) ----
MEDIAN_WIN = 3
SPIKE_MULT = 8.0          # jump > mult x recent median step => drop the frame
SPIKE_FLOOR = 0.002       # p95 step while drawing a circle is ~0.004; too high a floor kills tracking points
SPIKE_WIN = 24
ONE_EURO_MINCUTOFF = 1.0  # tracks better than 1.8; combined with beta suppresses circle lag
ONE_EURO_BETA = 0.045
ONE_EURO_DCUTOFF = 1.0
Z_PEN_DOWN = 0.26         # pen down when z stays below this for PEN_MIN_FRAMES frames
Z_PEN_UP = 0.32           # pen up when z stays above this; in between keep previous state
PEN_MIN_FRAMES = 2


class MedianWin:
    """Sliding median, removes single-frame spikes."""

    def __init__(self, win=MEDIAN_WIN):
        self.xs = deque(maxlen=win)
        self.ys = deque(maxlen=win)

    def reset(self):
        self.xs.clear()
        self.ys.clear()

    def __call__(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        sx = sorted(self.xs)[len(self.xs) // 2]
        sy = sorted(self.ys)[len(self.ys) // 2]
        return sx, sy


class SpikeGate:
    """Adaptive spike gate: drop jumps that are abnormal relative to the recent median step.

    Only isolated spikes are dropped: max_reject consecutive over-limit frames
    mean genuine fast movement, so accept the new position and re-anchor.
    Otherwise the anchor stays at the old position forever, the per-frame
    distance only grows, and the gate locks up (cursor frozen at one point).
    """

    def __init__(self, mult=SPIKE_MULT, floor=SPIKE_FLOOR, win=SPIKE_WIN,
                 max_reject=3):
        self.mult = mult
        self.floor = floor
        self.max_reject = max_reject
        self.steps = deque(maxlen=win)
        self.x = None
        self.y = None
        self._rej = 0

    def reset(self):
        self.steps.clear()
        self.x = self.y = None
        self._rej = 0

    def __call__(self, x, y):
        if self.x is None:
            self.x, self.y = x, y
            return x, y
        d = math.hypot(x - self.x, y - self.y)
        med = sorted(self.steps)[len(self.steps) // 2] if len(self.steps) >= 5 else 0.003
        lim = max(self.floor, self.mult * max(med, 0.0004))
        if d > lim:
            self._rej += 1
            if self._rej < self.max_reject:
                return self.x, self.y
            # consecutive over-limit: treat as real movement and re-anchor (big step not added to stats)
            self._rej = 0
            self.x, self.y = x, y
            return x, y
        self._rej = 0
        self.steps.append(d)
        self.x, self.y = x, y
        return x, y


class OneEuro:
    """1€ filter (Casiez et al.): stable when slow, low latency when fast.

    __call__(x, t) adapts to real dt when a timestamp is given; t=None falls
    back to fixed freq. mincutoff can be overridden per call (e.g. stronger
    smoothing depending on z height).
    """

    def __init__(self, mincutoff=ONE_EURO_MINCUTOFF, beta=ONE_EURO_BETA,
                 dcutoff=ONE_EURO_DCUTOFF, freq=30.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.freq = freq
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t=None, mincutoff=None):
        mc = self.mincutoff if mincutoff is None else mincutoff
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        if t is not None and self.t_prev is not None:
            dt = t - self.t_prev
            if dt <= 0:
                dt = 1e-3
        else:
            dt = 1.0 / self.freq
        self.t_prev = t
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = mc + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class ZPenHysteresis:
    """Pen up/down from z height: dual-threshold hysteresis + consecutive-frame debounce.

    With a single threshold, z noise while the hand hovers at the critical
    height makes pen flip between 0/1 and one stroke shatters into fragments.
    Here pen goes down only after z < down for min_frames consecutive frames,
    and up only after z > up for min_frames frames; in between the previous
    state is kept (real-time version of pen_state.pen_from_z).
    """

    def __init__(self, down=Z_PEN_DOWN, up=Z_PEN_UP, min_frames=PEN_MIN_FRAMES):
        self.down = down
        self.up = up
        self.min_frames = min_frames
        self.state = 0
        self._cnt = 0

    def reset(self):
        self.state = 0
        self._cnt = 0

    def __call__(self, z):
        if self.state == 0:
            want = 1 if z < self.down else 0
        else:
            want = 0 if z > self.up else 1
        if want != self.state:
            self._cnt += 1
            if self._cnt >= self.min_frames:
                self.state = want
                self._cnt = 0
        else:
            self._cnt = 0
        return self.state
