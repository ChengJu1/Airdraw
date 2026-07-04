"""Trajectory denoising.

Two steps:
1. Outlier removal: points whose displacement to neighbors is far above the median
   displacement are treated as noise and repaired by linear interpolation.
2. 1€ filter (One Euro Filter): real-time friendly low-latency smoothing that balances
   jitter suppression and responsiveness.
   See Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter".
"""
from __future__ import annotations

import math

import numpy as np

from rt_filters import OneEuro, SpikeGate  # noqa: F401  shared implementation with the real-time side


def remove_outliers(x: np.ndarray, y: np.ndarray, max_jump_factor: float = 5.0):
    """Remove jump points with abnormal displacement and repair by interpolation. Returns repaired (x, y)."""
    x = x.astype(np.float64).copy()
    y = y.astype(np.float64).copy()
    n = len(x)
    if n < 3:
        return x, y

    valid = ~(np.isnan(x) | np.isnan(y))
    steps = np.hypot(np.diff(x), np.diff(y))
    med = np.nanmedian(steps[np.isfinite(steps)]) if np.any(np.isfinite(steps)) else 0.0
    if med <= 0:
        return _interp_nan(x), _interp_nan(y)

    thresh = max_jump_factor * med
    for i in range(1, n - 1):
        if not valid[i]:
            continue
        d_prev = math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
        d_next = math.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
        if d_prev > thresh and d_next > thresh:
            x[i] = np.nan
            y[i] = np.nan
    return _interp_nan(x), _interp_nan(y)


def _interp_nan(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    idx = np.arange(len(a))
    good = ~np.isnan(a)
    if good.sum() == 0:
        return a
    a[~good] = np.interp(idx[~good], idx[good], a[good])
    return a


def median_win(x: np.ndarray, y: np.ndarray, win: int = 3):
    """Centered sliding median, removes single-frame spikes.

    Offline uses a centered window with no phase lag (the real-time side can only see
    the past and uses the trailing version in rt_filters.MedianWin).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0 or win <= 1:
        return x.copy(), y.copy()
    half = win // 2
    px = np.pad(x, half, mode="edge")
    py = np.pad(y, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view
    sx = np.median(windows(px, win), axis=1)
    sy = np.median(windows(py, win), axis=1)
    return sx, sy


def smooth_xy(t: np.ndarray, x: np.ndarray, y: np.ndarray,
              mincutoff: float = 1.0, beta: float = 0.007):
    """Run one 1€ filter each on x and y (implementation in rt_filters.OneEuro). t=None assumes 30Hz."""
    fx = OneEuro(mincutoff=mincutoff, beta=beta)
    fy = OneEuro(mincutoff=mincutoff, beta=beta)
    sx = np.empty_like(x, dtype=np.float64)
    sy = np.empty_like(y, dtype=np.float64)
    for i in range(len(x)):
        ti = t[i] if t is not None else None
        sx[i] = fx(float(x[i]), ti)
        sy[i] = fy(float(y[i]), ti)
    return sx, sy


def denoise(traj, mincutoff: float = 1.0, beta: float = 0.045,
            max_jump_factor: float = 5.0, spike_mult: float = 8.0):
    """Full denoising: median -> spike gate -> outlier removal -> 1€ smoothing. Returns smoothed (x, y)."""
    x, y = median_win(traj.x, traj.y, win=3)
    gate = SpikeGate(mult=spike_mult)
    sx, sy = [], []
    for i in range(len(x)):
        px, py = gate(float(x[i]), float(y[i]))
        sx.append(px)
        sy.append(py)
    x, y = np.asarray(sx), np.asarray(sy)
    x, y = remove_outliers(x, y, max_jump_factor=max_jump_factor)
    x, y = smooth_xy(traj.t, x, y, mincutoff=mincutoff, beta=beta)
    return x, y
