"""轨迹降噪。

两步：
1. 离群跳点剔除：相邻点位移远大于中位位移的点视为噪点，用线性插值修补。
2. 1€ 滤波（One Euro Filter）：实时友好的低延迟平滑，兼顾抖动抑制与跟手性。
   参考 Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter"。
"""
from __future__ import annotations

import math

import numpy as np

from rt_filters import OneEuro, SpikeGate  # noqa: F401  与实时端共用同一实现


def remove_outliers(x: np.ndarray, y: np.ndarray, max_jump_factor: float = 5.0):
    """剔除位移异常的跳点并插值修补。返回修补后的 (x, y)。"""
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
    """居中滑动中值，去掉单帧尖刺。

    离线用居中窗口，无相位滞后（实时端只能看过去，用 rt_filters.MedianWin 的拖尾版）。
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
    """对 x、y 各跑一路 1€ 滤波（实现见 rt_filters.OneEuro）。t=None 按 30Hz 处理。"""
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
    """完整降噪：中值 -> 尖刺门控 -> 剔跳点 -> 1€ 平滑。返回平滑后的 (x, y)。"""
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
