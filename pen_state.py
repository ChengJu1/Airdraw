"""抬笔 / 落笔判定。

推荐方案：基于 Z 高度的双阈值迟滞（hysteresis）。
手压低（z 小）落笔画线，抬高（z 大）移动不画；中间区间保持上一状态，避免抖动。
若轨迹自带显式 pen 列，则直接采用。
"""
from __future__ import annotations

import numpy as np


def pen_from_z(
    z: np.ndarray,
    down_thresh: float = 0.30,
    up_thresh: float = 0.45,
    min_run: int = 2,
    z_is_height: bool = True,
) -> np.ndarray:
    """由 z 高度推断每个采样点是否落笔（True=落笔）。

    z_is_height=True 表示 z 越小手越靠近面板（落笔）。
    down_thresh < up_thresh 形成迟滞带。min_run 用于过滤过短的状态翻转。
    """
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    pen = np.zeros(n, dtype=bool)
    state = False
    for i in range(n):
        zi = z[i]
        if np.isnan(zi):
            pen[i] = False
            state = False
            continue
        if z_is_height:
            if not state and zi < down_thresh:
                state = True
            elif state and zi > up_thresh:
                state = False
        else:
            if not state and zi > up_thresh:
                state = True
            elif state and zi < down_thresh:
                state = False
        pen[i] = state
    return _debounce(pen, min_run)


def _debounce(pen: np.ndarray, min_run: int) -> np.ndarray:
    """剔除长度小于 min_run 的连续段，避免毛刺式的抬落笔。"""
    if min_run <= 1 or len(pen) == 0:
        return pen
    out = pen.copy()
    start = 0
    for i in range(1, len(out) + 1):
        if i == len(out) or out[i] != out[start]:
            if i - start < min_run:
                out[start:i] = not out[start] if start > 0 else out[start]
            start = i
    return out


def resolve_pen(traj) -> np.ndarray:
    """优先使用显式 pen 列，否则用 z 推断，再否则全部视为落笔。"""
    if traj.pen is not None and not np.all(np.isnan(traj.pen)):
        return np.nan_to_num(traj.pen, nan=0.0) > 0.5
    if traj.z is not None and not np.all(np.isnan(traj.z)):
        return pen_from_z(traj.z)
    return ~np.isnan(traj.x)
