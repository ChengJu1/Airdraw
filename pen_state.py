"""Pen up/down detection.

Recommended approach: dual-threshold hysteresis on Z height.
Hand low (small z) = pen down and drawing; hand high (large z) = moving without drawing;
in between, keep the previous state to avoid jitter.
If the trajectory carries an explicit pen column, use it directly.
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
    """Infer pen-down state for each sample from z height (True=down).

    z_is_height=True means smaller z = hand closer to the panel (pen down).
    down_thresh < up_thresh forms a hysteresis band. min_run filters out overly short state flips.
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
    """Remove runs shorter than min_run to avoid glitchy pen up/down flips."""
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
    """Use the explicit pen column first; else infer from z; else treat everything as pen down."""
    if traj.pen is not None and not np.all(np.isnan(traj.pen)):
        return np.nan_to_num(traj.pen, nan=0.0) > 0.5
    if traj.z is not None and not np.all(np.isnan(traj.z)):
        return pen_from_z(traj.z)
    return ~np.isnan(traj.x)
