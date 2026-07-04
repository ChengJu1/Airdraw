"""Trajectory extraction and normalization.

Pipeline: segment by pen up/down -> global normalization -> resample each stroke at equal
arc-length spacing -> convert to stroke-3.
(Normalize before resampling so resample_spacing is in normalized units, independent of
the original data range.)
stroke-3 is the Sketch-RNN input format: each row [dx, dy, pen_lift], pen_lift=1 means
the pen lifts after this point.
"""
from __future__ import annotations

import numpy as np


def segment_strokes(x: np.ndarray, y: np.ndarray, pen_down: np.ndarray,
                    min_points: int = 3):
    """Split into strokes by contiguous pen-down runs. Returns [(N_i, 2), ...]."""
    strokes = []
    n = len(x)
    i = 0
    while i < n:
        if pen_down[i] and not np.isnan(x[i]):
            j = i
            pts = []
            while j < n and pen_down[j] and not np.isnan(x[j]):
                pts.append((x[j], y[j]))
                j += 1
            if len(pts) >= min_points:
                strokes.append(np.asarray(pts, dtype=np.float64))
            i = j
        else:
            i += 1
    return strokes


def resample_stroke(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample one stroke at equal arc-length spacing. spacing is the target point distance (normalized units, default 0~255 scale)."""
    if len(points) < 2:
        return points
    d = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = cum[-1]
    if total <= 0:
        return points[:1]
    n = max(2, int(round(total / spacing)) + 1)
    targets = np.linspace(0.0, total, n)
    rx = np.interp(targets, cum, points[:, 0])
    ry = np.interp(targets, cum, points[:, 1])
    return np.stack([rx, ry], axis=1)


def normalize_strokes(strokes, target_size: float = 255.0):
    """Center all strokes and uniformly scale to a common size (aspect ratio preserved)."""
    if not strokes:
        return strokes
    allp = np.concatenate(strokes, axis=0)
    mn = allp.min(axis=0)
    mx = allp.max(axis=0)
    span = float((mx - mn).max())
    if span <= 0:
        span = 1.0
    scale = target_size / span
    center = (mn + mx) / 2.0
    return [(s - center) * scale for s in strokes]


def to_stroke3(strokes) -> np.ndarray:
    """Convert to Sketch-RNN stroke-3 format: [dx, dy, pen_lift]."""
    out = []
    prev = None
    for stroke in strokes:
        for (px, py) in stroke:
            if prev is None:
                prev = (px, py)
                continue
            out.append([px - prev[0], py - prev[1], 0.0])
            prev = (px, py)
        if out:
            out[-1][2] = 1.0
    return np.asarray(out, dtype=np.float32) if out else np.zeros((0, 3), np.float32)


def extract(x: np.ndarray, y: np.ndarray, pen_down: np.ndarray,
            resample_spacing: float = 3.0, target_size: float = 255.0,
            min_points: int = 3):
    """End-to-end extraction: returns (normalized strokes, stroke3)."""
    strokes = segment_strokes(x, y, pen_down, min_points=min_points)
    strokes = normalize_strokes(strokes, target_size=target_size)
    strokes = [resample_stroke(s, resample_spacing) for s in strokes]
    stroke3 = to_stroke3(strokes)
    return strokes, stroke3
