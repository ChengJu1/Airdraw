"""Trajectory data structures and CSV I/O.

Raw trajectories are stored as CSV with at least: t, x, y; optional z, pen, in_range.
- t        timestamp (seconds)
- x, y     normalized coordinates (0~1 from the Skywriter; may be empty/NaN when the hand is out of range)
- z        hand height above the panel (0~1); in practice the Skywriter z saturates easily,
           so it is not recommended as the sole pen up/down threshold
- pen      optional: explicit pen up/down flag (1=down, 0=up); used first if present
- in_range optional: whether the hand is in the sensing area (1=in=down, 0=out=up).
           The capture script writes this column; it is more reliable than a z threshold
           and is used as pen when loading
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Trajectory:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: Optional[np.ndarray] = None
    pen: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.x)


def load_trajectory(path: str) -> Trajectory:
    cols: dict[str, list[float]] = {}
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        names = [n.strip().lower() for n in (reader.fieldnames or [])]
        for n in names:
            cols[n] = []
        for row in reader:
            for raw_key, val in row.items():
                if raw_key is None:      # row has more columns than the header, drop them
                    continue
                key = raw_key.strip().lower()
                if key not in cols:
                    continue
                try:
                    cols[key].append(float(val))
                except (TypeError, ValueError):
                    cols[key].append(np.nan)

    def arr(name: str) -> Optional[np.ndarray]:
        return np.asarray(cols[name], dtype=np.float64) if name in cols else None

    n = len(next(iter(cols.values()))) if cols else 0
    t = arr("t")
    if t is None:
        t = np.arange(n, dtype=np.float64)
    # Prefer the explicit pen column; otherwise use in_range (hand in sensing area) for pen up/down
    pen = arr("pen")
    if pen is None:
        pen = arr("in_range")
    return Trajectory(t=t, x=arr("x"), y=arr("y"), z=arr("z"), pen=pen)


def save_trajectory(path: str, traj: Trajectory) -> None:
    fields = ["t", "x", "y"]
    if traj.z is not None:
        fields.append("z")
    if traj.pen is not None:
        fields.append("pen")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for i in range(len(traj)):
            row = [traj.t[i], traj.x[i], traj.y[i]]
            if traj.z is not None:
                row.append(traj.z[i])
            if traj.pen is not None:
                row.append(traj.pen[i])
            writer.writerow(row)
