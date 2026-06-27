"""轨迹数据的读写与数据结构。

原始轨迹以 CSV 存储，列至少包含: t, x, y；可选 z, pen, in_range。
- t        时间戳（秒）
- x, y     归一化坐标（Skywriter 给的 0~1，手不在感应区时可为空/NaN）
- z        手到面板的高度（0~1）；实测 Skywriter 的 z 容易饱和，不建议单独当抬落笔阈值
- pen      可选：显式抬笔/落笔标记（1=落笔，0=抬笔），有则优先使用
- in_range 可选：手是否在感应区（1=在=落笔，0=离开=抬笔）。采集脚本输出此列，
           比 z 阈值更可靠，载入时会作为 pen 使用
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
                if raw_key is None:      # 某行比表头多出的列，丢弃
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
    # 优先用显式 pen 列；没有则用 in_range（手在不在感应区）当抬笔/落笔依据
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
