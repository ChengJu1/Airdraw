"""用模拟数据跑通 降噪 + 提取 全流程，并输出可视化与 stroke-3。

真实数据接入：把 Skywriter 录制的 CSV 放到 data/ 下，运行：
    python demo.py data/your_capture.csv
CSV 至少含 t,x,y（可选 z, pen）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

from denoise import denoise
from extract import extract
from pen_state import resolve_pen
from trajectory import Trajectory, load_trajectory, save_trajectory
from visualize import plot_pipeline


def make_synthetic(seed: int = 0) -> Trajectory:
    """生成一条带噪声、含抬笔/落笔的模拟轨迹：一个方框 + 一条斜线。"""
    rng = np.random.default_rng(seed)
    segments = []  # (points, pen_down)

    def line(p0, p1, k):
        ts = np.linspace(0, 1, k)
        return np.stack([p0[0] + (p1[0] - p0[0]) * ts,
                         p0[1] + (p1[1] - p0[1]) * ts], axis=1)

    square = [((0.3, 0.3), (0.7, 0.3)), ((0.7, 0.3), (0.7, 0.7)),
              ((0.7, 0.7), (0.3, 0.7)), ((0.3, 0.7), (0.3, 0.3))]
    for a, b in square:
        segments.append((line(a, b, 25), True))
    segments.append((line((0.3, 0.3), (0.45, 0.45), 12), False))  # 抬笔移动
    segments.append((line((0.45, 0.45), (0.6, 0.6), 20), True))    # 第二笔

    xs, ys, zs, pens = [], [], [], []
    for pts, down in segments:
        for (px, py) in pts:
            jitter = rng.normal(0, 0.006, size=2)
            xs.append(px + jitter[0])
            ys.append(py + jitter[1])
            zs.append(0.2 + rng.normal(0, 0.02) if down else 0.6 + rng.normal(0, 0.02))
            pens.append(1.0 if down else 0.0)

    x = np.asarray(xs)
    y = np.asarray(ys)
    # 注入几个离群跳点
    for idx in rng.choice(len(x), size=4, replace=False):
        x[idx] += rng.normal(0, 0.15)
        y[idx] += rng.normal(0, 0.15)

    t = np.arange(len(x)) / 30.0
    return Trajectory(t=t, x=x, y=y, z=np.asarray(zs), pen=np.asarray(pens))


def run(traj: Trajectory, out_prefix: str) -> None:
    pen_down = resolve_pen(traj)
    clean_x, clean_y = denoise(traj)
    strokes, stroke3 = extract(clean_x, clean_y, pen_down)

    img = plot_pipeline(traj.x, traj.y, clean_x, clean_y, pen_down, strokes,
                        out_path=out_prefix + "_pipeline.png")
    np.save(out_prefix + "_stroke3.npy", stroke3)

    print(f"采样点数:        {len(traj)}")
    print(f"落笔点数:        {int(np.sum(pen_down))}")
    print(f"提取笔画数:      {len(strokes)}")
    print(f"stroke-3 步数:   {len(stroke3)}")
    print(f"可视化已保存:    {img}")
    print(f"stroke-3 已保存: {out_prefix}_stroke3.npy")


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    if len(sys.argv) > 1:
        traj = load_trajectory(sys.argv[1])
        prefix = os.path.join("out", os.path.splitext(os.path.basename(sys.argv[1]))[0])
    else:
        traj = make_synthetic()
        os.makedirs("data", exist_ok=True)
        save_trajectory("data/synthetic.csv", traj)
        prefix = os.path.join("out", "synthetic")
        print("未提供数据文件，使用模拟轨迹（已存到 data/synthetic.csv）。\n")
    run(traj, prefix)
