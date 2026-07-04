"""Run the full denoise + extract pipeline on synthetic data, output visualization and stroke-3.

For real data: put a Skywriter-recorded CSV under data/ and run:
    python demo.py data/your_capture.csv
The CSV needs at least t,x,y (optional z, pen).
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
    """Generate a noisy synthetic trajectory with pen up/down: a square plus a diagonal line."""
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
    segments.append((line((0.3, 0.3), (0.45, 0.45), 12), False))  # pen-up move
    segments.append((line((0.45, 0.45), (0.6, 0.6), 20), True))    # second stroke

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
    # Inject a few outlier jump points
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

    print(f"samples:         {len(traj)}")
    print(f"pen-down points: {int(np.sum(pen_down))}")
    print(f"strokes:         {len(strokes)}")
    print(f"stroke-3 steps:  {len(stroke3)}")
    print(f"plot saved:      {img}")
    print(f"stroke-3 saved:  {out_prefix}_stroke3.npy")


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
        print("No data file given, using synthetic trajectory (saved to data/synthetic.csv).\n")
    run(traj, prefix)
