"""Offline automatic processing of a Skywriter capture CSV: drop saturation -> segment on
jumps -> per-stroke 1€ smoothing -> robust normalization -> stroke-3, and output a
"raw vs cleaned" comparison plot.

Usage:
    python process_capture.py cap_xxx.csv
Fully automatic, no manual range/orientation tuning needed.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trajectory import load_trajectory
from denoise import smooth_xy
from extract import resample_stroke, normalize_strokes, to_stroke3

# ---- Parameters (robust defaults set from the data distribution) ----
SAT_EPS = 0.003        # points this close to 0/1 are treated as saturated and dropped
Z_MAX = 0.6            # hand too high (z>this) distorts position, drop these points (1.0=keep all)
JUMP_BREAK = 0.025     # jump between adjacent kept points above this => break stroke (normal motion ~0.003, fast moves/lead-ins larger)
MIN_PTS = 8            # strokes with fewer points are dropped (debris)
RESAMPLE = 3.0         # resampling point spacing (normalized coordinates)
SMOOTH_MINCUTOFF = 1.0
SMOOTH_BETA = 0.045
FLIP_Y = True          # sensor y points up, plot y points down; flip for intuition
STITCH = False         # for recognition: keep original multi-stroke layout, no stitching (stitching connects scattered strokes with crossing lines)


def segment(t, x, y, pen, jump_break, min_pts):
    """Segment on pen-up/jumps; each stroke keeps (t, x, y), timestamps for the 1€ filter."""
    strokes, cur, last = [], [], None
    for i in range(len(x)):
        if pen[i]:
            p = (x[i], y[i])
            if last is not None and np.hypot(p[0] - last[0], p[1] - last[1]) > jump_break:
                if len(cur) >= min_pts:
                    strokes.append(np.asarray(cur))
                cur = []
            cur.append((t[i], x[i], y[i]))
            last = p
        else:
            if len(cur) >= min_pts:
                strokes.append(np.asarray(cur))
            cur, last = [], None
    if len(cur) >= min_pts:
        strokes.append(np.asarray(cur))
    return strokes


def smooth_stroke(s):
    """s: (N,3) of t,x,y. Run 1€ with the real CSV timestamps, more accurate when the sample rate fluctuates."""
    sx, sy = smooth_xy(s[:, 0], s[:, 1], s[:, 2], SMOOTH_MINCUTOFF, SMOOTH_BETA)
    return np.stack([sx, sy], axis=1)


def stitch_strokes(strokes):
    """Translate each stroke so its start joins the previous stroke's end => one continuous line."""
    if not strokes:
        return []
    out = [strokes[0]]
    for i in range(1, len(strokes)):
        shift = out[-1][-1] - strokes[i][0]
        out.append(strokes[i] + shift)
    return out


def main(path):
    tr = load_trajectory(path)
    x, y = tr.x.copy(), tr.y.copy()
    inr = tr.pen  # in the old format, in_range is loaded as pen
    n = len(x)

    valid = ~(np.isnan(x) | np.isnan(y))
    if inr is not None:
        valid = valid & (np.nan_to_num(inr) > 0.5)
    sat = (x <= SAT_EPS) | (x >= 1 - SAT_EPS) | (y <= SAT_EPS) | (y >= 1 - SAT_EPS)
    if tr.z is not None:
        sat = sat | (np.nan_to_num(tr.z) > Z_MAX)   # hand too high => position distorted, drop as saturated
    pen = valid & (~sat)

    strokes_raw = segment(tr.t, x, y, pen, JUMP_BREAK, MIN_PTS)
    strokes_sm = [smooth_stroke(s) for s in strokes_raw]
    strokes_norm = normalize_strokes(strokes_sm, target_size=255.0)
    strokes_rs = [resample_stroke(s, RESAMPLE) for s in strokes_norm]

    # Stitch into a single stroke (ignore displacement during pen-up)
    stitched = stitch_strokes(strokes_rs)
    single = [np.concatenate(stitched, axis=0)] if stitched else []

    # stroke-3 for Sketch-RNN: single continuous stroke if stitched, otherwise keep segments
    stroke3 = to_stroke3(single if STITCH else strokes_rs)

    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs("out", exist_ok=True)
    out_png = os.path.join("out", base + "_clean.png")
    out_npy = os.path.join("out", base + "_stroke3.npy")
    np.save(out_npy, stroke3)

    # ---- Visualization: raw / cleaned segments / stitched single stroke ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    yf = (lambda a: -a) if FLIP_Y else (lambda a: a)

    ax = axes[0]
    xv, yv = x[valid], y[valid]
    ax.plot(xv, yf(yv), "-", color="#d33", lw=0.8, alpha=0.9)
    ax.set_title("1) Raw capture (%.0f%% saturation noise)" % (np.mean(sat[valid]) * 100 if valid.any() else 0))
    ax.set_aspect("equal"); ax.grid(alpha=0.2)

    ax = axes[1]
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(strokes_rs):
        ax.plot(s[:, 0], yf(s[:, 1]), "-", color=cmap(i % 10), lw=2.2)
    ax.set_title("2) Cleaned + segmented (%d strokes)" % len(strokes_rs))
    ax.set_aspect("equal"); ax.grid(alpha=0.2)

    ax = axes[2]
    if single:
        sp = single[0]
        ax.plot(sp[:, 0], yf(sp[:, 1]), "-", color="#111", lw=2.2)
    ax.set_title("3) Stitched single line (one-stroke)")
    ax.set_aspect("equal"); ax.grid(alpha=0.2)

    fig.suptitle("Skywriter auto-cleanup + stitch: %s" % base)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    print("total frames:    %d" % n)
    print("valid (in range):%d" % int(valid.sum()))
    print("saturated ratio: %.1f%%" % (np.mean(sat[valid]) * 100 if valid.any() else 0))
    print("strokes:         %d" % len(strokes_rs))
    print("stroke-3 steps:  %d" % len(stroke3))
    print("plot saved:      %s" % out_png)
    print("stroke-3 saved:  %s" % out_npy)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_capture.py <capture.csv>")
        sys.exit(1)
    main(sys.argv[1])
