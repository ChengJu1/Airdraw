"""Trajectory visualization (saved as PNG for headless environments)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_pipeline(raw_x, raw_y, clean_x, clean_y, pen_down, strokes,
                  out_path: str, title: str = "Skywriter trajectory"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(raw_x, raw_y, "-", color="#c0392b", lw=0.8, alpha=0.8)
    ax.set_title("1) Raw (noisy)")
    _square(ax)

    ax = axes[1]
    cx = np.asarray(clean_x)
    cy = np.asarray(clean_y)
    down = np.asarray(pen_down, dtype=bool)
    ax.plot(cx, cy, "-", color="#bdc3c7", lw=0.6, alpha=0.6)
    seg = cx.copy()
    seg_y = cy.copy()
    seg[~down] = np.nan
    seg_y[~down] = np.nan
    ax.plot(seg, seg_y, "-", color="#2c3e50", lw=1.4)
    ax.set_title("2) Denoised + pen-down (dark)")
    _square(ax)

    ax = axes[2]
    for s in strokes:
        if len(s):
            ax.plot(s[:, 0], s[:, 1], "-", color="#2980b9", lw=1.6)
    ax.set_title("3) Extracted strokes (normalized)")
    _square(ax)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _square(ax):
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()
    ax.grid(True, ls=":", alpha=0.3)
