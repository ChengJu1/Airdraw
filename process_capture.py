"""离线自动处理一段 Skywriter 采集 CSV：剔饱和 -> 跳变断笔分段 -> 逐笔 1€ 平滑
-> 稳健归一化 -> stroke-3，并输出「原始 vs 清理后」对比图。

用法:
    python process_capture.py cap_xxx.csv
全自动，无需手调范围/方向。
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

# ---- 参数（依据数据分布自动设的稳健默认） ----
SAT_EPS = 0.003        # 贴近 0/1 这么近视为饱和点，剔除
Z_MAX = 0.6            # 手抬太高(z>此值)位置失真，剔除这些点(1.0=不剔)
JUMP_BREAK = 0.025     # 相邻保留点跳变超过它 => 断笔(正常移动 ~0.003，快速移动/引线更大)
MIN_PTS = 8            # 一笔少于这么多点丢弃(碎屑)
RESAMPLE = 3.0         # 重采样点间距(归一化坐标)
SMOOTH_MINCUTOFF = 1.0
SMOOTH_BETA = 0.045
FLIP_Y = True          # 传感器 y 向上、绘图 y 向下，翻转以符合直觉
STITCH = False         # 识别用：保留原始多笔布局，不缝合(缝合会把分散笔画连成穿越线)


def segment(t, x, y, pen, jump_break, min_pts):
    """按抬笔/跳变分段，每笔保留 (t, x, y) 三列，时间戳供 1€ 滤波用。"""
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
    """s: (N,3) 的 t,x,y。用 CSV 里的真实时间戳跑 1€，采样率波动时更准。"""
    sx, sy = smooth_xy(s[:, 0], s[:, 1], s[:, 2], SMOOTH_MINCUTOFF, SMOOTH_BETA)
    return np.stack([sx, sy], axis=1)


def stitch_strokes(strokes):
    """把每一笔平移，使其起点接到上一笔的终点 => 连成连续的一笔画。"""
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
    inr = tr.pen  # 旧格式里 in_range 被当作 pen 载入
    n = len(x)

    valid = ~(np.isnan(x) | np.isnan(y))
    if inr is not None:
        valid = valid & (np.nan_to_num(inr) > 0.5)
    sat = (x <= SAT_EPS) | (x >= 1 - SAT_EPS) | (y <= SAT_EPS) | (y >= 1 - SAT_EPS)
    if tr.z is not None:
        sat = sat | (np.nan_to_num(tr.z) > Z_MAX)   # 手抬太高 => 位置失真，按饱和剔除
    pen = valid & (~sat)

    strokes_raw = segment(tr.t, x, y, pen, JUMP_BREAK, MIN_PTS)
    strokes_sm = [smooth_stroke(s) for s in strokes_raw]
    strokes_norm = normalize_strokes(strokes_sm, target_size=255.0)
    strokes_rs = [resample_stroke(s, RESAMPLE) for s in strokes_norm]

    # 缝合成一笔画（忽略抬笔期间的位移）
    stitched = stitch_strokes(strokes_rs)
    single = [np.concatenate(stitched, axis=0)] if stitched else []

    # 用于 Sketch-RNN 的 stroke-3：缝合后为单笔连续，否则保留分段
    stroke3 = to_stroke3(single if STITCH else strokes_rs)

    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs("out", exist_ok=True)
    out_png = os.path.join("out", base + "_clean.png")
    out_npy = os.path.join("out", base + "_stroke3.npy")
    np.save(out_npy, stroke3)

    # ---- 可视化：原始 / 清理分段 / 缝合成一笔 ----
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

    print("总帧:            %d" % n)
    print("有效(在感应区):  %d" % int(valid.sum()))
    print("饱和点占比:      %.1f%%" % (np.mean(sat[valid]) * 100 if valid.any() else 0))
    print("分出笔画:        %d 笔" % len(strokes_rs))
    print("stroke-3 步数:   %d" % len(stroke3))
    print("对比图已保存:    %s" % out_png)
    print("stroke-3 已保存: %s" % out_npy)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python process_capture.py <capture.csv>")
        sys.exit(1)
    main(sys.argv[1])
