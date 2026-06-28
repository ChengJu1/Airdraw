"""实时滤波链(纯标准库,树莓派/电脑通用)。

web_capture.py / draw_app.py / denoise.py 共用,避免同一份滤波代码复制三遍、
参数各自漂移。部署到树莓派时把本文件和 mgc3130.py 一起拷过去。

链路: 3 点中值(MedianWin) -> 尖刺门控(SpikeGate) -> 1€ 平滑(OneEuro)
抬落笔: ZPenHysteresis(双阈值迟滞 + 连续帧去抖)
"""
from __future__ import annotations

import math
from collections import deque

# ---- 实测调好的一组默认参数(web_capture 上标定,draw_app 直接沿用) ----
MEDIAN_WIN = 3
SPIKE_MULT = 8.0          # 跳变 > mult×近期步长中位数 => 丢弃该帧
SPIKE_FLOOR = 0.002       # 正常画圆 p95 步长约 0.004；floor 过高会误杀跟手点
SPIKE_WIN = 24
ONE_EURO_MINCUTOFF = 1.0  # 比 1.8 跟手；配合 beta 抑制画圆滞后
ONE_EURO_BETA = 0.045
ONE_EURO_DCUTOFF = 1.0
Z_PEN_DOWN = 0.26         # z 低于它(连续 PEN_MIN_FRAMES 帧)落笔
Z_PEN_UP = 0.32           # z 高于它(连续 PEN_MIN_FRAMES 帧)抬笔；中间保持原状态
PEN_MIN_FRAMES = 2


class MedianWin:
    """滑动中值，去掉单帧尖刺。"""

    def __init__(self, win=MEDIAN_WIN):
        self.xs = deque(maxlen=win)
        self.ys = deque(maxlen=win)

    def reset(self):
        self.xs.clear()
        self.ys.clear()

    def __call__(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        sx = sorted(self.xs)[len(self.xs) // 2]
        sy = sorted(self.ys)[len(self.ys) // 2]
        return sx, sy


class SpikeGate:
    """自适应尖刺门控：相对近期步长中位数的异常跳变直接丢弃。

    只丢弃孤立尖刺：连续 max_reject 帧都超限说明是真实的快速移动，
    此时接受新位置重新锚定。否则锚点永远停在旧位置、之后每帧距离
    只增不减，门控会卡死（画面冻结在一个点）。
    """

    def __init__(self, mult=SPIKE_MULT, floor=SPIKE_FLOOR, win=SPIKE_WIN,
                 max_reject=3):
        self.mult = mult
        self.floor = floor
        self.max_reject = max_reject
        self.steps = deque(maxlen=win)
        self.x = None
        self.y = None
        self._rej = 0

    def reset(self):
        self.steps.clear()
        self.x = self.y = None
        self._rej = 0

    def __call__(self, x, y):
        if self.x is None:
            self.x, self.y = x, y
            return x, y
        d = math.hypot(x - self.x, y - self.y)
        med = sorted(self.steps)[len(self.steps) // 2] if len(self.steps) >= 5 else 0.003
        lim = max(self.floor, self.mult * max(med, 0.0004))
        if d > lim:
            self._rej += 1
            if self._rej < self.max_reject:
                return self.x, self.y
            # 连续超限：按真实移动重新锚定（这一大步不计入步长统计）
            self._rej = 0
            self.x, self.y = x, y
            return x, y
        self._rej = 0
        self.steps.append(d)
        self.x, self.y = x, y
        return x, y


class OneEuro:
    """1€ 滤波（Casiez et al.）：慢动作稳、快动作低延迟。

    __call__(x, t) 传时间戳则按真实 dt 自适应；t=None 时退化为固定 freq。
    mincutoff 可逐次调用覆盖（如按 z 高度动态加强平滑）。
    """

    def __init__(self, mincutoff=ONE_EURO_MINCUTOFF, beta=ONE_EURO_BETA,
                 dcutoff=ONE_EURO_DCUTOFF, freq=30.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.freq = freq
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t=None, mincutoff=None):
        mc = self.mincutoff if mincutoff is None else mincutoff
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        if t is not None and self.t_prev is not None:
            dt = t - self.t_prev
            if dt <= 0:
                dt = 1e-3
        else:
            dt = 1.0 / self.freq
        self.t_prev = t
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = mc + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class ZPenHysteresis:
    """z 高度抬落笔：双阈值迟滞 + 连续帧去抖。

    单阈值的问题：手悬在临界高度时 z 的噪声让 pen 在 0/1 间抖动，一笔断成碎段。
    这里 z < down 连续 min_frames 帧才落笔，z > up 连续 min_frames 帧才抬笔，
    中间区间保持原状态（与离线端 pen_state.pen_from_z 同思路的实时版）。
    """

    def __init__(self, down=Z_PEN_DOWN, up=Z_PEN_UP, min_frames=PEN_MIN_FRAMES):
        self.down = down
        self.up = up
        self.min_frames = min_frames
        self.state = 0
        self._cnt = 0

    def reset(self):
        self.state = 0
        self._cnt = 0

    def __call__(self, z):
        if self.state == 0:
            want = 1 if z < self.down else 0
        else:
            want = 0 if z > self.up else 1
        if want != self.state:
            self._cnt += 1
            if self._cnt >= self.min_frames:
                self.state = want
                self._cnt = 0
        else:
            self._cnt = 0
        return self.state
