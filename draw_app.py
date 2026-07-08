#!/usr/bin/env python3
"""树莓派独立全屏作画程序（展览主程序，无需电脑/浏览器/键盘）。

直接驱动 HDMI 全屏(Pygame)，读 Skywriter，实时处理(范围校准 / 中值+尖刺门控+1€ 平滑 /
出界+跳变断笔)，并录制 CSV。滤波链与 web_capture.py 共用 rt_filters.py。

展览闭环：
  画画 -> 停笔且手离开 AI_WAIT_SEC 秒 -> 自动送 AI 识别(Gemini 看图) ->
  文生图重建(Pollinations 免费) -> 全屏对比展示(你的涂鸦 vs AI 重建) ->
  RESULT_SEC 秒后清屏等下一位。没网/没 key 时自动降级为纯画画+自动清屏。

展览 UI / 交互：
  - 深色背景，传感器范围固定映射到屏幕中央画布（手在哪、光标就在哪）
  - 实时光标：落笔=实心亮点，抬笔=空心圆环
  - 右下角小地图：传感器 XY 感应范围 + 出界抬笔区 + 当前位置点 + Z 高度计
  - 手势：快速划过=清屏(停笔片刻后生效)；颜色每抬一笔自动轮换
  - 声音反馈：落笔/抬笔/清屏/换色/识别完成；无声卡自动静音
  - 传感器 I2C 自愈：启动失败/中途掉线不退出，硬件复位循环重连并在屏幕提示
  - 键盘(调试用)：ESC/Q 退出  空格/C 清屏  回车 立即送AI  D 调试
    方向校准(实时生效): X 左右镜像  Y 上下镜像  S 交换XY轴

依赖(在已装 lgpio/smbus2 的同一 venv 内): pip install pygame
AI 部分走 REST(urllib)，无需 google-genai/PIL。密钥读 key_local.py 或环境变量
GEMINI_API_KEY —— 需连同 mgc3130.py、rt_filters.py、key_local.py 拷到同一目录。
开机自启+崩溃自动重启: 见 skywriter-draw.service（装法见该文件头部注释）。
手动运行: ~/start_draw.sh
"""
import array
import base64
import csv
import io
import json
import math
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
import atexit

from mgc3130 import MGC3130, parse_frame, parse_gesture, probe_i2c, FLICKS
from rt_filters import MedianWin, SpikeGate, OneEuro, ZPenHysteresis

# ============== 处理参数（滤波链参数在 rt_filters.py，与 web_capture.py 共用） ==============
FLIP_X = False
FLIP_Y = True
SWAP_XY = False
X_LO, X_HI = 0.08, 0.92         # 实测(cal_20260705)两端各~8%饱和贴轨，只用中段线性区
Y_LO, Y_HI = 0.08, 0.92
OUT_EDGE = 0.08                 # 贴边即抬笔；与 LO/HI 对齐，饱和区(钳0/1)不画
# z 迟滞判抬落笔：z > Z_HIGH_UP 抬笔，回落 < Z_HIGH_DOWN 恢复
# 实测自然画画高度 z≈0.2~0.5：落笔阈值不能低于 0.5，否则画画高度上半段贴着
# 阈值边界会碎成断点(0.45 试过，断触严重)
Z_HIGH_UP = 0.60
Z_HIGH_DOWN = 0.50
MAX_JUMP = 0.07                 # 快速移动的引线判为抬笔，不画出来(放宽减少误断)
AUTO_CLEAR_SEC = 8.0            # (AI 不可用时的兜底)手离开这么久自动清屏

# ============== 手势 & 声音 ==============
FLICK_CLEARS = True             # 快速划过 = 清屏
FLICK_DEBOUNCE = 1.2            # 两次 flick 最小间隔(秒)
GESTURE_IDLE = 0.5              # 停笔这么久后手势才解锁，画画途中不误触
SOUND = True                    # 声音反馈总开关

# ============== AI 识别与重建（流程同 reconstruct_llm.py，纯 urllib 版） ==============
AI_WAIT_SEC = 3.0               # 停笔且手离开这么久 => 自动送 AI
RESULT_SEC = 15.0               # 结果展示时长，之后清屏等下一位
MIN_PTS_FOR_AI = 12             # 总点数太少(误触)不送 AI
G_VISION_MODEL = "gemini-2.5-flash"
SYS_PROMPT = (
    "你是手绘草图识别助手。用户给你一张非常粗糙的单色简笔画轨迹，"
    "可能是单个物体，也可能是几样东西拼成的一个场景。"
    "请判断画的是什么，然后写一段用于文生图模型的英文提示词，"
    "目标是生成一张干净、可识别、风格统一的插画；"
    "若有多个元素，把它们自然地组合进同一张场景图，保持原画的相对布局。"
    '只输出 JSON：{"label": "short english name", "prompt": "english text-to-image prompt"}。'
    "label 和 prompt 都用英文，label 不超过 4 个词。提示词里加上 "
    "'simple, clean line illustration, white background, centered'。"
)

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
SF_KEY = os.getenv("SILICONFLOW_API_KEY", "")
REPLICATE_KEY = os.getenv("REPLICATE_API_TOKEN", "")
try:
    import key_local as _k
    GEMINI_KEY = GEMINI_KEY or getattr(_k, "GEMINI_API_KEY", "")
    SF_KEY = SF_KEY or getattr(_k, "SILICONFLOW_API_KEY", "")
    REPLICATE_KEY = REPLICATE_KEY or getattr(_k, "REPLICATE_API_TOKEN", "")
except ImportError:
    pass
AI_ENABLED = bool(SF_KEY or GEMINI_KEY)
# 主方案：Gemini 付费层一个 key 全包(识别 gemini-2.5-flash + 出图 nano banana)。
# 注意免费层不够用：识别每天仅 20 次、出图额度为 0，务必开计费。
# 识别顺序：硅基流动(如有key) -> Gemini
# 文生图顺序：Gemini -> Replicate -> 硅基流动 -> Pollinations(免费兜底)
G_IMAGE_MODEL = "gemini-2.5-flash-image"
SF_VISION_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
SF_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

# ============== 展示外观 ==============
FILL = 0.80                     # 画布占屏幕短边比例
BG = (10, 12, 18)               # 全屏底色
CANVAS_BG = (17, 20, 30)        # 画布底色
CANVAS_BORDER = (54, 62, 88)
TXT = (225, 230, 240)
TXT_DIM = (108, 118, 142)
GREEN = (90, 220, 160)          # 落笔/可画状态
AMBER = (255, 190, 90)          # 抬笔/边界警示
LINE_W = 5
INK_PALETTE = [                 # 每抬一笔自动轮换一种颜色
    (240, 242, 248),            # 亮白
    (120, 200, 255),            # 天蓝
    (140, 235, 190),            # 薄荷
    (250, 200, 120),            # 琥珀
    (210, 160, 255),            # 薰衣草
    (255, 150, 165),            # 珊瑚
]
# ===================================================================

# ---- 滤波链(实现与参数见 rt_filters.py) ----
med_f = MedianWin()
spike = SpikeGate()
fx = OneEuro()
fy = OneEuro()
zpen = ZPenHysteresis(down=Z_HIGH_DOWN, up=Z_HIGH_UP)
# 光标悬停平滑（独立于画线滤波，抬笔时也持续工作）
hx = OneEuro()
hy = OneEuro()


def reset_filters():
    med_f.reset()
    spike.reset()
    fx.reset()
    fy.reset()


def stretch(v, lo, hi):
    return v if hi - lo < 1e-6 else (v - lo) / (hi - lo)


def lerp_color(a, b, f):
    return (int(a[0] + (b[0] - a[0]) * f),
            int(a[1] + (b[1] - a[1]) * f),
            int(a[2] + (b[2] - a[2]) * f))


# ---- 声音（程序化合成，不依赖音频文件；无声卡则静音） ----
SND = {}


def snd(name):
    if SOUND and name in SND:
        try:
            SND[name].play()
        except Exception:
            pass


def build_sounds():
    """启动时合成几段短音效。mixer 没起来(无声卡)返回空。"""
    import pygame
    if not pygame.mixer.get_init():
        return {}
    sr = 22050

    def tone(freq, ms, vol=0.4, freq2=None):
        n = int(sr * ms / 1000)
        a = array.array('h')
        for i in range(n):
            f = freq if freq2 is None else freq + (freq2 - freq) * i / n
            env = 1.0 - i / n
            a.append(int(32767 * vol * env * math.sin(2 * math.pi * f * i / sr)))
        return pygame.mixer.Sound(buffer=a.tobytes())

    def arp(freqs, ms=95, vol=0.38):
        a = array.array('h')
        n = int(sr * ms / 1000)
        for f in freqs:
            for i in range(n):
                env = 1.0 - i / n
                a.append(int(32767 * vol * env * math.sin(2 * math.pi * f * i / sr)))
        return pygame.mixer.Sound(buffer=a.tobytes())

    return {
        "down": tone(880, 45, 0.5),            # 落笔"嗒"
        "up": tone(520, 35, 0.22),             # 抬笔轻响
        "clear": tone(700, 230, 0.4, freq2=170),   # 清屏下滑音
        "tick": tone(1320, 25, 0.3),           # 换色咔哒
        "success": arp([523, 659, 784, 1046]),  # 识别完成小琶音
        "fail": tone(170, 260, 0.35),          # AI 失败低鸣
    }


# ---- CSV ----
name = sys.argv[1] if len(sys.argv) > 1 else time.strftime("app_%Y%m%d_%H%M%S")
outdir = os.path.expanduser("~/captures"); os.makedirs(outdir, exist_ok=True)
csv_path = os.path.join(outdir, name + ".csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["t", "x", "y", "z", "in_range", "pen"])


@atexit.register
def _close():
    try:
        csv_file.flush(); csv_file.close()
    except Exception:
        pass


# ---- 共享绘画/光标/传感器状态 ----
lock = threading.Lock()
strokes = [[0, []]]            # 每元素 [调色板序号, 点列表(0..1)]
color_idx = 0                  # 当前画笔颜色(AirWheel 转圈 / 抬笔自动轮换)
pen_now = 0
last_activity = time.time()    # 最近一次检测到手的时刻
draw_last = 0.0                # 最近一次真正落笔画点的时刻(AI 触发依据)
raw_range = [1.0, 0.0, 1.0, 0.0]   # mnx,mxx,mny,mxy (调试用)
t0 = time.time()

cur_x, cur_y = 0.5, 0.5        # 光标位置(0..1，画布坐标)
cur_z = 1.0                    # 当前手高度(0..1)
cur_seen = 0.0                 # 最近一次有效帧时刻
prev_pen = 0
gest_dbg = [0, 0, 0]           # 最近的手势码 / airwheel活跃 / 计数(调试栏显示)

sensor = None                  # 由 reader 线程创建/重建（I2C 自愈）
sensor_ok = False

# ---- AI 状态(主循环与 worker 线程共享) ----
ai_lock = threading.Lock()
ai = {
    "mode": "draw",            # draw / thinking / result
    "token": 0,                # 每次触发/取消 +1，worker 结果过期即丢弃
    "label": "",
    "img": None,               # 生成图 Surface
    "sketch": None,            # 送识别的简笔画 Surface(白底黑线)
    "t0": 0.0,                 # 进入 result 的时刻
    "err": "",
    "err_t": 0.0,
    "_cache": None,            # result 缩放缓存
}


def clear_canvas():
    global strokes, prev_pen, draw_last
    with lock:
        strokes = [[color_idx, []]]
    prev_pen = 0
    draw_last = 0.0


def reader():
    """传感器读取线程：I2C 自愈 + 手势 + 滤波成笔画。"""
    global sensor, sensor_ok
    global pen_now, last_activity, draw_last, prev_pen, color_idx
    global cur_x, cur_y, cur_z, cur_seen
    last_ts = None
    last_draw = None
    last_flick = 0.0
    z_hist = []                # z 三点中值窗，压掉单帧尖刺引起的误断笔
    err_streak = 0
    last_frame_t = time.time()
    cnt = 0

    def reconnect():
        """关掉旧句柄，硬件复位重连。失败慢速重试，程序不退出。"""
        nonlocal err_streak, last_frame_t
        global sensor, sensor_ok
        sensor_ok = False
        if sensor is not None:
            try:
                sensor.close()
            except Exception:
                pass
            sensor = None
        try:
            sensor = MGC3130(retries=2)
            sensor_ok = True
            err_streak = 0
            last_frame_t = time.time()
            print("传感器已连接")
        except Exception as exc:  # noqa: BLE001
            print("传感器重连失败(2秒后再试): %r" % (exc,))
            time.sleep(2.0)

    while True:
      try:
        if sensor is None:
            reconnect()
            continue
        if sensor.data_ready():
            d = sensor.read_frame()
            if d is None:
                err_streak += 1
                if err_streak > 100:
                    reconnect()
                time.sleep(0.002)
                continue
            if len(d) >= 26 and d[3] == 0x91:
                ts = d[6]
                if ts != last_ts:
                    last_ts = ts
                    err_streak = 0
                    last_frame_t = time.time()
                    sensor_ok = True
                    t = time.time() - t0
                    rx, ry, rz, valid = parse_frame(d)
                    gest, aw_active, aw_count = parse_gesture(d)

                    # 手势：停笔 GESTURE_IDLE 秒后解锁，画画途中不误触
                    now = time.time()
                    gest_dbg[0], gest_dbg[1], gest_dbg[2] = gest, int(aw_active), aw_count
                    gesture_ok = (draw_last == 0 or now - draw_last > GESTURE_IDLE)

                    # 快速划过 = 清屏(也会跳过结果页)
                    if FLICK_CLEARS and gest in FLICKS and gesture_ok and \
                            now - last_flick > FLICK_DEBOUNCE:
                        last_flick = now
                        ai_cancel()
                        clear_canvas()
                        snd("clear")
                        print("手势: 划过清屏 (code=%d)" % gest)

                    if valid:
                        raw_range[0] = min(raw_range[0], rx); raw_range[1] = max(raw_range[1], rx)
                        raw_range[2] = min(raw_range[2], ry); raw_range[3] = max(raw_range[3], ry)

                        # z 先过三点中值再进迟滞，单帧尖刺不再造成误断笔
                        z_hist.append(rz)
                        if len(z_hist) > 3:
                            z_hist.pop(0)
                        zf = sorted(z_hist)[len(z_hist) // 2]

                        # z 迟滞判抬落笔；边缘饱和 => 强制抬笔
                        edge_out = (rx < OUT_EDGE or rx > 1 - OUT_EDGE or
                                    ry < OUT_EDGE or ry > 1 - OUT_EDGE)
                        pen = zpen(zf)
                        if edge_out:
                            pen = 0
                            zpen.reset()

                        ax, ay = (ry, rx) if SWAP_XY else (rx, ry)
                        mx = stretch(ax, X_LO, X_HI)
                        my = stretch(ay, Y_LO, Y_HI)
                        if FLIP_X:
                            mx = 1.0 - mx
                        if FLIP_Y:
                            my = 1.0 - my
                        mx = min(1.0, max(0.0, mx)); my = min(1.0, max(0.0, my))

                        # 悬停光标：无论抬落笔都持续平滑，观众始终能看到"我在哪"
                        hvx = hx(mx, t); hvy = hy(my, t)

                        sx = sy = None
                        if not pen:
                            if prev_pen:
                                reset_filters()
                            last_draw = None
                        else:
                            if not prev_pen:
                                reset_filters()
                            mx, my = med_f(mx, my)
                            mx, my = spike(mx, my)
                            sx = fx(mx, t); sy = fy(my, t)
                            # 跳变保护：快速重定位不连长线
                            if last_draw is not None and \
                                    math.hypot(sx - last_draw[0], sy - last_draw[1]) > MAX_JUMP:
                                pen = 0
                                reset_filters()
                                last_draw = None
                            else:
                                last_draw = (sx, sy)

                        if pen != prev_pen:
                            snd("down" if pen else "up")

                        if pen and sx is not None:
                            with lock:
                                if not strokes:
                                    strokes.append([color_idx, []])
                                if not strokes[-1][1]:      # 本笔第一个点定颜色
                                    strokes[-1][0] = color_idx
                                strokes[-1][1].append((sx, sy))
                            cur_x, cur_y = sx, sy
                            draw_last = time.time()
                        else:
                            with lock:
                                if strokes and strokes[-1][1]:
                                    # 抬笔=开新一笔，颜色自动走一格(AirWheel 可再调)
                                    color_idx = (color_idx + 1) % len(INK_PALETTE)
                                    strokes.append([color_idx, []])
                            cur_x, cur_y = hvx, hvy
                        cur_z = rz
                        cur_seen = time.time()
                        last_activity = time.time()
                        pen_now = pen
                        prev_pen = pen
                        csv_writer.writerow(["%.4f" % t, "%.5f" % rx, "%.5f" % ry,
                                             "%.5f" % rz, 1, pen])
                    else:
                        if prev_pen:
                            snd("up")
                        pen_now = 0
                        prev_pen = 0
                        zpen.reset()
                        reset_filters()
                        hx.reset()
                        hy.reset()
                        z_hist.clear()
                        last_draw = None
                        with lock:
                            if strokes and strokes[-1][1]:
                                color_idx = (color_idx + 1) % len(INK_PALETTE)
                                strokes.append([color_idx, []])
                        csv_writer.writerow(["%.4f" % t, "", "", "", 0, 0])
                    cnt += 1
                    if cnt % 50 == 0:
                        csv_file.flush()
        else:
            # 长时间无帧不一定是挂死：空闲(无人)时芯片可能不出帧。
            # 先探测总线，芯片还应答就只是空闲；真掉线才复位重连。
            if time.time() - last_frame_t > 3.0:
                if probe_i2c():
                    last_frame_t = time.time()
                    sensor_ok = True
                else:
                    print("传感器无数据且总线无应答，复位重连")
                    reconnect()
        time.sleep(0.001)
      except OSError:
        # 偶发 I2C / GPIO 错误：跳过；连续太多则重连
        err_streak += 1
        if err_streak > 100:
            reconnect()
        time.sleep(0.002)
      except Exception as exc:  # noqa: BLE001
        print("reader 异常(已跳过): %r" % (exc,))
        err_streak += 1
        if err_streak > 100:
            reconnect()
        time.sleep(0.002)


# ================== AI：简笔画渲染 / 识别 / 文生图 ==================
def sketch_png(snap, px=512):
    """把笔画快照渲染成白底黑线 PNG(bytes) + Surface，居中自适应缩放。"""
    import pygame
    surf = pygame.Surface((px, px))
    surf.fill((255, 255, 255))
    pts = [p for _, s in snap for p in s]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    a, b, c, d = min(xs), max(xs), min(ys), max(ys)
    sc = px * 0.84 / max(b - a, d - c, 1e-3)
    cx = (a + b) / 2.0; cy = (c + d) / 2.0
    for _, s in snap:
        proj = [(px / 2 + (x - cx) * sc, px / 2 + (y - cy) * sc) for x, y in s]
        if len(proj) >= 2:
            pygame.draw.lines(surf, (0, 0, 0), False, proj, 6)
            for q in proj:
                pygame.draw.circle(surf, (0, 0, 0), (int(q[0]), int(q[1])), 3)
    buf = io.BytesIO()
    pygame.image.save(surf, buf, "sketch.png")
    return buf.getvalue(), surf


def _json_from_text(txt):
    """从模型回复里抠出 JSON(容忍 ```json 围栏和前后废话)。"""
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("回复里没有 JSON: %r" % txt[:120])
    return json.loads(txt[i:j + 1])


def sf_recognize(png_bytes):
    """硅基流动 Qwen-VL 看图识别，返回 {"label":..., "prompt":...}。"""
    b64 = base64.b64encode(png_bytes).decode()
    body = {
        "model": SF_VISION_MODEL,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + b64}},
                {"type": "text", "text": "这是粗糙轨迹，请识别并给出提示词。"},
            ]},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }
    req = urllib.request.Request(
        "https://api.siliconflow.cn/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + SF_KEY})
    with urllib.request.urlopen(req, timeout=40) as r:
        resp = json.load(r)
    return _json_from_text(resp["choices"][0]["message"]["content"])


def recognize(png_bytes):
    """识别：硅基流动优先，失败退 Gemini(免费层每天仅20次)。"""
    if SF_KEY:
        try:
            return sf_recognize(png_bytes)
        except Exception as exc:  # noqa: BLE001
            print("硅基流动识别失败，试 Gemini: %r" % (exc,))
    return gemini_recognize(png_bytes)


def gemini_recognize(png_bytes):
    """Gemini REST 看图识别，返回 {"label":..., "prompt":...}。"""
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "%s:generateContent?key=%s" % (G_VISION_MODEL, GEMINI_KEY))
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/png",
                             "data": base64.b64encode(png_bytes).decode()}},
            {"text": SYS_PROMPT + "\n这是粗糙轨迹，请识别并给出提示词。"},
        ]}],
        "generationConfig": {"response_mime_type": "application/json"},
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.load(r)
    return json.loads(resp["candidates"][0]["content"]["parts"][0]["text"])


def gemini_image(prompt):
    """Gemini 文生图(gemini-2.5-flash-image，需付费层)，返回图片 bytes。"""
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "%s:generateContent?key=%s" % (G_IMAGE_MODEL, GEMINI_KEY))
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.load(r)
    for p in resp["candidates"][0]["content"]["parts"]:
        if "inlineData" in p:
            return base64.b64decode(p["inlineData"]["data"])
    raise ValueError("Gemini 没返回图片 part")


def replicate_image(prompt):
    """Replicate FLUX.1 schnell 文生图(Prefer:wait 同步返回)，返回图片 bytes。"""
    body = {"input": {"prompt": prompt, "aspect_ratio": "1:1",
                      "output_format": "jpg", "num_outputs": 1}}
    req = urllib.request.Request(
        "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + REPLICATE_KEY,
                 "Prefer": "wait"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.load(r)
    out = resp.get("output")
    url = out[0] if isinstance(out, list) else out
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def sf_image(prompt):
    """硅基流动 FLUX.1 schnell 文生图，1-3 秒出图，返回图片 bytes。

    接口先返回图片 URL(有效期 1 小时)，再下载成 bytes。
    """
    body = {"model": SF_IMAGE_MODEL, "prompt": prompt,
            "image_size": "768x768", "batch_size": 1,
            "num_inference_steps": 4}
    req = urllib.request.Request(
        "https://api.siliconflow.cn/v1/images/generations",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + SF_KEY})
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.load(r)
    url = resp["images"][0]["url"]
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def pollinations_image(prompt):
    """免费文生图(无需 key)，慢但兜底，返回图片 bytes。"""
    url = ("https://image.pollinations.ai/prompt/"
           + urllib.parse.quote(prompt)
           + "?width=768&height=768&nologo=true&model=flux")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def generate_image(prompt):
    """按 Gemini -> Replicate -> 硅基流动 -> Pollinations 顺序尝试，有 key 就试。"""
    for name, key, fn in (("Gemini", GEMINI_KEY, gemini_image),
                          ("Replicate", REPLICATE_KEY, replicate_image),
                          ("硅基流动", SF_KEY, sf_image)):
        if key:
            try:
                return fn(prompt)
            except Exception as exc:  # noqa: BLE001
                print("%s 生成失败，试下一家: %r" % (name, exc))
    return pollinations_image(prompt)


def ai_worker(snap, token):
    """后台线程：渲染 -> 识别 -> 生成。token 过期(访客又画了/清屏)则丢弃结果。"""
    import pygame
    try:
        png, surf = sketch_png(snap)
        info = recognize(png)
        label = str(info.get("label", "?"))
        prompt = info.get("prompt") or (
            "simple, clean line illustration of a %s, white background, centered" % label)
        img_bytes = generate_image(prompt)
        hint = "img.png" if img_bytes[:4] == b"\x89PNG" else "img.jpg"
        img = pygame.image.load(io.BytesIO(img_bytes), hint)
        with ai_lock:
            if ai["token"] == token:
                ai.update(mode="result", label=label, img=img, sketch=surf,
                          t0=time.time(), err="", _cache=None)
                snd("success")
    except Exception as exc:  # noqa: BLE001
        print("AI 失败(降级继续画画): %r" % (exc,))
        with ai_lock:
            if ai["token"] == token:
                ai.update(mode="draw", err=str(exc), err_t=time.time())
                snd("fail")


def ai_cancel():
    with ai_lock:
        ai["token"] += 1
        ai["mode"] = "draw"
        ai["_cache"] = None


def ai_trigger(snap):
    with ai_lock:
        ai["token"] += 1
        ai["mode"] = "thinking"
        token = ai["token"]
    threading.Thread(target=ai_worker, args=(snap, token), daemon=True).start()


# ============================== 主程序 ==============================
def main():
    global FLIP_X, FLIP_Y, SWAP_XY
    import pygame
    pygame.mixer.pre_init(22050, -16, 1, 512)
    pygame.init()
    try:
        SND.update(build_sounds())
    except Exception as exc:  # noqa: BLE001
        print("声音初始化失败(静音继续): %r" % (exc,))
    pygame.mouse.set_visible(False)
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    W, H = screen.get_size()
    clock = pygame.time.Clock()
    f_big = pygame.font.SysFont(None, 64)
    f_mid = pygame.font.SysFont(None, 30)
    f_sml = pygame.font.SysFont(None, 21)

    # 画布：传感器范围固定映射到屏幕中央正方形
    side = int(min(W, H) * FILL)
    canvas = pygame.Rect((W - side) // 2, (H - side) // 2, side, side)

    def to_px(p):
        return (canvas.left + p[0] * side, canvas.top + p[1] * side)

    # 小地图布局（右下角）
    MM = max(120, int(min(W, H) * 0.16))       # 小地图边长
    GW = 16                                    # 高度计宽
    PAD = 14
    panel = pygame.Rect(0, 0, MM + GW + PAD * 3, MM + PAD * 2 + 26)
    panel.bottomright = (W - 28, H - 28)
    mm_rect = pygame.Rect(panel.left + PAD, panel.top + PAD + 22, MM, MM)
    gauge = pygame.Rect(mm_rect.right + PAD, mm_rect.top, GW, MM)

    # 手势说明 + 调色板（左下角）
    hint_panel = pygame.Rect(0, 0, 300, panel.height)
    hint_panel.bottomleft = (28, H - 28)

    threading.Thread(target=reader, daemon=True).start()

    trigger_draw_last = -1.0    # 触发 AI 时的 draw_last，之后又画了就取消
    show_debug = False
    running = True
    while running:
        now = time.time()
        with ai_lock:
            mode = ai["mode"]

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key in (pygame.K_SPACE, pygame.K_c):
                    ai_cancel()
                    clear_canvas()
                    snd("clear")
                elif ev.key == pygame.K_d:
                    show_debug = not show_debug
                elif ev.key == pygame.K_x:      # 方向校准，实时生效
                    FLIP_X = not FLIP_X
                elif ev.key == pygame.K_y:
                    FLIP_Y = not FLIP_Y
                elif ev.key == pygame.K_s:
                    SWAP_XY = not SWAP_XY
                elif ev.key == pygame.K_RETURN and AI_ENABLED and mode == "draw":
                    with lock:
                        snap_now = [(ci, list(s)) for ci, s in strokes
                                    if len(s) >= 2]
                    if sum(len(s) for _, s in snap_now) >= MIN_PTS_FOR_AI:
                        trigger_draw_last = draw_last
                        ai_trigger(snap_now)
                        mode = "thinking"

        hand = (now - cur_seen) < 0.35          # 最近有有效帧 => 手在感应区
        idle = now - last_activity

        with lock:
            snap = [(ci, list(s)) for ci, s in strokes if len(s) >= 2]
            has_content = any(s for _, s in strokes)
        n_pts = sum(len(s) for _, s in snap)

        # ---------- 状态机 ----------
        if mode == "draw":
            # 停笔+手离开 => 自动送 AI；AI 不可用则走兜底自动清屏
            if AI_ENABLED and n_pts >= MIN_PTS_FOR_AI and not hand \
                    and draw_last > 0 and now - draw_last > AI_WAIT_SEC:
                trigger_draw_last = draw_last
                ai_trigger(snap)
                mode = "thinking"
            elif has_content and idle > AUTO_CLEAR_SEC and \
                    (not AI_ENABLED or n_pts < MIN_PTS_FOR_AI):
                clear_canvas()
        elif mode == "thinking":
            if draw_last != trigger_draw_last:   # 访客回来又画了 => 取消本次
                ai_cancel()
                mode = "draw"
        elif mode == "result":
            with ai_lock:
                shown = now - ai["t0"]
            if shown > RESULT_SEC or pen_now:    # 到时 / 有人开始画 => 清屏重来
                ai_cancel()
                clear_canvas()
                mode = "draw"

        # ---------- 结果展示页 ----------
        if mode == "result":
            with ai_lock:
                label, img, sk = ai["label"], ai["img"], ai["sketch"]
                cache = ai["_cache"]
            screen.fill(BG)
            ph = int(min(W, H) * 0.52)
            gap = int(ph * 0.08)
            if cache is None or cache[0] != ph:
                img_s = pygame.transform.smoothscale(img.convert(), (ph, ph))
                sk_s = pygame.transform.smoothscale(sk, (ph, ph))
                cache = (ph, sk_s, img_s)
                with ai_lock:
                    ai["_cache"] = cache
            _, sk_s, img_s = cache
            x0 = W // 2 - ph - gap // 2
            x1 = W // 2 + gap // 2
            y0 = H // 2 - ph // 2
            title = f_big.render('Looks like:  %s' % label.upper(), True, TXT)
            screen.blit(title, title.get_rect(center=(W // 2, y0 - 64)))
            for x, surf_i, cap in ((x0, sk_s, "YOUR SKETCH"),
                                   (x1, img_s, "AI RECONSTRUCTION")):
                r = pygame.Rect(x, y0, ph, ph)
                screen.blit(surf_i, r)
                pygame.draw.rect(screen, CANVAS_BORDER, r, width=2, border_radius=2)
                c = f_sml.render(cap, True, TXT_DIM)
                screen.blit(c, c.get_rect(midtop=(r.centerx, r.bottom + 10)))
            remain = max(0, int(RESULT_SEC - (now - ai["t0"]) + 0.99))
            tip = f_mid.render("wave to draw again  ·  flick to skip  ·  %d" % remain,
                               True, TXT_DIM)
            screen.blit(tip, tip.get_rect(midbottom=(W // 2, H - 26)))
            pygame.display.flip()
            clock.tick(60)
            continue

        # ---------- 背景 & 画布 ----------
        screen.fill(BG)
        pygame.draw.rect(screen, CANVAS_BG, canvas, border_radius=14)
        pygame.draw.rect(screen, CANVAS_BORDER, canvas, width=2, border_radius=14)

        # 画布中心十字：给观众一个"原点"参照
        ccx, ccy = canvas.center
        cross = lerp_color(CANVAS_BG, TXT_DIM, 0.55)
        pygame.draw.line(screen, cross, (ccx - 12, ccy), (ccx + 12, ccy))
        pygame.draw.line(screen, cross, (ccx, ccy - 12), (ccx, ccy + 12))

        # ---------- 笔画（辉光两遍：粗暗晕 + 亮细芯） ----------
        screen.set_clip(canvas)
        for ci, s in snap:
            color = INK_PALETTE[ci % len(INK_PALETTE)]
            proj = [to_px(p) for p in s]
            halo = lerp_color(CANVAS_BG, color, 0.25)
            pygame.draw.lines(screen, halo, False, proj, LINE_W + 8)
            pygame.draw.lines(screen, color, False, proj, LINE_W)
            r = LINE_W // 2
            for px, py in proj:                 # 圆头关节，转角不豁口
                pygame.draw.circle(screen, color, (int(px), int(py)), r)

        # ---------- 实时光标 ----------
        if hand:
            cx, cy = to_px((cur_x, cur_y))
            cx, cy = int(cx), int(cy)
            pulse = 0.5 + 0.5 * math.sin(now * 5.0)
            ink = INK_PALETTE[color_idx % len(INK_PALETTE)]
            if pen_now:
                pygame.draw.circle(screen, lerp_color(CANVAS_BG, ink, 0.35),
                                   (cx, cy), 14 + int(4 * pulse))
                pygame.draw.circle(screen, ink, (cx, cy), 7)
            else:
                pygame.draw.circle(screen, AMBER, (cx, cy),
                                   11 + int(3 * pulse), width=2)
                pygame.draw.circle(screen, ink, (cx, cy), 3)
        screen.set_clip(None)

        # ---------- 传感器离线提示 ----------
        if not sensor_ok:
            pulse = 0.5 + 0.5 * math.sin(now * 3.0)
            t1 = f_big.render("SENSOR RECONNECTING", True,
                              lerp_color(TXT_DIM, AMBER, pulse))
            t2 = f_mid.render("please wait / check the Skywriter board", True, TXT_DIM)
            screen.blit(t1, t1.get_rect(center=(ccx, ccy - 24)))
            screen.blit(t2, t2.get_rect(center=(ccx, ccy + 28)))
        # ---------- 待机引导（无人且画布为空） ----------
        elif not hand and not has_content:
            for k in (0.0, 0.5):
                ph_ = ((now * 0.45 + k) % 1.0)
                rr = int(20 + ph_ * 90)
                col = lerp_color(BG, GREEN, max(0.0, 0.5 * (1.0 - ph_)))
                pygame.draw.circle(screen, col, (ccx, ccy), rr, width=2)
            t1 = f_big.render("WAVE TO DRAW", True, TXT)
            t2 = f_mid.render("hold your hand above the sensor", True, TXT_DIM)
            screen.blit(t1, t1.get_rect(center=(ccx, ccy - 90)))
            screen.blit(t2, t2.get_rect(center=(ccx, ccy + 78)))

        # ---------- AI 识别中提示 ----------
        if mode == "thinking":
            dots = "." * (1 + int(now * 2) % 3)
            t1 = f_big.render("Recognizing%s" % dots, True, TXT)
            t2 = f_mid.render("the AI is looking at your sketch", True, TXT_DIM)
            screen.blit(t1, t1.get_rect(center=(ccx, canvas.top + 56)))
            screen.blit(t2, t2.get_rect(center=(ccx, canvas.top + 100)))

        # ---------- 顶部标题 & 状态 ----------
        title = f_mid.render("S K Y W R I T E R", True, TXT_DIM)
        screen.blit(title, title.get_rect(midtop=(W // 2, 16)))

        if not sensor_ok:
            dot_c, msg = AMBER, "sensor offline - reconnecting"
        elif pen_now:
            dot_c, msg = GREEN, "DRAWING"
        elif hand:
            dot_c, msg = AMBER, "PEN UP  -  lower your hand to draw"
        else:
            dot_c, msg = TXT_DIM, "show your hand above the sensor"
        pygame.draw.circle(screen, dot_c, (30, 30), 7)
        screen.blit(f_mid.render(msg, True, TXT), (46, 19))

        # AI 触发倒计时 / 失败提示
        if AI_ENABLED and mode == "draw" and n_pts >= MIN_PTS_FOR_AI \
                and not hand and draw_last > 0:
            left_s = AI_WAIT_SEC - (now - draw_last)
            if 0 < left_s <= AI_WAIT_SEC:
                tip = f_mid.render("done? sending to AI in %d..."
                                   % max(1, int(left_s + 0.99)), True, TXT_DIM)
                screen.blit(tip, tip.get_rect(midbottom=(W // 2, canvas.bottom - 10)))
        with ai_lock:
            err, err_t = ai["err"], ai["err_t"]
        if err and now - err_t < 5.0:
            tip = f_sml.render("AI unavailable - keep drawing", True, AMBER)
            screen.blit(tip, tip.get_rect(midbottom=(W // 2, canvas.bottom - 10)))

        # ---------- 左下角：手势说明 + 调色板 ----------
        pygame.draw.rect(screen, CANVAS_BG, hint_panel, border_radius=10)
        pygame.draw.rect(screen, CANVAS_BORDER, hint_panel, width=1, border_radius=10)
        lab = f_sml.render("GESTURES", True, TXT_DIM)
        screen.blit(lab, (hint_panel.left + PAD, hint_panel.top + 7))
        h1 = f_sml.render("fast swipe (hand raised)  =  clear", True, TXT)
        h2 = f_sml.render("every new stroke  =  new colour", True, TXT)
        screen.blit(h1, (hint_panel.left + PAD, hint_panel.top + 32))
        screen.blit(h2, (hint_panel.left + PAD, hint_panel.top + 56))
        # 调色板：当前色放大加环
        sw_y = hint_panel.bottom - 28
        for i, c in enumerate(INK_PALETTE):
            sw_x = hint_panel.left + PAD + 10 + i * 34
            if i == color_idx % len(INK_PALETTE):
                pygame.draw.circle(screen, c, (sw_x, sw_y), 11)
                pygame.draw.circle(screen, TXT, (sw_x, sw_y), 14, width=2)
            else:
                pygame.draw.circle(screen, lerp_color(CANVAS_BG, c, 0.6),
                                   (sw_x, sw_y), 8)

        # ---------- 右下角：传感器范围小地图 + 高度计 ----------
        pygame.draw.rect(screen, CANVAS_BG, panel, border_radius=10)
        pygame.draw.rect(screen, CANVAS_BORDER, panel, width=1, border_radius=10)
        lab = f_sml.render("SENSOR RANGE", True, TXT_DIM)
        screen.blit(lab, (panel.left + PAD, panel.top + 7))

        # 外框=全部感应范围(边缘为抬笔区)，内框=可作画区
        pygame.draw.rect(screen, lerp_color(CANVAS_BG, AMBER, 0.5), mm_rect, width=1)
        inset = int(OUT_EDGE * MM)
        act = mm_rect.inflate(-2 * inset, -2 * inset)
        pygame.draw.rect(screen, lerp_color(CANVAS_BG, GREEN, 0.45), act, width=1)
        if hand:
            dx = mm_rect.left + int(cur_x * MM)
            dy = mm_rect.top + int(cur_y * MM)
            if pen_now:
                pygame.draw.circle(screen, GREEN, (dx, dy), 5)
            else:
                pygame.draw.circle(screen, AMBER, (dx, dy), 5, width=2)

        # Z 高度计：条越高=手越高，越过刻度线=抬笔
        pygame.draw.rect(screen, lerp_color(CANVAS_BG, TXT_DIM, 0.35), gauge, width=1)
        if hand:
            fill_h = int(min(1.0, max(0.0, cur_z)) * MM)
            bar = pygame.Rect(gauge.left + 2, gauge.bottom - fill_h,
                              GW - 4, max(0, fill_h - 2))
            pygame.draw.rect(screen, GREEN if pen_now else AMBER, bar)
        ty = gauge.bottom - int(Z_HIGH_UP * MM)
        pygame.draw.line(screen, TXT, (gauge.left - 3, ty), (gauge.right + 3, ty))
        zlab = f_sml.render("Z", True, TXT_DIM)
        screen.blit(zlab, zlab.get_rect(midtop=(gauge.centerx, gauge.bottom + 3)))

        # ---------- 调试(按 D) ----------
        if show_debug:
            dbg = "raw x[%.2f~%.2f] y[%.2f~%.2f]  z=%.2f  fps=%d  ai=%s  snd=%s  flipX=%d flipY=%d swap=%d  g=%d aw=%d cnt=%d" % (
                raw_range[0], raw_range[1], raw_range[2], raw_range[3],
                cur_z, clock.get_fps(), "on" if AI_ENABLED else "off",
                "on" if SND else "off", FLIP_X, FLIP_Y, SWAP_XY,
                gest_dbg[0], gest_dbg[1], gest_dbg[2])
            screen.blit(f_sml.render(dbg, True, (0, 170, 0)), (10, H - 24))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
