#!/usr/bin/env python3
"""Standalone fullscreen drawing app for Raspberry Pi (main exhibition program, no PC/browser/keyboard needed).

Drives HDMI fullscreen directly (Pygame), reads the Skywriter, does real-time processing (range calibration /
median + spike gate + 1€ smoothing / out-of-range + jump stroke breaks), and records CSV. Filter chain shared with web_capture.py via rt_filters.py.

Exhibition loop:
  draw -> pen idle and hand away for AI_WAIT_SEC s -> auto-send to AI recognition (Gemini vision) ->
  text-to-image reconstruction -> fullscreen side-by-side (your doodle vs the AI remake) ->
  clear after RESULT_SEC s and wait for the next visitor. No network / no key: degrades to plain drawing + auto clear.

Exhibition UI / interaction:
  - Dark background, sensor range mapped to a fixed canvas at screen center (cursor goes where the hand goes)
  - Live cursor: pen down = solid bright dot, pen up = hollow ring
  - Bottom-right minimap: sensor XY range + out-of-range pen-up zone + current position dot + Z height gauge
  - Gestures: fast flick = clear (armed after a short pen pause); color auto-rotates on every stroke
  - Sound feedback: pen down/up/clear/recognition done; auto-mutes without a sound card
  - Sensor I2C self-healing: startup failure / mid-run dropout never exits; hardware-reset reconnect loop with an on-screen notice
  - Keyboard (debug): ESC/Q quit  Space/C clear  Enter send to AI now  D debug
    orientation calibration (applies live): X mirror left-right  Y mirror up-down  S swap XY axes

Dependencies (same venv that already has lgpio/smbus2): pip install pygame
AI part uses plain REST (urllib), no google-genai/PIL needed. Keys come from key_local.py or the env var
GEMINI_API_KEY -- copy mgc3130.py, rt_filters.py and key_local.py into the same directory.
Auto-start on boot + restart on crash: see skywriter-draw.service (install notes in its header comment).
Manual run: ~/start_draw.sh
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

# ============== Processing params (filter chain params live in rt_filters.py, shared with web_capture.py) ==============
FLIP_X = False
FLIP_Y = True
SWAP_XY = False
X_LO, X_HI = 0.08, 0.92         # measured (cal_20260705): ~8% saturation hugging each end, use only the linear middle
Y_LO, Y_HI = 0.08, 0.92
OUT_EDGE = 0.08                 # near the edge = pen up; aligned with LO/HI, saturated zone (clamped 0/1) not drawn
# z hysteresis for pen up/down: z > Z_HIGH_UP lifts the pen, falling back < Z_HIGH_DOWN resumes
# measured natural drawing height z≈0.2~0.5: pen-down threshold must not go below 0.5, otherwise the
# upper half of the drawing range hugs the threshold and shatters into breaks (tried 0.45, broke badly)
Z_HIGH_UP = 0.60
Z_HIGH_DOWN = 0.50
MAX_JUMP = 0.07                 # fast-move lead-in lines count as pen up, not drawn (loosened to reduce false breaks)
AUTO_CLEAR_SEC = 8.0            # (fallback when AI is unavailable) auto clear after hand away this long

# ============== Gestures & sound ==============
FLICK_CLEARS = True             # fast flick = clear canvas
FLICK_DEBOUNCE = 1.2            # min interval between two flicks (s)
GESTURE_IDLE = 0.5              # gestures unlock after the pen idles this long, no misfires while drawing
SOUND = True                    # master switch for sound feedback

# ============== AI recognition & reconstruction (same flow as reconstruct_llm.py, pure urllib) ==============
AI_WAIT_SEC = 3.0               # pen idle and hand away this long => auto-send to AI
RESULT_SEC = 15.0               # result display duration, then clear for the next visitor
MIN_PTS_FOR_AI = 12             # too few total points (accidental touch) => don't send to AI
G_VISION_MODEL = "gemini-2.5-flash"
SYS_PROMPT = (
    "You are a hand-drawn sketch recognition assistant. The user gives you a very rough monochrome line-drawing trace; "
    "it may be a single object, or a few things combined into one scene. "
    "Decide what was drawn, then write an English prompt for a text-to-image model "
    "aiming for a clean, recognizable, consistently styled illustration; "
    "if there are multiple elements, blend them naturally into one scene, keeping the original relative layout. "
    'Output JSON only: {"label": "short english name", "prompt": "english text-to-image prompt"}. '
    "Both label and prompt must be in English; label at most 4 words. Add "
    "'simple, clean line illustration, white background, centered' to the prompt."
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
# Main plan: one paid-tier Gemini key covers everything (recognition gemini-2.5-flash + images nano banana).
# Free tier is not enough: only 20 recognitions/day and zero image quota, make sure billing is on.
# Recognition order: SiliconFlow (if key present) -> Gemini
# Text-to-image order: Gemini -> Replicate -> SiliconFlow -> Pollinations (free fallback)
G_IMAGE_MODEL = "gemini-2.5-flash-image"
SF_VISION_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
SF_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

# ============== Display appearance ==============
FILL = 0.80                     # canvas size as a fraction of the shorter screen side
BG = (10, 12, 18)               # fullscreen background
CANVAS_BG = (17, 20, 30)        # canvas background
CANVAS_BORDER = (54, 62, 88)
TXT = (225, 230, 240)
TXT_DIM = (108, 118, 142)
GREEN = (90, 220, 160)          # pen down / drawable state
AMBER = (255, 190, 90)          # pen up / boundary warning
LINE_W = 5
INK_PALETTE = [                 # color auto-rotates on every pen up
    (240, 242, 248),            # bright white
    (120, 200, 255),            # sky blue
    (140, 235, 190),            # mint
    (250, 200, 120),            # amber
    (210, 160, 255),            # lavender
    (255, 150, 165),            # coral
]
# ===================================================================

# ---- Filter chain (implementation and params in rt_filters.py) ----
med_f = MedianWin()
spike = SpikeGate()
fx = OneEuro()
fy = OneEuro()
zpen = ZPenHysteresis(down=Z_HIGH_DOWN, up=Z_HIGH_UP)
# hover cursor smoothing (independent of the stroke filters, keeps running while pen is up)
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


# ---- Sound (synthesized in code, no audio files; silent without a sound card) ----
SND = {}


def snd(name):
    if SOUND and name in SND:
        try:
            SND[name].play()
        except Exception:
            pass


def build_sounds():
    """Synthesize a few short sound effects at startup. Returns empty if the mixer is down (no sound card)."""
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
        "down": tone(880, 45, 0.5),            # pen-down "tap"
        "up": tone(520, 35, 0.22),             # soft pen-up blip
        "clear": tone(700, 230, 0.4, freq2=170),   # clear-canvas downward sweep
        "tick": tone(1320, 25, 0.3),           # color-change click
        "success": arp([523, 659, 784, 1046]),  # little arpeggio on recognition done
        "fail": tone(170, 260, 0.35),          # low buzz on AI failure
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


# ---- Shared drawing / cursor / sensor state ----
lock = threading.Lock()
strokes = [[0, []]]            # each element: [palette index, list of points (0..1)]
color_idx = 0                  # current ink color (auto-rotates on every pen up)
pen_now = 0
last_activity = time.time()    # last time a hand was detected
draw_last = 0.0                # last time a point was actually drawn pen-down (drives the AI trigger)
raw_range = [1.0, 0.0, 1.0, 0.0]   # mnx,mxx,mny,mxy (debug)
t0 = time.time()

cur_x, cur_y = 0.5, 0.5        # cursor position (0..1, canvas coords)
cur_z = 1.0                    # current hand height (0..1)
cur_seen = 0.0                 # last valid frame time
prev_pen = 0
gest_dbg = [0, 0, 0]           # last gesture code / airwheel active / count (debug bar)

sensor = None                  # created/rebuilt by the reader thread (I2C self-healing)
sensor_ok = False

# ---- AI state (shared between main loop and worker thread) ----
ai_lock = threading.Lock()
ai = {
    "mode": "draw",            # draw / thinking / result
    "token": 0,                # +1 on every trigger/cancel; stale worker results are dropped
    "label": "",
    "img": None,               # generated image Surface
    "sketch": None,            # sketch Surface sent for recognition (black lines on white)
    "t0": 0.0,                 # time result mode was entered
    "err": "",
    "err_t": 0.0,
    "_cache": None,            # scaled result cache
}


def clear_canvas():
    global strokes, prev_pen, draw_last
    with lock:
        strokes = [[color_idx, []]]
    prev_pen = 0
    draw_last = 0.0


def reader():
    """Sensor reader thread: I2C self-healing + gestures + filtering into strokes."""
    global sensor, sensor_ok
    global pen_now, last_activity, draw_last, prev_pen, color_idx
    global cur_x, cur_y, cur_z, cur_seen
    last_ts = None
    last_draw = None
    last_flick = 0.0
    z_hist = []                # 3-point median window on z, kills false breaks from single-frame spikes
    err_streak = 0
    last_frame_t = time.time()
    cnt = 0

    def reconnect():
        """Close the old handle, hardware-reset and reconnect. Slow retry on failure, never exits."""
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
            print("sensor connected")
        except Exception as exc:  # noqa: BLE001
            print("sensor reconnect failed (retry in 2 s): %r" % (exc,))
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

                    # gestures: unlock GESTURE_IDLE s after the pen idles, no misfires while drawing
                    now = time.time()
                    gest_dbg[0], gest_dbg[1], gest_dbg[2] = gest, int(aw_active), aw_count
                    gesture_ok = (draw_last == 0 or now - draw_last > GESTURE_IDLE)

                    # fast flick = clear canvas (also skips the result page)
                    if FLICK_CLEARS and gest in FLICKS and gesture_ok and \
                            now - last_flick > FLICK_DEBOUNCE:
                        last_flick = now
                        ai_cancel()
                        clear_canvas()
                        snd("clear")
                        print("gesture: flick clear (code=%d)" % gest)

                    if valid:
                        raw_range[0] = min(raw_range[0], rx); raw_range[1] = max(raw_range[1], rx)
                        raw_range[2] = min(raw_range[2], ry); raw_range[3] = max(raw_range[3], ry)

                        # z goes through the 3-point median before hysteresis, single-frame spikes no longer break strokes
                        z_hist.append(rz)
                        if len(z_hist) > 3:
                            z_hist.pop(0)
                        zf = sorted(z_hist)[len(z_hist) // 2]

                        # z hysteresis decides pen up/down; edge saturation => force pen up
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

                        # hover cursor: smoothed regardless of pen state, visitors always see "where am I"
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
                            # jump guard: fast repositioning doesn't connect a long line
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
                                if not strokes[-1][1]:      # first point of the stroke fixes its color
                                    strokes[-1][0] = color_idx
                                strokes[-1][1].append((sx, sy))
                            cur_x, cur_y = sx, sy
                            draw_last = time.time()
                        else:
                            with lock:
                                if strokes and strokes[-1][1]:
                                    # pen up = start a new stroke, color steps one
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
            # a long gap without frames isn't necessarily a hang: the chip may go quiet when idle (nobody there).
            # probe the bus first; if the chip still answers it's just idle, only a real dropout triggers reset+reconnect.
            if time.time() - last_frame_t > 3.0:
                if probe_i2c():
                    last_frame_t = time.time()
                    sensor_ok = True
                else:
                    print("sensor silent and bus not answering, reset and reconnect")
                    reconnect()
        time.sleep(0.001)
      except OSError:
        # occasional I2C / GPIO error: skip; reconnect after too many in a row
        err_streak += 1
        if err_streak > 100:
            reconnect()
        time.sleep(0.002)
      except Exception as exc:  # noqa: BLE001
        print("reader exception (skipped): %r" % (exc,))
        err_streak += 1
        if err_streak > 100:
            reconnect()
        time.sleep(0.002)


# ================== AI: sketch rendering / recognition / text-to-image ==================
def sketch_png(snap, px=512):
    """Render a stroke snapshot as black-on-white PNG (bytes) + Surface, centered and auto-scaled."""
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
    """Dig the JSON out of a model reply (tolerates ```json fences and surrounding chatter)."""
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON in reply: %r" % txt[:120])
    return json.loads(txt[i:j + 1])


def sf_recognize(png_bytes):
    """SiliconFlow Qwen-VL vision recognition, returns {"label":..., "prompt":...}."""
    b64 = base64.b64encode(png_bytes).decode()
    body = {
        "model": SF_VISION_MODEL,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + b64}},
                {"type": "text", "text": "This is a rough trace. Identify it and give the prompt."},
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


def gemini_recognize(png_bytes):
    """Gemini REST vision recognition, returns {"label":..., "prompt":...}."""
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "%s:generateContent?key=%s" % (G_VISION_MODEL, GEMINI_KEY))
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/png",
                             "data": base64.b64encode(png_bytes).decode()}},
            {"text": SYS_PROMPT + "\nThis is a rough trace. Identify it and give the prompt."},
        ]}],
        "generationConfig": {"response_mime_type": "application/json"},
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.load(r)
    return json.loads(resp["candidates"][0]["content"]["parts"][0]["text"])


def recognize(png_bytes):
    """Recognition: SiliconFlow first, fall back to Gemini (free tier: only 20/day)."""
    if SF_KEY:
        try:
            return sf_recognize(png_bytes)
        except Exception as exc:  # noqa: BLE001
            print("SiliconFlow recognition failed, trying Gemini: %r" % (exc,))
    return gemini_recognize(png_bytes)


def gemini_image(prompt):
    """Gemini text-to-image (gemini-2.5-flash-image, paid tier required), returns image bytes."""
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
    raise ValueError("Gemini returned no image part")


def replicate_image(prompt):
    """Replicate FLUX.1 schnell text-to-image (Prefer:wait, synchronous), returns image bytes."""
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
    """SiliconFlow FLUX.1 schnell text-to-image, 1-3 s per image, returns image bytes.

    The API returns an image URL first (valid for 1 hour), then it's downloaded as bytes.
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
    """Free text-to-image (no key needed), slow but a safety net, returns image bytes."""
    url = ("https://image.pollinations.ai/prompt/"
           + urllib.parse.quote(prompt)
           + "?width=768&height=768&nologo=true&model=flux")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def generate_image(prompt):
    """Try Gemini -> Replicate -> SiliconFlow -> Pollinations in order, whichever has a key."""
    for name, key, fn in (("Gemini", GEMINI_KEY, gemini_image),
                          ("Replicate", REPLICATE_KEY, replicate_image),
                          ("SiliconFlow", SF_KEY, sf_image)):
        if key:
            try:
                return fn(prompt)
            except Exception as exc:  # noqa: BLE001
                print("%s generation failed, trying next: %r" % (name, exc))
    return pollinations_image(prompt)


def ai_worker(snap, token):
    """Background thread: render -> recognize -> generate. Stale token (visitor drew again / cleared) drops the result."""
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
        print("AI failed (degrading, drawing continues): %r" % (exc,))
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


# ============================== Main program ==============================
def main():
    global FLIP_X, FLIP_Y, SWAP_XY
    import pygame
    pygame.mixer.pre_init(22050, -16, 1, 512)
    pygame.init()
    try:
        SND.update(build_sounds())
    except Exception as exc:  # noqa: BLE001
        print("sound init failed (continuing muted): %r" % (exc,))
    pygame.mouse.set_visible(False)
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    W, H = screen.get_size()
    clock = pygame.time.Clock()
    f_big = pygame.font.SysFont(None, 64)
    f_mid = pygame.font.SysFont(None, 30)
    f_sml = pygame.font.SysFont(None, 21)

    # canvas: sensor range mapped to a fixed square in the screen center
    side = int(min(W, H) * FILL)
    canvas = pygame.Rect((W - side) // 2, (H - side) // 2, side, side)

    def to_px(p):
        return (canvas.left + p[0] * side, canvas.top + p[1] * side)

    # minimap layout (bottom right)
    MM = max(120, int(min(W, H) * 0.16))       # minimap side length
    GW = 16                                    # height gauge width
    PAD = 14
    panel = pygame.Rect(0, 0, MM + GW + PAD * 3, MM + PAD * 2 + 26)
    panel.bottomright = (W - 28, H - 28)
    mm_rect = pygame.Rect(panel.left + PAD, panel.top + PAD + 22, MM, MM)
    gauge = pygame.Rect(mm_rect.right + PAD, mm_rect.top, GW, MM)

    # gesture hints + palette (bottom left)
    hint_panel = pygame.Rect(0, 0, 300, panel.height)
    hint_panel.bottomleft = (28, H - 28)

    threading.Thread(target=reader, daemon=True).start()

    trigger_draw_last = -1.0    # draw_last at AI trigger time; drawing after that cancels
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
                elif ev.key == pygame.K_x:      # orientation calibration, applies live
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

        hand = (now - cur_seen) < 0.35          # valid frame recently => hand in range
        idle = now - last_activity

        with lock:
            snap = [(ci, list(s)) for ci, s in strokes if len(s) >= 2]
            has_content = any(s for _, s in strokes)
        n_pts = sum(len(s) for _, s in snap)

        # ---------- State machine ----------
        if mode == "draw":
            # pen idle + hand away => auto-send to AI; fallback auto clear if AI unavailable
            if AI_ENABLED and n_pts >= MIN_PTS_FOR_AI and not hand \
                    and draw_last > 0 and now - draw_last > AI_WAIT_SEC:
                trigger_draw_last = draw_last
                ai_trigger(snap)
                mode = "thinking"
            elif has_content and idle > AUTO_CLEAR_SEC and \
                    (not AI_ENABLED or n_pts < MIN_PTS_FOR_AI):
                clear_canvas()
        elif mode == "thinking":
            if draw_last != trigger_draw_last:   # visitor came back and drew => cancel this run
                ai_cancel()
                mode = "draw"
        elif mode == "result":
            with ai_lock:
                shown = now - ai["t0"]
            if shown > RESULT_SEC or pen_now:    # time's up / someone starts drawing => clear and restart
                ai_cancel()
                clear_canvas()
                mode = "draw"

        # ---------- Result page ----------
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

        # ---------- Background & canvas ----------
        screen.fill(BG)
        pygame.draw.rect(screen, CANVAS_BG, canvas, border_radius=14)
        pygame.draw.rect(screen, CANVAS_BORDER, canvas, width=2, border_radius=14)

        # canvas center cross: gives visitors an "origin" reference
        ccx, ccy = canvas.center
        cross = lerp_color(CANVAS_BG, TXT_DIM, 0.55)
        pygame.draw.line(screen, cross, (ccx - 12, ccy), (ccx + 12, ccy))
        pygame.draw.line(screen, cross, (ccx, ccy - 12), (ccx, ccy + 12))

        # ---------- Strokes (two-pass glow: wide dim halo + bright thin core) ----------
        screen.set_clip(canvas)
        for ci, s in snap:
            color = INK_PALETTE[ci % len(INK_PALETTE)]
            proj = [to_px(p) for p in s]
            halo = lerp_color(CANVAS_BG, color, 0.25)
            pygame.draw.lines(screen, halo, False, proj, LINE_W + 8)
            pygame.draw.lines(screen, color, False, proj, LINE_W)
            r = LINE_W // 2
            for px, py in proj:                 # round joints, no gaps at corners
                pygame.draw.circle(screen, color, (int(px), int(py)), r)

        # ---------- Live cursor ----------
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

        # ---------- Sensor offline notice ----------
        if not sensor_ok:
            pulse = 0.5 + 0.5 * math.sin(now * 3.0)
            t1 = f_big.render("SENSOR RECONNECTING", True,
                              lerp_color(TXT_DIM, AMBER, pulse))
            t2 = f_mid.render("please wait / check the Skywriter board", True, TXT_DIM)
            screen.blit(t1, t1.get_rect(center=(ccx, ccy - 24)))
            screen.blit(t2, t2.get_rect(center=(ccx, ccy + 28)))
        # ---------- Idle attract screen (nobody around and canvas empty) ----------
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

        # ---------- AI recognizing notice ----------
        if mode == "thinking":
            dots = "." * (1 + int(now * 2) % 3)
            t1 = f_big.render("Recognizing%s" % dots, True, TXT)
            t2 = f_mid.render("the AI is looking at your sketch", True, TXT_DIM)
            screen.blit(t1, t1.get_rect(center=(ccx, canvas.top + 56)))
            screen.blit(t2, t2.get_rect(center=(ccx, canvas.top + 100)))

        # ---------- Top title & status ----------
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

        # AI trigger countdown / failure notice
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

        # ---------- Bottom left: gesture hints + palette ----------
        pygame.draw.rect(screen, CANVAS_BG, hint_panel, border_radius=10)
        pygame.draw.rect(screen, CANVAS_BORDER, hint_panel, width=1, border_radius=10)
        lab = f_sml.render("GESTURES", True, TXT_DIM)
        screen.blit(lab, (hint_panel.left + PAD, hint_panel.top + 7))
        h1 = f_sml.render("fast swipe (hand raised)  =  clear", True, TXT)
        h2 = f_sml.render("every new stroke  =  new colour", True, TXT)
        screen.blit(h1, (hint_panel.left + PAD, hint_panel.top + 32))
        screen.blit(h2, (hint_panel.left + PAD, hint_panel.top + 56))
        # palette: current color enlarged with a ring
        sw_y = hint_panel.bottom - 28
        for i, c in enumerate(INK_PALETTE):
            sw_x = hint_panel.left + PAD + 10 + i * 34
            if i == color_idx % len(INK_PALETTE):
                pygame.draw.circle(screen, c, (sw_x, sw_y), 11)
                pygame.draw.circle(screen, TXT, (sw_x, sw_y), 14, width=2)
            else:
                pygame.draw.circle(screen, lerp_color(CANVAS_BG, c, 0.6),
                                   (sw_x, sw_y), 8)

        # ---------- Bottom right: sensor range minimap + height gauge ----------
        pygame.draw.rect(screen, CANVAS_BG, panel, border_radius=10)
        pygame.draw.rect(screen, CANVAS_BORDER, panel, width=1, border_radius=10)
        lab = f_sml.render("SENSOR RANGE", True, TXT_DIM)
        screen.blit(lab, (panel.left + PAD, panel.top + 7))

        # outer box = full sensing range (edges are the pen-up zone), inner box = drawable area
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

        # Z height gauge: taller bar = higher hand, above the tick line = pen up
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

        # ---------- Debug (press D) ----------
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
