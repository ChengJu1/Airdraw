"""端到端重建 Demo：粗糙轨迹 -> 视觉大模型识别+提示词 -> 文生图 -> 左右对比。

流程
  1) 读 stroke-3 (.npy)，还原成干净简笔画 PNG（左：原始轨迹）
  2) 视觉模型看图：识别画的是什么，并写一段文生图提示词
  3) 文生图模型按提示词生成精修图（右：重建结果）
  4) 拼成左右对比图保存到 out/<name>_compare.png

默认用 Google Gemini（有免费额度）。顶部 PROVIDER 可切到 OpenAI。

用法（Gemini，免费）
  在 https://aistudio.google.com 拿 API key（免费）
  PowerShell: $env:GEMINI_API_KEY="你的key"
  python reconstruct_llm.py out/circle_stroke3.npy
  python reconstruct_llm.py out/circle_stroke3.npy --label 刺猬   # 跳过识别，直接给标签

依赖：google-genai, openai, pillow, numpy, matplotlib
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys

import numpy as np

# 自动从 key_local.py 读密钥写入环境变量（该文件已被 .gitignore 排除）
try:
    import key_local as _k
    for _name in ("GEMINI_API_KEY", "OPENAI_API_KEY"):
        _v = getattr(_k, _name, "")
        if _v and not os.getenv(_name):
            os.environ[_name] = _v
except ImportError:
    pass

# ---------------- 可调：服务商与模型 ----------------
PROVIDER = "gemini"              # 识别用：'gemini'(免费) 或 'openai'
IMAGE_PROVIDER = "pollinations"  # 文生图用：'pollinations'(免费无key) / 'gemini'(付费) / 'openai'(付费)

# Gemini（识别免费；图像生成需付费层）
G_VISION_MODEL = "gemini-2.5-flash"        # 看图识别 + 写提示词
G_IMAGE_MODEL = "gemini-2.5-flash-image"   # 文生图(付费)；可换 gemini-3.1-flash-image

# OpenAI（付费）
O_VISION_MODEL = "gpt-4o-mini"
O_IMAGE_MODEL = "gpt-image-1"
O_IMAGE_QUALITY = "low"
O_IMAGE_SIZE = "1024x1024"

SYS_PROMPT = (
    "你是手绘草图识别助手。用户给你一张非常粗糙的单色简笔画轨迹，"
    "请判断它最可能想画的是什么物体，然后写一段用于文生图模型的英文提示词，"
    "目标是生成一张干净、可识别、风格统一的该物体插画。"
    '只输出 JSON：{"label": "english object name", "prompt": "english text-to-image prompt"}。'
    "label 和 prompt 都用英文。提示词里加上 "
    "'simple, clean line illustration, white background, centered'。"
)


# ---------------- stroke-3 -> 图片 ----------------
def stroke3_to_strokes(s3: np.ndarray):
    """还原成绝对坐标的多笔 [(N,2), ...]。

    抬笔(lift=1)后到下一笔起点的位移视为“抬笔重定位”，不画连线。
    """
    strokes, cur, new_stroke = [], None, True
    x = y = 0.0
    for dx, dy, lift in s3:
        x += float(dx); y += float(dy)
        if new_stroke:
            cur = [[x, y]]; new_stroke = False     # 新一笔的起点（不与上一笔相连）
        else:
            cur.append([x, y])
        if lift >= 0.5:
            if len(cur) >= 2:
                strokes.append(np.asarray(cur))
            new_stroke = True
    if cur is not None and not new_stroke and len(cur) >= 2:
        strokes.append(np.asarray(cur))
    return strokes


def render_sketch(s3: np.ndarray, px: int = 512, lw: float = 6.0) -> bytes:
    """把 stroke-3 渲染成白底黑线的干净简笔画 PNG（bytes）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strokes = stroke3_to_strokes(s3)
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_facecolor("white")
    for s in strokes:
        if len(s) >= 2:
            ax.plot(s[:, 0], -s[:, 1], "-", color="black",
                    lw=lw, solid_capstyle="round", solid_joinstyle="round")
    ax.set_aspect("equal")
    allp = np.concatenate(strokes, axis=0)
    cx = (allp[:, 0].min() + allp[:, 0].max()) / 2
    yy = -allp[:, 1]
    cy = (yy.min() + yy.max()) / 2
    r = max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) / 2 * 1.15 + 1
    ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    return buf.getvalue()


# ---------------- Gemini ----------------
def _gemini_client():
    from google import genai
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        sys.exit('缺少 GEMINI_API_KEY。PowerShell: $env:GEMINI_API_KEY="你的key"  '
                 "（在 https://aistudio.google.com 免费获取）")
    return genai.Client(api_key=key)


def _gemini_recognize(img_png: bytes) -> dict:
    from google.genai import types
    client = _gemini_client()
    resp = client.models.generate_content(
        model=G_VISION_MODEL,
        contents=[types.Part.from_bytes(data=img_png, mime_type="image/png"),
                  SYS_PROMPT + "\n这是粗糙轨迹，请识别并给出提示词。"],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return json.loads(resp.text)


def _gemini_image(prompt: str) -> bytes:
    client = _gemini_client()
    resp = client.models.generate_content(model=G_IMAGE_MODEL, contents=prompt)
    for part in resp.parts:
        if getattr(part, "inline_data", None) is not None:
            return part.inline_data.data
    sys.exit("Gemini 没返回图片，可能该 key 无图像生成额度。可改 PROVIDER 或换模型。")


# ---------------- OpenAI ----------------
def _openai_client():
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit('缺少 OPENAI_API_KEY。PowerShell: $env:OPENAI_API_KEY="sk-..."')
    from openai import OpenAI
    return OpenAI()


def _openai_recognize(img_png: bytes) -> dict:
    client = _openai_client()
    b64 = base64.b64encode(img_png).decode()
    resp = client.chat.completions.create(
        model=O_VISION_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "这是粗糙轨迹，请识别并给出提示词。"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def _openai_image(prompt: str) -> bytes:
    client = _openai_client()
    kw = dict(model=O_IMAGE_MODEL, prompt=prompt, size=O_IMAGE_SIZE, n=1)
    if O_IMAGE_MODEL.startswith("gpt-image"):
        kw["quality"] = O_IMAGE_QUALITY
    else:
        kw["response_format"] = "b64_json"
    res = client.images.generate(**kw)
    return base64.b64decode(res.data[0].b64_json)


# ---------------- Pollinations（免费、无需 key）----------------
def _pollinations_image(prompt: str) -> bytes:
    import urllib.parse
    import urllib.request
    url = ("https://image.pollinations.ai/prompt/"
           + urllib.parse.quote(prompt)
           + "?width=768&height=768&nologo=true&model=flux")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def recognize(img_png: bytes) -> dict:
    return _gemini_recognize(img_png) if PROVIDER == "gemini" else _openai_recognize(img_png)


def generate_image(prompt: str) -> bytes:
    if IMAGE_PROVIDER == "pollinations":
        return _pollinations_image(prompt)
    if IMAGE_PROVIDER == "gemini":
        return _gemini_image(prompt)
    return _openai_image(prompt)


# ---------------- 拼对比图 ----------------
def side_by_side(left_png: bytes, right_png: bytes, label: str, out_path: str):
    from PIL import Image, ImageDraw
    L = Image.open(io.BytesIO(left_png)).convert("RGB")
    R = Image.open(io.BytesIO(right_png)).convert("RGB")
    h = max(L.height, R.height)
    L = L.resize((int(L.width * h / L.height), h))
    R = R.resize((int(R.width * h / R.height), h))
    pad, top = 20, 50
    W = L.width + R.width + pad * 3
    canvas = Image.new("RGB", (W, h + top + pad), "white")
    canvas.paste(L, (pad, top)); canvas.paste(R, (pad * 2 + L.width, top))
    d = ImageDraw.Draw(canvas)
    font = None
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 22)
    except OSError:
        pass
    d.text((pad, 14), "Original (Skywriter)", fill="black", font=font)
    d.text((pad * 2 + L.width, 14), "Reconstructed: %s" % label, fill="black", font=font)
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stroke3", help="stroke-3 .npy 路径")
    ap.add_argument("--label", default=None, help="跳过识别，直接指定标签")
    args = ap.parse_args()

    s3 = np.load(args.stroke3)
    if s3.ndim != 2 or s3.shape[1] != 3 or len(s3) < 2:
        sys.exit("stroke-3 格式不对，应为 (N,3)。")

    print("[1/4] 渲染原始简笔画 ...")
    left = render_sketch(s3)

    if args.label:
        label = args.label
        prompt = ("simple, clean line illustration of a %s, white background, centered"
                  % args.label)
    else:
        print("[2/4] 识别中（%s / %s）..."
              % (PROVIDER, G_VISION_MODEL if PROVIDER == "gemini" else O_VISION_MODEL))
        info = recognize(left)
        label, prompt = info.get("label", "?"), info["prompt"]
        print("      识别结果:", label)
        print("      提示词:", prompt)

    print("[3/4] 生成精修图 ...")
    right = generate_image(prompt)

    print("[4/4] 拼接对比图 ...")
    os.makedirs("out", exist_ok=True)
    base = os.path.splitext(os.path.basename(args.stroke3))[0]
    out_path = os.path.join("out", base + "_compare.png")
    side_by_side(left, right, label, out_path)
    print("完成 ->", out_path)


if __name__ == "__main__":
    main()
