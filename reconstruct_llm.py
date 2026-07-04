"""End-to-end reconstruction demo: rough trajectory -> vision LLM recognition + prompt -> text-to-image -> side-by-side comparison.

Pipeline
  1) Read stroke-3 (.npy), render it into a clean sketch PNG (left: original trajectory)
  2) Vision model looks at the image: identifies what was drawn and writes a text-to-image prompt
  3) Text-to-image model generates a refined image from the prompt (right: reconstruction)
  4) Combine into a side-by-side comparison saved to out/<name>_compare.png

Uses Google Gemini by default (free tier available). PROVIDER at the top can switch to OpenAI.

Usage (Gemini, free)
  Get an API key at https://aistudio.google.com (free)
  PowerShell: $env:GEMINI_API_KEY="your key"
  python reconstruct_llm.py out/circle_stroke3.npy
  python reconstruct_llm.py out/circle_stroke3.npy --label hedgehog   # skip recognition, give the label directly

Dependencies: google-genai, openai, pillow, numpy, matplotlib
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys

import numpy as np

# Auto-load keys from key_local.py into environment variables (file is excluded by .gitignore)
try:
    import key_local as _k
    for _name in ("GEMINI_API_KEY", "OPENAI_API_KEY"):
        _v = getattr(_k, _name, "")
        if _v and not os.getenv(_name):
            os.environ[_name] = _v
except ImportError:
    pass

# ---------------- Tunable: provider and model ----------------
PROVIDER = "gemini"              # recognition: 'gemini' (free) or 'openai'
IMAGE_PROVIDER = "pollinations"  # text-to-image: 'pollinations' (free, no key) / 'gemini' (paid) / 'openai' (paid)

# Gemini (recognition is free; image generation needs paid tier)
G_VISION_MODEL = "gemini-2.5-flash"        # image recognition + prompt writing
G_IMAGE_MODEL = "gemini-2.5-flash-image"   # text-to-image (paid); can switch to gemini-3.1-flash-image

# OpenAI (paid)
O_VISION_MODEL = "gpt-4o-mini"
O_IMAGE_MODEL = "gpt-image-1"
O_IMAGE_QUALITY = "low"
O_IMAGE_SIZE = "1024x1024"

SYS_PROMPT = (
    "You are a hand-drawn sketch recognition assistant. The user gives you a very rough monochrome sketch trajectory; "
    "decide what object it most likely depicts, then write an English prompt for a text-to-image model, "
    "aiming to generate a clean, recognizable, consistently styled illustration of that object. "
    'Output JSON only: {"label": "english object name", "prompt": "english text-to-image prompt"}. '
    "Both label and prompt must be in English. Add "
    "'simple, clean line illustration, white background, centered' to the prompt."
)


# ---------------- stroke-3 -> image ----------------
def stroke3_to_strokes(s3: np.ndarray):
    """Restore into multiple strokes in absolute coordinates [(N,2), ...].

    The displacement between a pen lift (lift=1) and the next stroke's start is treated as a "pen-up reposition"; no connecting line is drawn.
    """
    strokes, cur, new_stroke = [], None, True
    x = y = 0.0
    for dx, dy, lift in s3:
        x += float(dx); y += float(dy)
        if new_stroke:
            cur = [[x, y]]; new_stroke = False     # start of a new stroke (not connected to the previous one)
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
    """Render stroke-3 into a clean white-background black-line sketch PNG (bytes)."""
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
        sys.exit('GEMINI_API_KEY missing. PowerShell: $env:GEMINI_API_KEY="your key"  '
                 "(get one for free at https://aistudio.google.com)")
    return genai.Client(api_key=key)


def _gemini_recognize(img_png: bytes) -> dict:
    from google.genai import types
    client = _gemini_client()
    resp = client.models.generate_content(
        model=G_VISION_MODEL,
        contents=[types.Part.from_bytes(data=img_png, mime_type="image/png"),
                  SYS_PROMPT + "\nThis is a rough trajectory; identify it and give a prompt."],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return json.loads(resp.text)


def _gemini_image(prompt: str) -> bytes:
    client = _gemini_client()
    resp = client.models.generate_content(model=G_IMAGE_MODEL, contents=prompt)
    for part in resp.parts:
        if getattr(part, "inline_data", None) is not None:
            return part.inline_data.data
    sys.exit("Gemini returned no image; the key may lack image-generation quota. Change PROVIDER or the model.")


# ---------------- OpenAI ----------------
def _openai_client():
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit('OPENAI_API_KEY missing. PowerShell: $env:OPENAI_API_KEY="sk-..."')
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
                {"type": "text", "text": "This is a rough trajectory; identify it and give a prompt."},
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


# ---------------- Pollinations (free, no key required) ----------------
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


# ---------------- Comparison image ----------------
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
    ap.add_argument("stroke3", help="stroke-3 .npy path")
    ap.add_argument("--label", default=None, help="skip recognition, specify the label directly")
    args = ap.parse_args()

    s3 = np.load(args.stroke3)
    if s3.ndim != 2 or s3.shape[1] != 3 or len(s3) < 2:
        sys.exit("Bad stroke-3 format, expected (N,3).")

    print("[1/4] Rendering original sketch ...")
    left = render_sketch(s3)

    if args.label:
        label = args.label
        prompt = ("simple, clean line illustration of a %s, white background, centered"
                  % args.label)
    else:
        print("[2/4] Recognizing (%s / %s) ..."
              % (PROVIDER, G_VISION_MODEL if PROVIDER == "gemini" else O_VISION_MODEL))
        info = recognize(left)
        label, prompt = info.get("label", "?"), info["prompt"]
        print("      Label:", label)
        print("      Prompt:", prompt)

    print("[3/4] Generating refined image ...")
    right = generate_image(prompt)

    print("[4/4] Composing comparison image ...")
    os.makedirs("out", exist_ok=True)
    base = os.path.splitext(os.path.basename(args.stroke3))[0]
    out_path = os.path.join("out", base + "_compare.png")
    side_by_side(left, right, label, out_path)
    print("Done ->", out_path)


if __name__ == "__main__":
    main()
