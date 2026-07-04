#!/usr/bin/env python3
"""Skywriter air drawing: real-time hand tracking display + CSV recording (algorithm-only tuning, no buttons).

Algorithm fixes for "laggy tracking / messy shapes / sluggish pen up/down":
  1. Push only the "latest point" to the browser (no queueing) -- removes latency backlog, tracks the hand
  2. 1-euro filter -- smooth on slow motion, near-zero lag on fast motion
  3. Effective-range calibration + stretch to fill the canvas -- fixes "small canvas / squeezed into the middle"
  4. Out-of-range (edge saturation) + jump auto pen-up -- button-free pen up/down, removes crossing straight lines
  5. FLIP_X / FLIP_Y / SWAP_XY -- adjust orientation until tracking feels natural

Recording: each frame writes CSV(t, x, y, z, in_range, pen); x/y are raw coordinates, pen is the algorithm's pen-down decision.

Usage (Raspberry Pi, inside venv; copy mgc3130.py and rt_filters.py to the same directory):
  source ~/sky/bin/activate
  python3 ~/web_capture.py            # saves ~/captures/cap_<time>.csv
  python3 ~/web_capture.py circle     # saves ~/captures/circle.csv
Open http://Dissertation.local:5000 in a browser to watch while drawing; the top-left shows the measured raw x/y range,
used to calibrate X_LO/X_HI/Y_LO/Y_HI below. Press Ctrl+C to save when done.
Keep your hand away from the sensor at startup (so it calibrates cleanly), then start drawing.
"""
import os
import sys
import time
import csv
import math
import threading
import json
import atexit

import base64
import urllib.parse
import urllib.request

from flask import Flask, Response, request, jsonify

from mgc3130 import MGC3130, parse_frame
from rt_filters import MedianWin, SpikeGate, OneEuro, ZPenHysteresis, Z_PEN_UP

# Auto-load GEMINI_API_KEY from key_local.py (file is excluded by .gitignore)
try:
    import key_local as _k
    if getattr(_k, "GEMINI_API_KEY", "") and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = _k.GEMINI_API_KEY
except ImportError:
    pass

# Recognition + text-to-image model
VISION_MODEL = "gemini-2.5-flash"
SYS_PROMPT = (
    "You are a hand-drawn sketch recognition assistant. The user gives you a rough monochrome sketch; "
    "decide what object it most likely is, and output JSON only: "
    "{\"label\": \"english object name\"}."
)

# ============== Tunable parameters (run first, check measured range top-left, then tune) ==============
# 1) Orientation: flip y by default (screen y goes down, sensor y goes up); change these three if tracking feels wrong
FLIP_X = False
FLIP_Y = True
SWAP_XY = False

# 2) Effective range: use full scale (0~1); drawing size is handled by the frontend "auto-center + auto-fit zoom"
X_LO, X_HI = 0.0, 1.0
Y_LO, Y_HI = 0.0, 1.0
# Pen-up decision: no longer use x/y saturation for pen-up by default (avoids "stroke breaks on right-side saturation").
# Out-of-range saturation guard: when the hand leaves the sensing field rx/ry get pinned at 0 or 1, drawing a box along the edges.
# With OUT_EDGE>0, anything within OUT_EDGE of any edge counts as "pen up", so the box disappears.
# 0.04 ~ discard the outer 4% unreliable zone; lower it if drawings come out too small, raise it if the box persists.
OUT_EDGE = 0.04

# z pen-up: dual-threshold hysteresis + consecutive-frame debounce (rt_filters.ZPenHysteresis).
# The old single threshold (Z_CUT=0.28) broke strokes from z noise jitter at the critical height;
# now z < 0.26 for 2 consecutive frames => pen down, z > 0.32 for 2 frames => pen up, otherwise keep state.
# Thresholds are tuned in one place: rt_filters.Z_PEN_DOWN / Z_PEN_UP.
Z_MAX = 0.60              # position distorts when too high, force pen up

# 3) Filter chain: 3-point median -> spike gate -> 1-euro (steadier the higher z is).
#    Window / cutoff parameters live at the top of rt_filters.py; draw_app.py shares the same set.
Z_SMOOTH_T0 = 0.12        # smooth harder above this z (position gets noisier near the pen-up threshold)

# 4) Jump guard: adjacent pen-down points farther apart (mapped 0~1) than this => break stroke, no long line
#    Normal drawing moves ~0.003 per frame; "quickly moving to the start point" is much larger.
#    Smaller => fast approach/reposition moves are auto-treated as pen up, so no lead-in line is drawn
#    (the point where you slow down and actually start drawing = the start point). Raise it if tracking feels cut off.
MAX_JUMP = 0.05

# 5) Stitching: on pen down after pen up, attach the start to "where the pen last stopped".
#    With camera-follow, stitching does more harm than good (returning from out-of-range drags a long line); off by default =>
#    absolute-position drawing, out-of-range/pen-up ends the current stroke, coming back starts a fresh one, never connected.
STITCH = False
# =====================================================================

sensor = MGC3130()

# ---------------- Filter chain (implementation and parameters in rt_filters.py) ----------------
med_f = MedianWin()
spike = SpikeGate()
fx = OneEuro()
fy = OneEuro()
zpen = ZPenHysteresis()


def z_mincutoff(rz):
    """The higher the hand (z near the pen-up threshold), the noisier the position; strengthen the low-pass slightly."""
    if rz <= Z_SMOOTH_T0:
        return fx.mincutoff
    t = min(1.0, (rz - Z_SMOOTH_T0) / max(Z_PEN_UP - Z_SMOOTH_T0, 1e-6))
    return fx.mincutoff * (1.0 + 1.2 * t)


def reset_filters():
    med_f.reset()
    spike.reset()
    fx.reset()
    fy.reset()


def stretch(v, lo, hi):
    if hi - lo < 1e-6:
        return v
    return (v - lo) / (hi - lo)


# ---------------- CSV ----------------
name = sys.argv[1] if len(sys.argv) > 1 else time.strftime("cap_%Y%m%d_%H%M%S")
outdir = os.path.expanduser("~/captures"); os.makedirs(outdir, exist_ok=True)
csv_path = os.path.join(outdir, name + ".csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["t", "x", "y", "z", "in_range", "pen"])
print("Recording -> %s" % csv_path)
print("Open http://<pi>:5000 in a browser to watch while drawing; Ctrl+C to stop and save.")


@atexit.register
def _close():
    try:
        csv_file.flush(); csv_file.close()
        print("\nSaved -> %s" % csv_path)
    except Exception:
        pass


# Shared "latest point" state (keep only the newest, avoid backlog)
state = {"x": 0.5, "y": 0.5, "pen": 0, "rx": 0.0, "ry": 0.0, "rz": 0.0, "seq": 0, "live": 0}
t0 = time.time()
_count = 0

# Global state for stitching
disp_off = [0.0, 0.0]   # display offset: ignore movement while pen is up
last_disp = None        # last displayed (pen-stop) position
prev_pen = 0
reset_flag = False


def reader():
    global _count, last_disp, prev_pen, reset_flag
    last_ts = None
    last_draw = None        # most recent filtered pen-down point, for jump detection
    last_raw = None         # mapped raw point, for spike/reposition detection
    while True:
      try:
        if sensor.data_ready():
            d = sensor.read_frame()
            if d and len(d) >= 26 and d[3] == 0x91:
                ts = d[6]
                if ts != last_ts:
                    last_ts = ts
                    t = time.time() - t0
                    if reset_flag:          # double-click page -> reset stitching (works even with hand outside the field)
                        disp_off[0] = disp_off[1] = 0.0
                        last_disp = None
                        prev_pen = 0
                        reset_filters()
                        zpen.reset()
                        last_draw = None
                        last_raw = None
                        reset_flag = False
                    rx, ry, rz, valid = parse_frame(d)
                    if valid:
                        # z hysteresis decides pen up/down; edge saturation or z too high (position distorted) => force pen up
                        edge_out = (rx < OUT_EDGE or rx > 1 - OUT_EDGE or
                                    ry < OUT_EDGE or ry > 1 - OUT_EDGE)
                        pen = zpen(rz)
                        if edge_out or rz > Z_MAX:
                            pen = 0
                            zpen.reset()

                        # Orientation + range stretch
                        ax, ay = (ry, rx) if SWAP_XY else (rx, ry)
                        mx = stretch(ax, X_LO, X_HI)
                        my = stretch(ay, Y_LO, Y_HI)
                        if FLIP_X:
                            mx = 1.0 - mx
                        if FLIP_Y:
                            my = 1.0 - my
                        mx = min(1.0, max(0.0, mx))
                        my = min(1.0, max(0.0, my))

                        if not pen:
                            if prev_pen:
                                reset_filters()
                            last_draw = None
                            last_raw = None
                            sx, sy = mx, my       # still track the hand while pen is up, but unfiltered
                        else:
                            if not prev_pen:
                                reset_filters()
                            mx, my = med_f(mx, my)
                            mx, my = spike(mx, my)
                            mc = z_mincutoff(rz)
                            sx = fx(mx, t, mc)
                            sy = fy(my, t, mc)
                            # Jump guard: check both filtered and raw coordinates, so 1-euro lag can't draw a long line
                            if last_draw is not None:
                                if (math.hypot(sx - last_draw[0], sy - last_draw[1]) > MAX_JUMP or
                                        (last_raw is not None and
                                         math.hypot(mx - last_raw[0], my - last_raw[1]) > MAX_JUMP * 1.2)):
                                    pen = 0
                                    reset_filters()
                                    last_draw = None
                                    last_raw = None
                                    sx, sy = mx, my
                            if pen:
                                last_draw = (sx, sy)
                                last_raw = (mx, my)
                            else:
                                sx, sy = mx, my

                        if STITCH:
                            if pen:
                                # Pen-down resume: attach the start to the last displayed pen-stop position
                                if prev_pen == 0 and last_disp is not None:
                                    disp_off[0] = last_disp[0] - sx
                                    disp_off[1] = last_disp[1] - sy
                                dx = sx + disp_off[0]
                                dy = sy + disp_off[1]
                                last_disp = (dx, dy)
                                state["x"] = dx; state["y"] = dy
                            state["pen"] = pen
                        else:
                            state["x"] = sx; state["y"] = sy; state["pen"] = pen

                        prev_pen = pen
                        state["rx"] = rx; state["ry"] = ry; state["rz"] = rz
                        state["live"] = 1; state["seq"] += 1
                        csv_writer.writerow(["%.4f" % t, "%.5f" % rx, "%.5f" % ry,
                                             "%.5f" % rz, 1, pen])
                    else:
                        if prev_pen:
                            reset_filters()
                        zpen.reset()
                        last_draw = None
                        last_raw = None
                        state["pen"] = 0
                        state["live"] = 0
                        state["seq"] += 1      # pen up / leaving the field is also a state change, let /stream push it
                        prev_pen = 0
                        csv_writer.writerow(["%.4f" % t, "", "", "", 0, 0])
                    _count += 1
                    if _count % 50 == 0:
                        csv_file.flush()
        time.sleep(0.001)
      except OSError:
        # Occasional I2C / GPIO IO error: skip, don't let the reader thread crash
        time.sleep(0.002)
      except Exception as exc:  # noqa: BLE001
        # Unexpected errors like bad frames: if the reader thread dies the page "freezes", log and continue
        print("reader error (skipped): %r" % (exc,))
        time.sleep(0.002)


threading.Thread(target=reader, daemon=True).start()
app = Flask(__name__)

# ---------------- Recognition + text-to-image (combine multiple shapes into a scene) ----------------
scene = []   # completed shapes: [{"en": "..."}]


def recognize_png(png_bytes):
    """Call Gemini to recognize a sketch, return the label."""
    from google import genai
    from google.genai import types
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set on the Pi (put it in key_local.py or an env var)")
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model=VISION_MODEL,
        contents=[types.Part.from_bytes(data=png_bytes, mime_type="image/png"), SYS_PROMPT],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    info = json.loads(resp.text)
    return info.get("label", "?")


def gen_scene_png(prompt):
    """Pollinations free text-to-image, returns PNG bytes."""
    url = ("https://image.pollinations.ai/prompt/"
           + urllib.parse.quote(prompt)
           + "?width=768&height=768&nologo=true&model=flux")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<style>
body{margin:0;background:#111;overflow:hidden;font-family:sans-serif}
canvas#c{display:block;background:#fff}
#hud{position:fixed;left:8px;top:6px;font:13px monospace;color:#0f0;
     background:rgba(0,0,0,.55);padding:4px 8px;border-radius:4px;white-space:pre;z-index:5}
#panel{position:fixed;left:8px;bottom:8px;background:rgba(0,0,0,.72);color:#fff;
       padding:10px 12px;border-radius:8px;z-index:5;max-width:340px}
#status{font-size:13px;margin-bottom:6px;color:#bfe}
#chips span{display:inline-block;background:#356;padding:3px 9px;margin:2px;border-radius:12px;font-size:13px}
#panel button{font-size:15px;padding:8px 12px;margin:6px 6px 0 0;border:0;border-radius:6px;cursor:pointer;color:#fff}
#bFin{background:#2a8}#bGen{background:#c63}#bClr{background:#555}
#result{position:fixed;right:8px;bottom:8px;width:40vw;max-width:520px;display:none;
        border:3px solid #fff;border-radius:8px;z-index:6;background:#fff}
#mm{position:fixed;right:8px;top:8px;border:1px solid #888;border-radius:4px;background:#fff;z-index:5}
#mmlbl{position:fixed;right:8px;top:132px;font:11px monospace;color:#9c9;z-index:5}
</style></head><body>
<div id=hud></div>
<canvas id=c></canvas>
<canvas id=mm width=120 height=120></canvas>
<div id=mmlbl>active area</div>
<img id=result>
<div id=panel>
  <div id=status>Draw a shape → click "Finish shape" → more shapes → "Generate scene"</div>
  <div id=chips></div>
  <button id=bFin onclick=finishShape()>✓ Finish shape</button>
  <button id=bGen onclick=generateScene()>🎨 Generate scene</button>
  <button id=bClr onclick=clearAll()>🗑 Clear</button>
</div>
<script>
const cv=document.getElementById('c'), hud=document.getElementById('hud');
const mm=document.getElementById('mm'), mctx=mm.getContext('2d');
cv.width=innerWidth; cv.height=innerHeight;
const ctx=cv.getContext('2d');
const SCALE=Math.min(cv.width,cv.height)*1.15;  // slightly smaller, so a small hand move doesn't push the canvas too far
const FOLLOW=0.22;                              // camera follow (a bit steadier)
const Z_MAX=0.6;                                // matches backend Z_MAX (position-distortion warning)
const MIN_PT=0.004;                             // skip points that are too close, reduces jaggies
const OUT_EDGE=0.04;                            // matches backend: reliable-zone margin
const MAX_PTS=8000;                             // cap on total stroke history points, drop oldest strokes past it (avoids frame drops in long sessions)
let strokes=[[]], penNow=0, mnx=1,mxx=0,mny=1,mxy=0, satNow=false, zNow=0, totalPts=0;
let cur=[0.5,0.5], cam=null, rawNow=[0.5,0.5], liveNow=0;
function toXY(pt){return[cv.width/2+(pt[0]-cam[0])*SCALE,cv.height/2+(pt[1]-cam[1])*SCALE];}
function strokeCurve(c,pts){
  if(pts.length<2)return;
  c.beginPath();
  const p0=toXY(pts[0]); c.moveTo(p0[0],p0[1]);
  if(pts.length===2){const p1=toXY(pts[1]);c.lineTo(p1[0],p1[1]);c.stroke();return;}
  for(let i=1;i<pts.length-1;i++){
    const p=toXY(pts[i]),n=toXY(pts[i+1]);
    c.quadraticCurveTo(p[0],p[1],(p[0]+n[0])/2,(p[1]+n[1])/2);}
  const last=toXY(pts[pts.length-1]); c.lineTo(last[0],last[1]); c.stroke();
}
function setStatus(t){document.getElementById('status').textContent=t;}
new EventSource('/stream').onmessage=e=>{
  const p=JSON.parse(e.data);            // [x,y,pen,rawx,rawy,rawz,live]
  penNow=p[2]; zNow=p[5]; rawNow=[p[3],p[4]]; liveNow=p[6];
  satNow=(p[3]<=0.02||p[3]>=0.98||p[4]<=0.02||p[4]>=0.98);
  cur=[p[0],p[1]];
  if(cam===null) cam=cur.slice();
  if(p[2]){
    const s=strokes[strokes.length-1], last=s.length?s[s.length-1]:null;
    if(!last||Math.hypot(p[0]-last[0],p[1]-last[1])>=MIN_PT){
      s.push([p[0],p[1]]); totalPts++;
      while(totalPts>MAX_PTS&&strokes.length>1){totalPts-=strokes[0].length;strokes.shift();}
    }
  }
  else if(strokes[strokes.length-1].length){ strokes.push([]); }
  if(p[3]||p[4]){mnx=Math.min(mnx,p[3]);mxx=Math.max(mxx,p[3]);
                 mny=Math.min(mny,p[4]);mxy=Math.max(mxy,p[4]);}
};
function drawMM(){
  const W=mm.width,H=mm.height;
  mctx.fillStyle='#fff';mctx.fillRect(0,0,W,H);
  mctx.strokeStyle='#bbb';mctx.lineWidth=1;mctx.strokeRect(1,1,W-2,H-2);
  const m=OUT_EDGE*W;                            // reliable zone (outermost edge removed)
  mctx.strokeStyle='#3a3';mctx.lineWidth=2;mctx.strokeRect(m,m,W-2*m,H-2*m);
  const px=cur[0]*W, py=cur[1]*H;                 // same mapped coordinates as the canvas, consistent with the hand
  mctx.fillStyle=(satNow||zNow>Z_MAX)?'#e33':'#1a1';
  mctx.beginPath();mctx.arc(px,py,5,0,7);mctx.fill();
}
function render(){
  ctx.clearRect(0,0,cv.width,cv.height);
  if(cam){
    cam[0]+=(cur[0]-cam[0])*FOLLOW;
    cam[1]+=(cur[1]-cam[1])*FOLLOW;
    ctx.lineWidth=4; ctx.lineCap='round'; ctx.lineJoin='round'; ctx.strokeStyle='#000';
    for(const s of strokes){ if(s.length<2)continue; strokeCurve(ctx,s); }
  }
  // Camera center = current hand/pen position. Fixed marker: small circle inside a big one
  // (blue = height OK, lower to draw; red = hand too high / out of range, lower it or move back; small circle filled while pen is down)
  {
    const hx=cv.width/2, hy=cv.height/2;
    const bad=(zNow>Z_MAX)||satNow, col=bad?'#e33':(penNow?'#111':'#2a7bff');
    ctx.save();
    ctx.strokeStyle=col; ctx.fillStyle=col; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(hx,hy,9,0,7); ctx.stroke();            // big circle
    ctx.beginPath(); ctx.arc(hx,hy,3,0,7);                          // small circle
    penNow ? ctx.fill() : ctx.stroke();                             // pen down = filled
    ctx.restore();
  }
  drawMM();
  hud.textContent='z='+(zNow||0).toFixed(2)+'  '+(penNow?'● pen down':'○ pen up')
    +(satNow?'  ⚠out of range':(zNow>Z_MAX?'  ⚠too high':''));
  hud.style.color = (satNow||zNow>Z_MAX) ? '#f55' : (penNow ? '#0f0' : '#8cf');
  requestAnimationFrame(render);
}
render();
// Render current strokes into a 512x512 white-background black-line PNG (bbox centered), for recognition
function shapePNG(){
  let any=false,a=1e9,b=-1e9,c=1e9,d=-1e9;
  for(const s of strokes)for(const pt of s){any=true;
    a=Math.min(a,pt[0]);b=Math.max(b,pt[0]);c=Math.min(c,pt[1]);d=Math.max(d,pt[1]);}
  if(!any) return null;
  const w=Math.max(b-a,1e-3),h=Math.max(d-c,1e-3);
  const off=document.createElement('canvas');off.width=512;off.height=512;
  const o=off.getContext('2d');o.fillStyle='#fff';o.fillRect(0,0,512,512);
  const pad=60,sc=Math.min((512-2*pad)/w,(512-2*pad)/h),cx=(a+b)/2,cy=(c+d)/2;
  o.strokeStyle='#000';o.lineWidth=6;o.lineCap='round';o.lineJoin='round';
  for(const s of strokes){ if(s.length<2)continue;o.beginPath();
    o.moveTo(256+(s[0][0]-cx)*sc,256+(s[0][1]-cy)*sc);
    if(s.length===2){o.lineTo(256+(s[1][0]-cx)*sc,256+(s[1][1]-cy)*sc);}
    else{for(let i=1;i<s.length-1;i++){
      const px=256+(s[i][0]-cx)*sc,py=256+(s[i][1]-cy)*sc;
      const nx=256+(s[i+1][0]-cx)*sc,ny=256+(s[i+1][1]-cy)*sc;
      o.quadraticCurveTo(px,py,(px+nx)/2,(py+ny)/2);}
      const L=s.length-1;
      o.lineTo(256+(s[L][0]-cx)*sc,256+(s[L][1]-cy)*sc);}
    o.stroke();}
  return off.toDataURL('image/png');
}
function clearCanvas(){strokes=[[]];totalPts=0;cam=null;mnx=1;mxx=0;mny=1;mxy=0;fetch('/reset').catch(()=>{});}
function finishShape(){
  const img=shapePNG();
  if(!img){setStatus('Draw something first');return;}
  setStatus('Recognizing…');
  fetch('/finish',{method:'POST',headers:{'Content-Type':'application/json'},
                   body:JSON.stringify({img:img})})
    .then(r=>r.json()).then(j=>{
      if(j.error){setStatus('Recognition failed: '+j.error);return;}
      const sp=document.createElement('span');sp.textContent=j.en;
      document.getElementById('chips').appendChild(sp);
      setStatus('Added: '+j.en+'. Draw the next shape, or click "Generate scene"');
      clearCanvas();
    }).catch(e=>setStatus('Error: '+e));
}
function generateScene(){
  const im=document.getElementById('result');
  setStatus('Generating scene… (about 10-20 s)');
  im.onload=()=>setStatus('Done!');
  im.onerror=()=>setStatus('Generation failed (empty scene or network issue)');
  im.src='/generate?ts='+Date.now();
  im.style.display='block';
}
function clearAll(){
  fetch('/clearscene').catch(()=>{});
  document.getElementById('chips').innerHTML='';
  document.getElementById('result').style.display='none';
  clearCanvas();setStatus('Cleared, start over');
}
document.ondblclick=clearCanvas;
</script></body></html>"""


@app.route('/')
def index():
    return PAGE


@app.route('/reset')
def reset():
    global reset_flag
    reset_flag = True
    return "ok"


@app.route('/finish', methods=['POST'])
def finish():
    try:
        data = request.get_json(force=True)
        b64 = data["img"].split(",", 1)[1]
        png = base64.b64decode(b64)
        en = recognize_png(png)
        scene.append({"en": en})
        return jsonify({"en": en})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)})


@app.route('/generate')
def generate():
    if not scene:
        return "no shapes", 400
    items = [s["en"] for s in scene]
    joined = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    prompt = ("A simple, clean line illustration of " + joined +
              ", together in one scene, white background, centered")
    try:
        png = gen_scene_png(prompt)
        return Response(png, mimetype="image/png")
    except Exception as exc:  # noqa: BLE001
        return str(exc), 500


@app.route('/clearscene')
def clearscene():
    scene.clear()
    return "ok"


@app.route('/stream')
def stream():
    def gen():
        last_seq = -1
        idle = 0.0
        while True:
            s = state
            # Push only the "latest point" (~60Hz max), no backlog => tracks the hand;
            # unchanged seq means no new frame, skip; send a heartbeat every 0.5s to keep alive
            if s["seq"] != last_seq or idle >= 0.5:
                last_seq = s["seq"]
                idle = 0.0
                yield "data: " + json.dumps([round(s["x"], 4), round(s["y"], 4),
                                             s["pen"], round(s["rx"], 3),
                                             round(s["ry"], 3), round(s["rz"], 3),
                                             s["live"]]) + "\n\n"
            time.sleep(0.016)
            idle += 0.016
    return Response(gen(), mimetype='text/event-stream')


app.run(host='0.0.0.0', port=5000, threaded=True)
