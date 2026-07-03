#!/usr/bin/env python3
"""Skywriter 空中作画：实时跟手显示 + 录制 CSV（纯算法优化，无需按钮）。

针对「不跟手 / 形状乱 / 抬落笔迟钝」做的算法优化：
  1. 只推“最新点”给浏览器（不排队）—— 消除延迟积压，跟手
  2. 1€ 滤波 —— 慢动作稳、快动作几乎零延迟的平滑
  3. 有效范围校准 + 拉伸铺满画布 —— 解决“画布小/挤中间”
  4. 出界(边缘饱和) + 跳变 自动断笔 —— 无需按钮的抬笔/落笔，去掉贯穿直线
  5. FLIP_X / FLIP_Y / SWAP_XY —— 把方向调到跟手直觉

录制: 每帧写 CSV(t, x, y, z, in_range, pen)，x/y 为原始坐标，pen 为算法判定的落笔。

用法(树莓派, venv 内；需连同 mgc3130.py、rt_filters.py 拷到同一目录):
  source ~/sky/bin/activate
  python3 ~/web_capture.py            # 存 ~/captures/cap_<时间>.csv
  python3 ~/web_capture.py circle     # 存 ~/captures/circle.csv
浏览器开 http://Dissertation.local:5000 边画边看；左上角显示原始 x/y 实测范围，
用来校准下面的 X_LO/X_HI/Y_LO/Y_HI。画完按 Ctrl+C 保存。
启动时手先离开传感器(让它干净校准)，再放上去画。
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

# 自动从 key_local.py 读 GEMINI_API_KEY（该文件已被 .gitignore 排除）
try:
    import key_local as _k
    if getattr(_k, "GEMINI_API_KEY", "") and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = _k.GEMINI_API_KEY
except ImportError:
    pass

# 识别+文生图模型
VISION_MODEL = "gemini-2.5-flash"
SYS_PROMPT = (
    "你是手绘草图识别助手。用户给你一张粗糙的单色简笔画，请判断它最可能是什么物体，"
    "只输出 JSON：{\"label\": \"english object name\", \"label_zh\": \"中文物体名\"}。"
)

# ============== 可调参数（先跑起来看左上角实测范围再调） ==============
# 1) 方向：先默认翻 y（屏幕 y 向下、传感器 y 向上）；不跟手就改这三个
FLIP_X = False
FLIP_Y = True
SWAP_XY = False

# 2) 有效范围：直接用满量程(0~1)，画面大小交给前端“自动居中+自适应缩放”
X_LO, X_HI = 0.0, 1.0
Y_LO, Y_HI = 0.0, 1.0
# 抬笔判定：默认不再用 x/y 饱和来抬笔(避免“往右一饱和就断笔”)。
# 出界饱和保护：手移出感应场时 rx/ry 会被钉在 0 或 1，沿边缘画出方框。
# OUT_EDGE>0 时，靠近任一边缘 OUT_EDGE 范围内就判为“抬笔”，方框消失。
# 0.04 ≈ 丢掉最外侧 4% 不可靠区；画得太小就调小，方框还在就调大。
OUT_EDGE = 0.04

# z 抬笔：双阈值迟滞 + 连续帧去抖(rt_filters.ZPenHysteresis)。
# 旧版单阈值(Z_CUT=0.28)在临界高度会因 z 噪声抖动断笔；
# 现在 z < 0.26 连续 2 帧落笔、z > 0.32 连续 2 帧抬笔，中间保持原状态。
# 阈值统一在 rt_filters.Z_PEN_DOWN / Z_PEN_UP 调。
Z_MAX = 0.60              # 太高位置失真，强制抬笔

# 3) 滤波链：3 点中值 -> 尖刺门控 -> 1€(z 越高越稳)。
#    窗口/截止频率等参数统一放在 rt_filters.py 顶部，draw_app.py 同用一套。
Z_SMOOTH_T0 = 0.12        # z 超过此值逐步加强平滑(接近抬笔阈值时位置更噪)

# 4) 跳变保护：相邻落笔点距离(映射后 0~1)超过它 => 断笔，不连长线
#    正常绘画每帧约 0.003；“快速移动到起点”的位移大得多。
#    设小一点 => 接近/重定位的快速移动自动判为抬笔，不会画出引线
#    (你放慢手真正开始画的那一点 = 起始点)。太跟手不够就调大。
MAX_JUMP = 0.05

# 5) 缝合：抬笔再落笔时，把起点接到“上次停笔的位置”。
#    已有相机跟随后缝合弊大于利(出界回来会扯长线)，默认关闭 => 绝对位置作图，
#    出界/抬笔即结束当前笔，回来是全新一笔，永不连线。
STITCH = False
# =====================================================================

sensor = MGC3130()

# ---------------- 滤波链(实现与参数见 rt_filters.py) ----------------
med_f = MedianWin()
spike = SpikeGate()
fx = OneEuro()
fy = OneEuro()
zpen = ZPenHysteresis()


def z_mincutoff(rz):
    """手越高(z 接近抬笔阈值)位置越噪，略加强低通。"""
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
print("录制中 -> %s" % csv_path)
print("浏览器开 http://<树莓派>:5000 边画边看，Ctrl+C 停止保存。")


@atexit.register
def _close():
    try:
        csv_file.flush(); csv_file.close()
        print("\n已保存 -> %s" % csv_path)
    except Exception:
        pass


# 共享“最新点”状态（只保留最新，避免积压）
state = {"x": 0.5, "y": 0.5, "pen": 0, "rx": 0.0, "ry": 0.0, "rz": 0.0, "seq": 0, "live": 0}
t0 = time.time()
_count = 0

# 缝合用的全局状态
disp_off = [0.0, 0.0]   # 显示偏移：忽略抬笔期间位移
last_disp = None        # 上次显示(停笔)位置
prev_pen = 0
reset_flag = False


def reader():
    global _count, last_disp, prev_pen, reset_flag
    last_ts = None
    last_draw = None        # 滤波后最近一个落笔点，用于跳变判定
    last_raw = None         # 映射后原始点，用于尖刺/重定位判定
    while True:
      try:
        if sensor.data_ready():
            d = sensor.read_frame()
            if d and len(d) >= 26 and d[3] == 0x91:
                ts = d[6]
                if ts != last_ts:
                    last_ts = ts
                    t = time.time() - t0
                    if reset_flag:          # 双击画面 -> 重置缝合(手不在感应区时也生效)
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
                        # z 迟滞判抬落笔；边缘饱和 或 z 过高(位置失真) => 强制抬笔
                        edge_out = (rx < OUT_EDGE or rx > 1 - OUT_EDGE or
                                    ry < OUT_EDGE or ry > 1 - OUT_EDGE)
                        pen = zpen(rz)
                        if edge_out or rz > Z_MAX:
                            pen = 0
                            zpen.reset()

                        # 方向 + 范围拉伸
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
                            sx, sy = mx, my       # 抬笔时仍跟手显示，但不滤波
                        else:
                            if not prev_pen:
                                reset_filters()
                            mx, my = med_f(mx, my)
                            mx, my = spike(mx, my)
                            mc = z_mincutoff(rz)
                            sx = fx(mx, t, mc)
                            sy = fy(my, t, mc)
                            # 跳变保护：滤波后 + 原始坐标都检查，避免 1€ 拖尾画出长线
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
                                # 落笔恢复：把起点接到上次停笔的显示位置
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
                        state["seq"] += 1      # 抬笔/离场也是状态变化，让 /stream 推出去
                        prev_pen = 0
                        csv_writer.writerow(["%.4f" % t, "", "", "", 0, 0])
                    _count += 1
                    if _count % 50 == 0:
                        csv_file.flush()
        time.sleep(0.001)
      except OSError:
        # 偶发 I2C / GPIO IO 错误：跳过，别让读取线程崩
        time.sleep(0.002)
      except Exception as exc:  # noqa: BLE001
        # 坏帧等意外错误：读取线程一死画面就"卡住"，打日志继续
        print("reader 异常(已跳过): %r" % (exc,))
        time.sleep(0.002)


threading.Thread(target=reader, daemon=True).start()
app = Flask(__name__)

# ---------------- 识别 + 文生图（多图形组合成场景） ----------------
scene = []   # 已完成的图形：[{"en": "...", "zh": "..."}]


def recognize_png(png_bytes):
    """调用 Gemini 识别一张简笔画，返回 (en, zh)。"""
    from google import genai
    from google.genai import types
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("树莓派上没有 GEMINI_API_KEY（放到 key_local.py 或环境变量）")
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model=VISION_MODEL,
        contents=[types.Part.from_bytes(data=png_bytes, mime_type="image/png"), SYS_PROMPT],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    info = json.loads(resp.text)
    return info.get("label", "?"), info.get("label_zh", info.get("label", "?"))


def gen_scene_png(prompt):
    """Pollinations 免费文生图，返回 PNG bytes。"""
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
<div id=mmlbl>可操作范围</div>
<img id=result>
<div id=panel>
  <div id=status>画一个图形 → 点“完成图形” → 多个图形 → “生成画面”</div>
  <div id=chips></div>
  <button id=bFin onclick=finishShape()>✓ 完成图形</button>
  <button id=bGen onclick=generateScene()>🎨 生成画面</button>
  <button id=bClr onclick=clearAll()>🗑 清空</button>
</div>
<script>
const cv=document.getElementById('c'), hud=document.getElementById('hud');
const mm=document.getElementById('mm'), mctx=mm.getContext('2d');
cv.width=innerWidth; cv.height=innerHeight;
const ctx=cv.getContext('2d');
const SCALE=Math.min(cv.width,cv.height)*1.15;  // 略缩小，手移一点画布不会跑太远
const FOLLOW=0.22;                              // 镜头跟随(稳一点)
const Z_MAX=0.6;                                // 与后端 Z_MAX 对应(位置失真警告)
const MIN_PT=0.004;                             // 太近的点不记，减少毛刺
const OUT_EDGE=0.04;                            // 与后端一致：可靠区边距
const MAX_PTS=8000;                             // 历史笔画总点数上限，超出丢最老的笔(防长时间作画掉帧)
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
  const m=OUT_EDGE*W;                            // 可靠区(去掉最外侧边缘)
  mctx.strokeStyle='#3a3';mctx.lineWidth=2;mctx.strokeRect(m,m,W-2*m,H-2*m);
  const px=cur[0]*W, py=cur[1]*H;                 // 用画布同一套映射坐标，与手一致
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
  // 镜头中心 = 手/笔的当前位置。大圈套小圈的固定标记
  // (蓝=高度OK,落下即可画; 红=手太高/出界，先压低或回中; 落笔时小圈为实心)
  {
    const hx=cv.width/2, hy=cv.height/2;
    const bad=(zNow>Z_MAX)||satNow, col=bad?'#e33':(penNow?'#111':'#2a7bff');
    ctx.save();
    ctx.strokeStyle=col; ctx.fillStyle=col; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(hx,hy,9,0,7); ctx.stroke();            // 大圈
    ctx.beginPath(); ctx.arc(hx,hy,3,0,7);                          // 小圈
    penNow ? ctx.fill() : ctx.stroke();                             // 落笔=实心
    ctx.restore();
  }
  drawMM();
  hud.textContent='z='+(zNow||0).toFixed(2)+'  '+(penNow?'● 落笔':'○ 抬笔')
    +(satNow?'  ⚠出界':(zNow>Z_MAX?'  ⚠太高':''));
  hud.style.color = (satNow||zNow>Z_MAX) ? '#f55' : (penNow ? '#0f0' : '#8cf');
  requestAnimationFrame(render);
}
render();
// 把当前 strokes 渲染成 512x512 白底黑线 PNG(bbox 居中)，用于识别
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
  if(!img){setStatus('先画点东西再完成');return;}
  setStatus('识别中…');
  fetch('/finish',{method:'POST',headers:{'Content-Type':'application/json'},
                   body:JSON.stringify({img:img})})
    .then(r=>r.json()).then(j=>{
      if(j.error){setStatus('识别失败: '+j.error);return;}
      const sp=document.createElement('span');sp.textContent=j.zh+' ('+j.en+')';
      document.getElementById('chips').appendChild(sp);
      setStatus('已加入: '+j.zh+'。继续画下一个，或点“生成画面”');
      clearCanvas();
    }).catch(e=>setStatus('错误: '+e));
}
function generateScene(){
  const im=document.getElementById('result');
  setStatus('生成场景中…(约 10-20 秒)');
  im.onload=()=>setStatus('完成！');
  im.onerror=()=>setStatus('生成失败(场景为空或网络问题)');
  im.src='/generate?ts='+Date.now();
  im.style.display='block';
}
function clearAll(){
  fetch('/clearscene').catch(()=>{});
  document.getElementById('chips').innerHTML='';
  document.getElementById('result').style.display='none';
  clearCanvas();setStatus('已清空，重新开始');
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
        en, zh = recognize_png(png)
        scene.append({"en": en, "zh": zh})
        return jsonify({"en": en, "zh": zh})
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
            # 只推“最新点”（约 60Hz 上限），不积压 => 跟手；
            # seq 没变说明没有新帧，跳过，隔 0.5s 发一次心跳保活
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
