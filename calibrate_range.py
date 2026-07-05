#!/usr/bin/env python3
"""传感器可工作范围标定（纯终端，无需屏幕/pygame）。

用法(Pi 上):
    python3 ~/calibrate_range.py
    手伸进感应区，沿四边和四角慢慢绕 2-3 圈，中间也扫几下，30-60 秒即可。
    Ctrl+C 结束，会打印实测范围和建议的 draw_app.py 配置值，
    并把每帧原始数据录到 ~/captures/cal_*.csv (t,x,y,z,valid)。

之后把 CSV 拷回电脑分析:
    (Mac 上) scp chengju@<Pi的IP>:~/captures/cal_*.csv ./data/
"""
import csv
import os
import sys
import time

from mgc3130 import MGC3130, parse_frame

sensor = MGC3130()

name = sys.argv[1] if len(sys.argv) > 1 else time.strftime("cal_%Y%m%d_%H%M%S")
outdir = os.path.expanduser("~/captures"); os.makedirs(outdir, exist_ok=True)
csv_path = os.path.join(outdir, name + ".csv")
f = open(csv_path, "w", newline="")
w = csv.writer(f)
w.writerow(["t", "x", "y", "z", "valid"])

xs, ys, zs = [], [], []
n_all = n_valid = 0
t0 = time.time()
last_ts = None
last_print = 0.0

print("录制中 -> %s" % csv_path)
print("手沿感应区四边/四角慢慢绕圈，Ctrl+C 结束\n")

try:
    while True:
        if sensor.data_ready():
            d = sensor.read_frame()
            if d and len(d) >= 26 and d[3] == 0x91:
                ts = d[6]
                if ts != last_ts:
                    last_ts = ts
                    t = time.time() - t0
                    rx, ry, rz, valid = parse_frame(d)
                    n_all += 1
                    if valid:
                        n_valid += 1
                        xs.append(rx); ys.append(ry); zs.append(rz)
                        w.writerow(["%.4f" % t, "%.5f" % rx, "%.5f" % ry,
                                    "%.5f" % rz, 1])
                    else:
                        w.writerow(["%.4f" % t, "", "", "", 0])
                    if n_all % 100 == 0:
                        f.flush()
                    now = time.time()
                    if now - last_print > 0.1:
                        last_print = now
                        if valid and xs:
                            sys.stdout.write(
                                "\rx=%.3f [%.3f~%.3f]  y=%.3f [%.3f~%.3f]  "
                                "z=%.3f  帧=%d 有效=%d " % (
                                    rx, min(xs), max(xs),
                                    ry, min(ys), max(ys),
                                    rz, n_all, n_valid))
                        else:
                            sys.stdout.write(
                                "\r(无手)                    帧=%d 有效=%d              "
                                % (n_all, n_valid))
                        sys.stdout.flush()
        time.sleep(0.001)
except KeyboardInterrupt:
    pass
finally:
    f.flush(); f.close()


def pct(v, p):
    """p 百分位(0-100)，去掉贴边毛刺用。"""
    s = sorted(v)
    k = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


print("\n\n===== 标定结果 =====")
print("总帧数 %d，有效 %d (%.0f%%)，CSV -> %s" % (
    n_all, n_valid, 100.0 * n_valid / max(1, n_all), csv_path))
if len(xs) < 50:
    print("有效数据太少，重新录一次(手放低些、动慢些)。")
else:
    print("x: 全范围 [%.3f ~ %.3f]   2%%~98%% 分位 [%.3f ~ %.3f]" % (
        min(xs), max(xs), pct(xs, 2), pct(xs, 98)))
    print("y: 全范围 [%.3f ~ %.3f]   2%%~98%% 分位 [%.3f ~ %.3f]" % (
        min(ys), max(ys), pct(ys, 2), pct(ys, 98)))
    print("z: 全范围 [%.3f ~ %.3f]   2%%~98%% 分位 [%.3f ~ %.3f]" % (
        min(zs), max(zs), pct(zs, 2), pct(zs, 98)))
    print("\n建议 draw_app.py 配置(用 2%%~98%% 分位，排除贴边毛刺):")
    print("    X_LO, X_HI = %.2f, %.2f" % (pct(xs, 2), pct(xs, 98)))
    print("    Y_LO, Y_HI = %.2f, %.2f" % (pct(ys, 2), pct(ys, 98)))
