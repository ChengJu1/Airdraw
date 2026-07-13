#!/usr/bin/env python3
"""Sensor working-range calibration (terminal only, no screen/pygame needed).

Usage (on the Pi):
    python3 ~/calibrate_range.py
    Put your hand into the sensing area, slowly trace along the four edges and corners
    for 2-3 laps, sweep the middle a few times too; 30-60 seconds is enough.
    Ctrl+C to stop; the measured range and suggested draw_app.py config values are printed,
    and every raw frame is recorded to ~/captures/cal_*.csv (t,x,y,z,valid).

Then copy the CSV back to the computer for analysis:
    (on the Mac) scp chengju@<pi-ip>:~/captures/cal_*.csv ./data/
"""
import csv
import json
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

print("recording -> %s" % csv_path)
print("Slowly circle your hand along the edges/corners of the sensing area, Ctrl+C to stop\n")

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
                                "z=%.3f  frames=%d valid=%d " % (
                                    rx, min(xs), max(xs),
                                    ry, min(ys), max(ys),
                                    rz, n_all, n_valid))
                        else:
                            sys.stdout.write(
                                "\r(no hand)                    frames=%d valid=%d              "
                                % (n_all, n_valid))
                        sys.stdout.flush()
        time.sleep(0.001)
except KeyboardInterrupt:
    pass
finally:
    f.flush(); f.close()


def pct(v, p):
    """p-th percentile (0-100), used to drop edge-clipping spikes."""
    s = sorted(v)
    k = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


print("\n\n===== calibration result =====")
print("total frames %d, valid %d (%.0f%%), CSV -> %s" % (
    n_all, n_valid, 100.0 * n_valid / max(1, n_all), csv_path))
if len(xs) < 50:
    print("Too little valid data, record again (hold hand lower, move slower).")
else:
    print("x: full range [%.3f ~ %.3f]   2%%~98%% percentile [%.3f ~ %.3f]" % (
        min(xs), max(xs), pct(xs, 2), pct(xs, 98)))
    print("y: full range [%.3f ~ %.3f]   2%%~98%% percentile [%.3f ~ %.3f]" % (
        min(ys), max(ys), pct(ys, 2), pct(ys, 98)))
    print("z: full range [%.3f ~ %.3f]   2%%~98%% percentile [%.3f ~ %.3f]" % (
        min(zs), max(zs), pct(zs, 2), pct(zs, 98)))
    print("\nSuggested draw_app.py config (2%%~98%% percentile, excludes edge-clipping spikes):")
    print("    X_LO, X_HI = %.2f, %.2f" % (pct(xs, 2), pct(xs, 98)))
    print("    Y_LO, Y_HI = %.2f, %.2f" % (pct(ys, 2), pct(ys, 98)))

    # Write calibration file, auto-loaded by draw_app.py at startup (saturated-clipping ends clamped into 0.05~0.95)
    xl, xh = max(pct(xs, 2), 0.05), min(pct(xs, 98), 0.95)
    yl, yh = max(pct(ys, 2), 0.05), min(pct(ys, 98), 0.95)
    if xh - xl >= 0.3 and yh - yl >= 0.3:
        cal = {"X_LO": round(xl, 3), "X_HI": round(xh, 3),
               "Y_LO": round(yl, 3), "Y_HI": round(yh, 3)}
        cal_path = os.path.expanduser("~/range_cal.json")
        with open(cal_path, "w") as jf:
            json.dump(cal, jf)
        print("\nwritten to %s (applied automatically when draw_app.py starts):" % cal_path)
        print("    %s" % json.dumps(cal))
    else:
        print("\nUsable range of some axis is below 0.3, calibration file not written -- check sensor/environment and record again.")
