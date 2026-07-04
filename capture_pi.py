#!/usr/bin/env python3
"""Capture Skywriter air trajectories on the Raspberry Pi and save as CSV for the PC-side denoise/extraction pipeline.

I2C reads reuse mgc3130.py (lgpio + smbus2 pure I2C + XFER handshake), copy it along when deploying.
Data is written to disk while capturing (flush every 50 frames): if the process gets killed
unexpectedly at most the last few dozen frames are lost, and sporadic I2C IO errors only
drop the current frame without interrupting capture.

Output CSV columns: t, x, y, z, in_range
  t        relative timestamp (seconds)
  x, y, z  raw normalized coordinates (0~1). Raw values are stored here, no vertical flip;
           flipping/range mapping happens later during PC-side visualization, keeping captured data clean.
  in_range 1=hand in sensing area (PositionValid, i.e. the "pen down" criterion), 0=hand left (pen up).

Usage (on the Pi, activate the virtualenv first):
  source ~/sky/bin/activate
  python3 ~/capture_pi.py            # saves to ~/captures/cap_<time>.csv
  python3 ~/capture_pi.py circle     # saves to ~/captures/circle.csv
Press Ctrl+C when done drawing to stop and save.

Note: the first 8 frames print raw bytes plus parsed values, for checking offsets.
If x/y/z disagree with the web canvas, compare against the raw bytes and check the
parse_frame field offsets (offsets may differ across firmware versions).
"""
import os
import sys
import time
import csv

from mgc3130 import MGC3130, parse_frame


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else time.strftime("cap_%Y%m%d_%H%M%S")
    outdir = os.path.expanduser("~/captures")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name + ".csv")

    sensor = MGC3130()
    t0 = time.time()
    last_ts = None
    n = 0
    saved = 0
    print("capturing -> %s" % path)
    print("Draw above the sensor, press Ctrl+C when done to stop and save.")
    print("(first 8 frames print raw bytes for checking offsets)\n")
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["t", "x", "y", "z", "in_range"])
    try:
        while True:
            if sensor.data_ready():
                d = sensor.read_frame()
                if not d or len(d) < 26:      # sporadic I2C error => read_frame returns None, drop the frame
                    continue
                # d[6] is a timestamp byte that increments per frame; use it to dedupe repeated frames
                ts = d[6]
                if ts == last_ts:
                    continue
                last_ts = ts

                x, y, z, valid = parse_frame(d)
                t = time.time() - t0
                w.writerow(["%.4f" % t, "%.5f" % x, "%.5f" % y, "%.5f" % z, int(valid)])
                saved += 1
                if saved % 50 == 0:
                    f.flush()

                if n < 8:
                    print("raw:", " ".join("%02x" % b for b in d))
                    print("  -> x=%.3f y=%.3f z=%.3f valid=%s" % (x, y, z, valid))
                elif n % 25 == 0:
                    state = "pen down" if valid else "pen up"
                    print("x=%.3f y=%.3f z=%.3f  %s   (%d points saved)" % (x, y, z, state, saved))
                n += 1
            else:
                time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        f.flush()
        f.close()
        print("\nsaved %d points -> %s" % (saved, path))
        sensor.close()


if __name__ == "__main__":
    main()
