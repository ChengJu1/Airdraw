#!/usr/bin/env python3
"""在树莓派上采集 Skywriter 空中轨迹，存成 CSV，供电脑端降噪/提取管线使用。

I2C 读取复用 mgc3130.py(lgpio + smbus2 纯 I2C + XFER 握手)，部署时一起拷过去。
数据边采边写盘(每 50 帧 flush 一次)：进程意外被杀最多丢最后几十帧，
偶发 I2C IO 错误只丢当帧，不会中断采集。

输出 CSV 列: t, x, y, z, in_range
  t        相对时间戳(秒)
  x, y, z  原始归一化坐标(0~1)。这里存“原始值”，不做上下翻转；
           翻转/范围映射放到电脑端可视化时再处理，保证采集数据干净。
  in_range 1=手在感应区(PositionValid，即“落笔”依据)，0=手离开(抬笔)。

用法(在树莓派上，先进虚拟环境):
  source ~/sky/bin/activate
  python3 ~/capture_pi.py            # 存到 ~/captures/cap_<时间>.csv
  python3 ~/capture_pi.py circle     # 存到 ~/captures/circle.csv
画完按 Ctrl+C 结束并保存。

注意: 前 8 帧会打印原始字节(raw)+解析结果，用于核对偏移。
若 x/y/z 与 web 画布不一致，对照 raw 字节检查 parse_frame 字段偏移(固件版本不同偏移可能不同)。
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
    print("开始采集 -> %s" % path)
    print("在传感器上方画图，画完按 Ctrl+C 结束保存。")
    print("(前 8 帧打印 raw 字节用于核对偏移)\n")
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["t", "x", "y", "z", "in_range"])
    try:
        while True:
            if sensor.data_ready():
                d = sensor.read_frame()
                if not d or len(d) < 26:      # 偶发 I2C 错误 => read_frame 返回 None，丢当帧
                    continue
                # d[6] 是时间戳字节，逐帧递增；用它去重，避免记录重复帧
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
                    state = "落笔" if valid else "抬笔"
                    print("x=%.3f y=%.3f z=%.3f  %s   (已采 %d 点)" % (x, y, z, state, saved))
                n += 1
            else:
                time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        f.flush()
        f.close()
        print("\n已保存 %d 个点 -> %s" % (saved, path))
        sensor.close()


if __name__ == "__main__":
    main()
