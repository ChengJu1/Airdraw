"""MGC3130 (Skywriter) 纯 I2C 读取：复位、运行时配置、XFER 握手读帧。

capture_pi.py / web_capture.py / draw_app.py 三个树莓派脚本共用。
部署到树莓派时把本文件和 rt_filters.py 一起拷过去。
"""
import time

import lgpio
from smbus2 import SMBus, i2c_msg

RESET, XFER, ADDR, BUS = 17, 27, 0x42, 1

# GestureInfo(帧第10字节) 手势码。固件手势识别已在 __init__ 的 cfg 里启用(0x85=0x7F)
GEST_NONE, GEST_GARBAGE = 0, 1
GEST_FLICK_WE, GEST_FLICK_EW, GEST_FLICK_SN, GEST_FLICK_NS = 2, 3, 4, 5
FLICKS = (GEST_FLICK_WE, GEST_FLICK_EW, GEST_FLICK_SN, GEST_FLICK_NS)

_INIT_HINT = (
    "Skywriter I2C 初始化失败。请在 Pi 上检查:\n"
    "  1) sudo raspi-config -> Interface -> I2C -> Enable，然后 reboot\n"
    "  2) i2cdetect -y 1  应看到地址 42\n"
    "  3) HAT 插紧，手/金属勿碰板子；断电重插 Skywriter\n"
    "  4) 没有其他程序占用 I2C(关掉旧的 web_capture 进程)\n"
    "  5) 3A+ 用 I2C bus 1；若改线了确认 SDA/SCL 接对"
)


def probe_i2c(bus=BUS, addr=ADDR):
    """读 1 字节探测设备是否在总线上。"""
    b = SMBus(bus)
    try:
        b.read_byte(addr)
        return True
    except OSError:
        return False
    finally:
        b.close()


class MGC3130:
    """打开 GPIO/I2C、复位并写入运行时配置；data_ready() 轮询新帧，read_frame() 取帧。"""

    def __init__(self, retries=5):
        self.h = lgpio.gpiochip_open(0)
        self.bus = SMBus(BUS)
        lgpio.gpio_claim_output(self.h, RESET, 0)
        time.sleep(0.15)
        lgpio.gpio_write(self.h, RESET, 1)
        time.sleep(0.8)          # 复位后多等一会再 cfg
        lgpio.gpio_claim_input(self.h, XFER, lgpio.SET_PULL_UP)
        cfgs = [
            [0x00, 0x00, 0xA2, 0x90, 0x00, 0x00, 0x00, 0x20,
             0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00],
            [0x00, 0x00, 0xA2, 0x85, 0x00, 0x00, 0x00, 0b01111111,
             0x00, 0x00, 0x00, 0b01111111, 0x00, 0x00, 0x00],
            [0x00, 0x00, 0xA2, 0xA0, 0x00, 0x00, 0x00, 0b00011111,
             0x00, 0x00, 0x00, 0b00011111, 0x00, 0x00, 0x00],
        ]
        last_err = None
        for attempt in range(retries):
            if not probe_i2c():
                last_err = OSError(5, "device 0x42 not on I2C bus")
                time.sleep(0.3)
                continue
            try:
                for p in cfgs:
                    self.cfg(p)
                return
            except OSError as exc:
                last_err = exc
                # 再来一轮硬件复位
                lgpio.gpio_write(self.h, RESET, 0)
                time.sleep(0.15)
                lgpio.gpio_write(self.h, RESET, 1)
                time.sleep(0.8)
        raise OSError(last_err.errno if last_err else 5, _INIT_HINT) from last_err

    def cfg(self, p, retries=3):
        err = None
        for _ in range(retries):
            try:
                self.bus.i2c_rdwr(i2c_msg.write(ADDR, [0x10] + p))
                time.sleep(0.05)
                return
            except OSError as exc:
                err = exc
                time.sleep(0.05)
        raise err

    def data_ready(self):
        """XFER 被拉低 = 传感器有新数据帧准备好（与 Pimoroni 库同款轮询）。"""
        return lgpio.gpio_read(self.h, XFER) == 0

    def read_frame(self):
        """读一帧 26 字节；偶发 I2C IO 错误返回 None，调用方丢弃该帧即可。"""
        lgpio.gpio_free(self.h, XFER); lgpio.gpio_claim_output(self.h, XFER, 0)
        try:
            msg = i2c_msg.read(ADDR, 26); self.bus.i2c_rdwr(msg)
            return list(msg)
        except OSError:
            return None
        finally:
            lgpio.gpio_free(self.h, XFER)
            lgpio.gpio_claim_input(self.h, XFER, lgpio.SET_PULL_UP)

    def close(self):
        try:
            lgpio.gpiochip_close(self.h)
            self.bus.close()
        except Exception:
            pass


def parse_frame(d):
    """标准 26 字节数据帧：SystemInfo @7 (bit0=PositionValid)，xyz @20..25（16-bit 小端）。

    返回 (x, y, z, valid)，坐标归一化到 0~1。
    """
    valid = bool(d[7] & 0x01)
    x = (d[20] | d[21] << 8) / 65536.0
    y = (d[22] | d[23] << 8) / 65536.0
    z = (d[24] | d[25] << 8) / 65536.0
    return x, y, z, valid


def parse_gesture(d):
    """同一 26 字节帧里的手势/AirWheel 信息(固件已启用，见 __init__ 的 cfg)。

    返回 (gesture_code, airwheel_active, airwheel_counter)：
      gesture_code: 0 无 / 2~5 四方向快速划过(见 FLICKS) / 6,7 圆圈(少见)
      airwheel_active: SystemInfo bit1，手正在空中画圈
      airwheel_counter: 0~255 环形计数，每整圈约 32
    """
    gesture = d[10]
    aw_active = bool(d[7] & 0x02)
    aw_count = d[18]
    return gesture, aw_active, aw_count
