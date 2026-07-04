"""MGC3130 (Skywriter) pure-I2C reader: reset, runtime config, XFER-handshake frame reads.

Shared by the three Raspberry Pi scripts capture_pi.py / web_capture.py / draw_app.py.
When deploying to the Pi, copy this file together with rt_filters.py.
"""
import time

import lgpio
from smbus2 import SMBus, i2c_msg

RESET, XFER, ADDR, BUS = 17, 27, 0x42, 1

# GestureInfo (frame byte 10) gesture codes. Firmware gesture recognition is enabled in __init__'s cfg (0x85=0x7F)
GEST_NONE, GEST_GARBAGE = 0, 1
GEST_FLICK_WE, GEST_FLICK_EW, GEST_FLICK_SN, GEST_FLICK_NS = 2, 3, 4, 5
FLICKS = (GEST_FLICK_WE, GEST_FLICK_EW, GEST_FLICK_SN, GEST_FLICK_NS)

_INIT_HINT = (
    "Skywriter I2C init failed. Check on the Pi:\n"
    "  1) sudo raspi-config -> Interface -> I2C -> Enable, then reboot\n"
    "  2) i2cdetect -y 1  should show address 42\n"
    "  3) HAT seated firmly, keep hands/metal off the board; power-cycle and reseat the Skywriter\n"
    "  4) no other program is using I2C (kill any old web_capture process)\n"
    "  5) 3A+ uses I2C bus 1; if rewired, confirm SDA/SCL are connected correctly"
)


def probe_i2c(bus=BUS, addr=ADDR):
    """Read 1 byte to probe whether the device is on the bus."""
    b = SMBus(bus)
    try:
        b.read_byte(addr)
        return True
    except OSError:
        return False
    finally:
        b.close()


class MGC3130:
    """Open GPIO/I2C, reset and write runtime config; data_ready() polls for new frames, read_frame() reads one."""

    def __init__(self, retries=5):
        self.h = lgpio.gpiochip_open(0)
        self.bus = SMBus(BUS)
        lgpio.gpio_claim_output(self.h, RESET, 0)
        time.sleep(0.15)
        lgpio.gpio_write(self.h, RESET, 1)
        time.sleep(0.8)          # wait a bit longer after reset before cfg
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
                # another round of hardware reset
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
        """XFER pulled low = sensor has a new data frame ready (same polling as the Pimoroni library)."""
        return lgpio.gpio_read(self.h, XFER) == 0

    def read_frame(self):
        """Read one 26-byte frame; returns None on sporadic I2C IO errors, caller just drops that frame."""
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
    """Standard 26-byte data frame: SystemInfo @7 (bit0=PositionValid), xyz @20..25 (16-bit little-endian).

    Returns (x, y, z, valid), coordinates normalized to 0~1.
    """
    valid = bool(d[7] & 0x01)
    x = (d[20] | d[21] << 8) / 65536.0
    y = (d[22] | d[23] << 8) / 65536.0
    z = (d[24] | d[25] << 8) / 65536.0
    return x, y, z, valid


def parse_gesture(d):
    """Gesture/AirWheel info in the same 26-byte frame (enabled in firmware, see __init__'s cfg).

    Returns (gesture_code, airwheel_active, airwheel_counter):
      gesture_code: 0 none / 2~5 four-direction flicks (see FLICKS) / 6,7 circles (rare)
      airwheel_active: SystemInfo bit1, hand is circling in the air
      airwheel_counter: 0~255 wrap-around counter, ~32 per full circle
    """
    gesture = d[10]
    aw_active = bool(d[7] & 0x02)
    aw_count = d[18]
    return gesture, aw_active, aw_count
