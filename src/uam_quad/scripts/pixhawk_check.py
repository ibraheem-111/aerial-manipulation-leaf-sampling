"""Read-only check of a PX4 flight controller over MAVLink: who it is, how it is set up, and live sensors.

    python scripts/pixhawk_check.py                                  # USB, /dev/ttyACM0
    python scripts/pixhawk_check.py --port /dev/ttyUSB0 --baud 57600 # TELEM1 via 3.3 V FTDI or SiK radio
    python scripts/pixhawk_check.py --port udpin:0.0.0.0:14540       # PX4 SITL
    python scripts/pixhawk_check.py --no-stream                      # identity + params only
    python scripts/pixhawk_check.py --seconds 30                     # stream attitude / IMU for 30 s
    python scripts/pixhawk_check.py --listen-only                    # send nothing; only what the board streams

Prints: USB descriptor (serial ports only), heartbeat, firmware version, board / USB IDs, SYS_AUTOSTART,
CA_ROTOR_COUNT, battery, serial-port and EKF2 aiding params, then live attitude, body rates (ω_x, ω_y, ω_z)
and IMU.

What it sends: a GCS heartbeat, MAV_CMD_REQUEST_MESSAGE (AUTOPILOT_VERSION), PARAM_REQUEST_READ and
MAV_CMD_SET_MESSAGE_INTERVAL (stream rates; not saved, gone after reboot). It never arms, never writes a
parameter, and never sends actuator, motor-test or setpoint commands.
"""
import argparse
import glob
import math
import os
import struct
import sys
import termios
import time

from pymavlink import mavutil

M = mavutil.mavlink

PARAMS = [
    # identity / airframe
    "SYS_AUTOSTART", "CA_AIRFRAME", "CA_ROTOR_COUNT", "SYS_HAS_GPS", "SYS_HAS_MAG",
    # battery / power module
    "BAT1_SOURCE", "BAT1_N_CELLS", "BAT1_V_CHARGED", "BAT1_V_EMPTY", "BAT1_CAPACITY",
    "BAT1_V_DIV", "BAT1_A_PER_V", "BAT1_I_OVERWRITE",
    # serial ports: MAVLink instances, uXRCE-DDS, baud rates
    "MAV_0_CONFIG", "MAV_0_MODE", "MAV_0_RATE", "MAV_1_CONFIG", "MAV_1_MODE",
    "UXRCE_DDS_CFG", "SER_TEL1_BAUD", "SER_TEL2_BAUD",
    # EKF2 aiding (sim mocap uses EKF2_EV_CTRL=11, EKF2_HGT_REF=3)
    "EKF2_EV_CTRL", "EKF2_HGT_REF", "EKF2_GPS_CTRL", "EKF2_MAG_TYPE",
]

INT_TYPES = {M.MAV_PARAM_TYPE_UINT8, M.MAV_PARAM_TYPE_INT8, M.MAV_PARAM_TYPE_UINT16,
             M.MAV_PARAM_TYPE_INT16, M.MAV_PARAM_TYPE_UINT32, M.MAV_PARAM_TYPE_INT32}


def enum_name(enum, value):
    e = M.enums.get(enum, {}).get(value)
    return e.name if e else str(value)


def usb_descriptor(port):
    """USB vendor/product strings from sysfs for /dev/ttyACM* and /dev/ttyUSB*, or None."""
    d = os.path.realpath(f"/sys/class/tty/{os.path.basename(port)}/device")
    while d != "/" and not os.path.exists(os.path.join(d, "idVendor")):
        d = os.path.dirname(d)
    if d == "/":
        return None

    def rd(name):
        try:
            with open(os.path.join(d, name)) as f:
                return f.read().strip()
        except OSError:
            return "?"
    return f"{rd('idVendor')}:{rd('idProduct')}  {rd('manufacturer')} / {rd('product')}  serial {rd('serial')}"


def serial_preflight(port):
    """Fail early with a useful hint if a serial port is missing or not openable."""
    if not os.path.exists(port):
        found = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
        print(f"{port} does not exist. Serial ports present: {found or 'none'}")
        if not found:
            print("  Nothing enumerated. If the board's LEDs are on but `lsusb` shows no new device, USB data is not"
                  " getting through: try a known data cable, another PC port, or TELEM1 with a 3.3 V FTDI adapter.")
        sys.exit(2)
    if not os.access(port, os.R_OK | os.W_OK):
        print(f"No read/write permission on {port}. Check `groups` includes dialout (log out and in after adding).")
        sys.exit(2)
    desc = usb_descriptor(port)
    print(f"port {port}: " + (f"USB {desc}" if desc else "no USB descriptor found"))
    holders = port_holders(port)
    if holders:                                     # two readers on one port split the byte stream: both break
        print(f"{port} is already open in: {', '.join(holders)}. Close it (QGroundControl grabs PX4 ports"
              " automatically) and run again.")
        sys.exit(2)


def port_holders(port):
    """Other processes with `port` open, from /proc (only processes we may inspect, i.e. our own user's)."""
    target, me, found = os.path.realpath(port), os.getpid(), []
    for fd_dir in glob.glob("/proc/[0-9]*/fd"):
        pid = int(fd_dir.split("/")[2])
        if pid == me:
            continue
        try:
            if any(os.path.realpath(os.path.join(fd_dir, fd)) == target for fd in os.listdir(fd_dir)):
                with open(f"/proc/{pid}/comm") as f:
                    found.append(f"{f.read().strip()} (pid {pid})")
        except OSError:
            continue
    return found


class Link:
    """MAVLink connection that keeps the latest message of each type and every PARAM_VALUE it sees."""

    def __init__(self, port, baud, listen_only=False):
        self.c = mavutil.mavlink_connection(port, baud=baud, source_system=255,
                                            source_component=M.MAV_COMP_ID_MISSIONPLANNER, autoreconnect=False)
        self.is_serial = port.startswith("/dev/")
        if self.is_serial:
            self.c.port.write_timeout = 1.0         # a board that stops accepting USB data must not hang us
        self.listen_only = listen_only
        self.latest = {}
        self.params = {}
        self.last_gcs_hb = 0.0

    @property
    def tx_ok(self):
        """False in --listen-only mode, or once a write has timed out (pymavlink then marks the port dead)."""
        return not self.listen_only and not getattr(self.c, "portdead", False)

    def close(self):
        if self.is_serial:                          # drop unsent bytes so close() doesn't wait on a stuck board
            try:
                termios.tcflush(self.c.port.fileno(), termios.TCOFLUSH)
            except (OSError, termios.error):
                pass
        self.c.close()

    def gcs_heartbeat(self):
        if not self.tx_ok:
            return
        now = time.monotonic()
        if now - self.last_gcs_hb > 1.0:
            self.c.mav.heartbeat_send(M.MAV_TYPE_GCS, M.MAV_AUTOPILOT_INVALID, 0, 0, M.MAV_STATE_ACTIVE)
            self.last_gcs_hb = now

    def pump(self, seconds):
        """Read for up to `seconds`, filing messages from the autopilot."""
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            self.gcs_heartbeat()
            m = self.c.recv_match(blocking=True, timeout=min(left, 0.2))
            if m is None:
                continue
            t = m.get_type()
            if t == "BAD_DATA":
                continue
            self.latest[t] = m
            if t == "PARAM_VALUE":
                self.params[m.param_id] = decode_param(m)

    def wait_for(self, predicate, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.pump(0.1)
            r = predicate()
            if r:
                return r
        return None

    def command(self, cmd, *p):
        p = list(p) + [0] * (7 - len(p))
        self.c.mav.command_long_send(self.c.target_system, self.c.target_component, cmd, 0, *p)


def decode_param(m):
    """PX4 encodes int params bytewise in the float field, so reinterpret the raw bytes, not the float value."""
    if m.param_type in INT_TYPES:
        buf = m.get_msgbuf()                        # param_value is the first payload field, after the header
        start = 10 if buf[0] == 0xFD else 6         # MAVLink 2 / MAVLink 1 header length
        raw = bytes(buf[start:start + 4]).ljust(4, b"\0")
        return struct.unpack("<i", raw)[0]
    return m.param_value


def wait_autopilot_heartbeat(link, timeout):
    def hb():
        m = link.latest.get("HEARTBEAT")
        if m and m.autopilot != M.MAV_AUTOPILOT_INVALID and m.type != M.MAV_TYPE_GCS:
            return m
        link.latest.pop("HEARTBEAT", None)          # was a GCS / companion heartbeat: keep waiting
        return None
    return link.wait_for(hb, timeout)


def print_heartbeat(link, hb):
    link.c.target_system, link.c.target_component = hb.get_srcSystem(), hb.get_srcComponent()
    armed = bool(hb.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED)
    framing = "MAVLink 2" if hb.get_msgbuf()[0] == 0xFD else "MAVLink 1"
    print(f"\nheartbeat   sys {hb.get_srcSystem()} comp {hb.get_srcComponent()}  ({framing} framing)")
    print(f"  autopilot {enum_name('MAV_AUTOPILOT', hb.autopilot)}   vehicle {enum_name('MAV_TYPE', hb.type)}")
    print(f"  state {enum_name('MAV_STATE', hb.system_status)}   mode {link.c.flightmode}   "
          f"{'ARMED' if armed else 'disarmed'}")
    if armed:
        print("  WARNING: vehicle reports ARMED. This script sends nothing that changes that; disarm from the RC/GCS.")
    if hb.autopilot != M.MAV_AUTOPILOT_PX4:
        print("  note: not PX4 firmware; the param names below are PX4's and may not exist.")


def print_version(link):
    link.command(M.MAV_CMD_REQUEST_MESSAGE, M.MAVLINK_MSG_ID_AUTOPILOT_VERSION)
    v = link.wait_for(lambda: link.latest.get("AUTOPILOT_VERSION"), 3)
    if v is None:                                   # older firmware: the deprecated request
        link.command(M.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES, 1)
        v = link.wait_for(lambda: link.latest.get("AUTOPILOT_VERSION"), 3)
    if v is None:
        print("\nfirmware    no AUTOPILOT_VERSION reply")
        return
    sw = v.flight_sw_version
    githash = bytes(v.flight_custom_version).rstrip(b"\0").hex() or "?"
    print(f"\nfirmware    {sw >> 24}.{(sw >> 16) & 0xff}.{(sw >> 8) & 0xff} "
          f"{enum_name('FIRMWARE_VERSION_TYPE', sw & 0xff)}   git {githash}")
    print(f"  board_version 0x{v.board_version:08x}   USB VID:PID {v.vendor_id:04x}:{v.product_id:04x}   "
          f"uid {v.uid:016x}")


def print_params(link):
    pending = list(PARAMS)
    for attempt in range(3):
        for name in pending:
            link.c.mav.param_request_read_send(link.c.target_system, link.c.target_component, name.encode(), -1)
            link.pump(0.05)
        link.wait_for(lambda: all(n in link.params for n in pending), 2.0)
        pending = [n for n in pending if n not in link.params]
        if not pending:
            break
    print("\nparameters")
    for name in PARAMS:
        val = link.params.get(name)
        if val is None:
            print(f"  {name:17s} (no reply: absent in this firmware, or link dropping packets)")
        elif isinstance(val, int):
            print(f"  {name:17s} {val}")
        else:
            print(f"  {name:17s} {val:.4g}")


def stream(link, seconds):
    for msg_id, hz in [] if not link.tx_ok else [(M.MAVLINK_MSG_ID_ATTITUDE, 10), (M.MAVLINK_MSG_ID_HIGHRES_IMU, 10),
                       (M.MAVLINK_MSG_ID_SYS_STATUS, 2), (M.MAVLINK_MSG_ID_BATTERY_STATUS, 2)]:
        link.command(M.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 1e6 / hz)
    print(f"\nlive for {seconds:g} s  (attitude deg, body rates ω rad/s, specific force m/s², battery)")
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        link.pump(0.5)
        a, imu = link.latest.get("ATTITUDE"), link.latest.get("HIGHRES_IMU")
        ss, bs = link.latest.get("SYS_STATUS"), link.latest.get("BATTERY_STATUS")
        parts = []
        if a:
            parts.append(f"rpy {math.degrees(a.roll):6.1f} {math.degrees(a.pitch):6.1f} {math.degrees(a.yaw):6.1f}"
                         f"  ω_x {a.rollspeed:+.2f} ω_y {a.pitchspeed:+.2f} ω_z {a.yawspeed:+.2f}")
        if imu:
            parts.append(f"f {imu.xacc:+5.2f} {imu.yacc:+5.2f} {imu.zacc:+5.2f}")
        if ss and ss.voltage_battery != 65535:
            amps = "" if ss.current_battery == -1 else f" {ss.current_battery / 100:.1f} A"
            pct = "" if ss.battery_remaining == -1 else f" {ss.battery_remaining}%"
            parts.append(f"bat {ss.voltage_battery / 1000:.2f} V{amps}{pct}")
        elif bs and bs.voltages[0] != 65535:        # 65535 = unknown (no battery / power module reading)
            parts.append(f"bat cells[0] {bs.voltages[0] / 1000:.2f} V")
        else:
            parts.append("bat: no reading")
        print("  " + ("  |  ".join(parts) if parts else "(no ATTITUDE / HIGHRES_IMU yet)"), flush=True)
        if "HEARTBEAT" in link.latest and link.latest["HEARTBEAT"].base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED:
            print("  WARNING: vehicle is ARMED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", default="/dev/ttyACM0", help="serial device, or a pymavlink URL such as udpin:0.0.0.0:14540")
    ap.add_argument("--baud", type=int, default=57600, help="serial baud (ignored by USB CDC; TELEM1 default 57600)")
    ap.add_argument("--no-stream", action="store_true", help="skip the live attitude/IMU section")
    ap.add_argument("--seconds", type=float, default=10.0, help="how long to stream live data")
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for the first heartbeat")
    ap.add_argument("--listen-only", action="store_true",
                    help="send nothing at all: heartbeat + whatever the board streams by itself")
    a = ap.parse_args()

    is_serial = a.port.startswith("/dev/")
    if is_serial:
        serial_preflight(a.port)
    try:
        link = Link(a.port, a.baud, a.listen_only)
    except Exception as e:                          # serial.SerialException, OSError (busy port, etc.)
        print(f"could not open {a.port}: {e}")
        print("  Is another program holding it (QGroundControl, MAVProxy, ModemManager)?")
        sys.exit(2)

    print(f"waiting up to {a.timeout:g} s for an autopilot heartbeat on {a.port}"
          + (f" @ {a.baud}" if is_serial else "") + " ...", flush=True)
    hb = wait_autopilot_heartbeat(link, a.timeout)
    if hb is None:
        print("no autopilot heartbeat.")
        if is_serial:
            print("  Bytes seen: " + ("yes (wrong baud or not MAVLink?)" if link.c.mav.total_bytes_received else "none")
                  + ". On TELEM: check baud matches SER_TEL1_BAUD, TX->RX / RX->TX, common GND,"
                    " and that MAV_0_CONFIG points at that port. On USB: some boards also expose a second"
                    " ACM port; try the other /dev/ttyACM*.")
        sys.exit(1)

    try:
        print_heartbeat(link, hb)
        if link.tx_ok:
            print_version(link)
        if link.tx_ok:
            print_params(link)
        if not link.tx_ok and not a.listen_only:
            print("\nWRITES TO THE BOARD FAIL: it sends MAVLink but does not accept data from this PC, so firmware"
                  " version and params can't be requested. Continuing listen-only. On a Cube Black (FMUv3) with"
                  " stock PX4 1.16/1.17 and an SD card in, this is the board running out of RAM: boot without the"
                  " card or flash the build from setup/build_px4_fmu_v3_trimmed.sh. Otherwise unplug/replug USB"
                  " and run again; if it persists, try TELEM1 or another cable/port.")
        if not a.no_stream:
            stream(link, a.seconds)
    except OSError as e:                            # includes serial.SerialException (an IOError subclass)
        print(f"\nlink lost: {e}\n  Cable pulled or bumped, board rebooted, or another program opened the port.")
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    link.close()


if __name__ == "__main__":
    main()
