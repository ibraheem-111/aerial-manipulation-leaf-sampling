"""Props-off motor test for PX4 over MAVLink: spin ONE motor at a time at low throttle for a short time.

    python scripts/motor_test.py --motor 1                   # motor 1, 7 % for 2 s
    python scripts/motor_test.py --motor 3 --throttle 5 --seconds 1.5
    python scripts/motor_test.py --sequence                  # motors 1,2,3,4 one after another

PROPS MUST BE OFF. You are asked to type "props off" before anything is sent.

Uses MAV_CMD_ACTUATOR_TEST (what QGroundControl's Actuators tab sends): output function Motor N, value = throttle
fraction, with a PX4-side timeout so the motor stops even if this script dies. An explicit stop (value NaN) is
sent after each spin, on Ctrl-C and on any error. Hard limits: throttle <= 15 %, duration <= 3 s, vehicle must be
disarmed. While a motor runs, SERVO_OUTPUT_RAW is printed so you can see which PWM output actually moved.

PX4 quad-X motor numbering (from the PX4 airframe reference; check against the Actuators tab picture):
    1 front-right CCW, 2 rear-left CCW, 3 front-left CW, 4 rear-right CW   (viewed from above)
"""
import argparse
import math
import sys
import time

from pymavlink import mavutil

M = mavutil.mavlink
MAX_THROTTLE_PCT = 15.0
MAX_SECONDS = 3.0
# param5 of MAV_CMD_ACTUATOR_TEST uses MAVLink's ACTUATOR_OUTPUT_FUNCTION numbering (Motor1 = 1), NOT PX4's
# internal PWM_MAIN_FUNCn value (Motor1 = 101). PX4 commander does not ACK an UNSUPPORTED result, so a wrong
# number shows up as silence (checked in PX4 v1.16.2 Commander.cpp, handleCommandActuatorTest).
MOTOR1_FUNCTION = M.ACTUATOR_OUTPUT_FUNCTION_MOTOR1


def ack_for(c, command, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        m = c.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.2)
        if m and m.command == command:
            return m.result
    return None


def actuator_test(c, motor, value, timeout_s):
    c.mav.command_long_send(c.target_system, c.target_component, M.MAV_CMD_ACTUATOR_TEST, 0,
                            value, timeout_s, 0, 0, MOTOR1_FUNCTION + motor - 1, 0, 0)
    return ack_for(c, M.MAV_CMD_ACTUATOR_TEST)


def stop(c, motor):
    for _ in range(3):                                    # NaN = stop / back to disarmed output
        if actuator_test(c, motor, math.nan, 0) == M.MAV_RESULT_ACCEPTED:
            return True
    return False


def spin(c, motor, pct, seconds, n_motors=4):
    print(f"\n>>> motor {motor}: {pct:g} % for {seconds:g} s")
    res = actuator_test(c, motor, pct / 100.0, seconds)
    if res != M.MAV_RESULT_ACCEPTED:
        name = M.enums["MAV_RESULT"][res].name if res is not None else "no COMMAND_ACK"
        print(f"    rejected: {name}")
        return False
    end = time.monotonic() + seconds
    last_print = 0.0
    while time.monotonic() < end:
        m = c.recv_match(type="SERVO_OUTPUT_RAW", blocking=True, timeout=0.2)
        if m and time.monotonic() - last_print > 0.25:
            pwm = [getattr(m, f"servo{i}_raw") for i in range(1, n_motors + 1)]
            print("    PWM out " + "  ".join(f"{i + 1}:{p}" for i, p in enumerate(pwm)), flush=True)
            last_print = time.monotonic()
    print(f"    stop sent: {'ok' if stop(c, motor) else 'NOT ACKED - unplug the battery'}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--motor", type=int, choices=[1, 2, 3, 4])
    g.add_argument("--sequence", action="store_true", help="motors 1..4 one after another")
    ap.add_argument("--throttle", type=float, default=7.0, help=f"percent, max {MAX_THROTTLE_PCT:g}")
    ap.add_argument("--seconds", type=float, default=2.0, help=f"per motor, max {MAX_SECONDS:g}")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=57600)
    ap.add_argument("--yes-props-off", action="store_true", help="skip the typed confirmation (scripted use)")
    a = ap.parse_args()
    if not 0 < a.throttle <= MAX_THROTTLE_PCT:
        sys.exit(f"--throttle must be in (0, {MAX_THROTTLE_PCT:g}]")
    if not 0 < a.seconds <= MAX_SECONDS:
        sys.exit(f"--seconds must be in (0, {MAX_SECONDS:g}]")

    c = mavutil.mavlink_connection(a.port, baud=a.baud, source_system=255,
                                   source_component=M.MAV_COMP_ID_MISSIONPLANNER, autoreconnect=False)
    if a.port.startswith("/dev/"):
        c.port.write_timeout = 1.0
    hb = None
    end = time.monotonic() + 10
    while time.monotonic() < end and hb is None:
        m = c.recv_match(type="HEARTBEAT", blocking=True, timeout=0.5)
        if m and m.autopilot != M.MAV_AUTOPILOT_INVALID and m.type != M.MAV_TYPE_GCS:
            hb = m
    if hb is None:
        sys.exit("no autopilot heartbeat")
    c.target_system, c.target_component = hb.get_srcSystem(), hb.get_srcComponent()
    if hb.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED:
        sys.exit("vehicle is ARMED - refusing. Disarm first.")

    ss = c.recv_match(type="SYS_STATUS", blocking=True, timeout=3)
    if ss and ss.voltage_battery not in (0, 65535):
        print(f"battery {ss.voltage_battery / 1000:.2f} V")
    else:
        print("WARNING: no battery voltage reported. Either no battery (motors cannot spin; you will only see the"
              " PWM change) or the power module sense lead is not connected.")

    motors = [1, 2, 3, 4] if a.sequence else [a.motor]
    if not a.yes_props_off:
        ans = input(f"Spin motor(s) {motors} at {a.throttle:g} % for {a.seconds:g} s each. "
                    "PROPELLERS REMOVED? Type 'props off' to continue: ")
        if ans.strip().lower() != "props off":
            sys.exit("not confirmed - nothing sent")

    current = motors[0]
    try:
        for current in motors:
            if not spin(c, current, a.throttle, a.seconds):
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nCtrl-C: stopping")
    finally:
        for mtr in motors:
            stop(c, mtr)
        if getattr(c, "portdead", False):
            print("!! writes to the board failed during the test - unplug the battery to be sure motors are off")
        c.close()


if __name__ == "__main__":
    main()
