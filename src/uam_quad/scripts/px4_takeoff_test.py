"""Smoke test for the PX4 + MuJoCo setup over MAVLink (no ROS 2 needed): arm, take off to 2.5 m,
hover, land. Run while scripts/px4_sim.py is running.

    python scripts/px4_takeoff_test.py
"""
import sys
import time

from pymavlink import mavutil


def main():
    c = mavutil.mavlink_connection("udpin:0.0.0.0:14540")
    print("waiting for PX4 heartbeat on udp 14540 ...", flush=True)
    c.wait_heartbeat(timeout=60)
    print(f"heartbeat from system {c.target_system}", flush=True)

    def cmd(command, *p):
        p = list(p) + [0] * (7 - len(p))
        c.mav.command_long_send(c.target_system, c.target_component, command, 0, *p)
        ack = c.recv_match(type="COMMAND_ACK", blocking=True, timeout=3)
        return ack.result if ack else None

    def local_z():
        """Latest altitude: drain the queue (PX4 streams faster than we read)."""
        last = None
        while (m := c.recv_match(type="LOCAL_POSITION_NED", blocking=False)) is not None:
            last = m
        if last is None:
            last = c.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=2)
        return None if last is None else -last.z

    # wait until the estimator is ready (PX4 refuses to arm before). Arm in Hold mode: Position mode
    # (the default) needs RC sticks, and there is no RC in simulation.
    t0 = time.time()
    while time.time() - t0 < 60:
        cmd(mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            4, 3)                                     # PX4 custom mode AUTO (4) / LOITER = Hold (3)
        r = cmd(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
        if r == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print("armed", flush=True)
            break
        time.sleep(1.0)
    else:
        sys.exit("could not arm within 60 s (see runs/uam_quad/px4_sitl.log)")

    # param 7 is altitude above sea level; NaN = PX4 default takeoff height (MIS_TAKEOFF_ALT, 2.5 m)
    r = cmd(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, float("nan"), float("nan"), float("nan"), float("nan"))
    print("takeoff command result", r, flush=True)
    t0 = time.time()
    while time.time() - t0 < 15:
        z = local_z()
        if z is not None:
            print(f"t={time.time() - t0:5.1f}s  altitude {z:5.2f} m", flush=True)
        time.sleep(1.0)
    r = cmd(mavutil.mavlink.MAV_CMD_NAV_LAND)
    print("land command result", r, flush=True)
    t0 = time.time()
    while time.time() - t0 < 15:
        z = local_z()
        if z is not None:
            print(f"t={time.time() - t0:5.1f}s  altitude {z:5.2f} m", flush=True)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
