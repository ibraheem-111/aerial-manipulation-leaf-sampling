"""PX4 SITL flying the MuJoCo F450 + arm (MuJoCo is the physics, PX4 is the flight controller).

Starts the MuJoCo bridge (TCP 4560) and PX4 SITL (v1.16, airframe none_iris with this F450's rotor
geometry passed as PX4_PARAM_* overrides). PX4's MAVLink and uXRCE-DDS (ROS 2) interfaces come up as
in any SITL run:
    MAVLink offboard API   udp 14540        (pymavlink, MAVSDK)
    QGroundControl         udp 14550
    ROS 2 (uXRCE-DDS)      udp 8888         (run: MicroXRCEAgent udp4 -p 8888)

    python scripts/px4_sim.py                 # headless, real time
    python scripts/px4_sim.py --viewer        # with the MuJoCo viewer
    python scripts/px4_sim.py --no-px4        # bridge only; start PX4 yourself
"""
import argparse
import os
import signal
import subprocess
import sys

import mujoco

from uam_quad import f450_stock, load_model, with_omx_arm
from uam_quad.paths import REPO, RUNS
from uam_quad.px4_bridge import MujocoPX4Bridge, px4_params, run

PX4_DIR = os.path.join(REPO, "external", "PX4-Autopilot")
TH_NOM = (-0.6, 1.0, 0.4)


def start_px4(params, log_path):
    rootfs = os.path.join(PX4_DIR, "build", "px4_sitl_default", "rootfs")
    exe = os.path.join(PX4_DIR, "build", "px4_sitl_default", "bin", "px4")
    if not os.path.exists(exe):
        sys.exit(f"PX4 not built: run  bash setup/build_px4.sh  ({exe} missing)")
    env = dict(os.environ, PX4_SIM_MODEL="none_iris")
    env.update({f"PX4_PARAM_{k}": str(v) for k, v in params.items()})
    log = open(log_path, "w")
    proc = subprocess.Popen([exe, "-d"], cwd=rootfs, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    print(f"[px4] started (pid {proc.pid}), log: {log_path}", flush=True)
    return proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--no-px4", action="store_true", help="only run the bridge")
    ap.add_argument("--realtime", type=float, default=1.0, help="sim speed vs wall clock; 0 = as fast as possible")
    ap.add_argument("--thr-mdl-fac", type=float, default=1.0)
    ap.add_argument("--no-noise", action="store_true")
    a = ap.parse_args()

    cfg = with_omx_arm(f450_stock())
    cfg.start_height = 0.2                       # resting on the landing legs
    model, cfg = load_model(cfg)
    data = mujoco.MjData(model)
    qa = [model.jnt_qposadr[model.joint(f"arm_j{i}").id] for i in (1, 2, 3)]
    data.qpos[qa] = TH_NOM
    mujoco.mj_forward(model, data)

    params = px4_params(cfg, a.thr_mdl_fac)
    os.makedirs(RUNS, exist_ok=True)
    px4 = None
    if not a.no_px4:
        # PX4 connects to the bridge's TCP server and retries until the bridge is listening
        px4 = start_px4(params, os.path.join(RUNS, "px4_sitl.log"))
    bridge = MujocoPX4Bridge(model, data, cfg, TH_NOM, thr_mdl_fac=a.thr_mdl_fac, noise=not a.no_noise)
    try:
        run(model, data, bridge, realtime=a.realtime, viewer=a.viewer)
    except KeyboardInterrupt:
        pass
    finally:
        if px4 is not None and px4.poll() is None:
            os.killpg(px4.pid, signal.SIGTERM)
            px4.wait(timeout=5)
            print("[px4] stopped", flush=True)


if __name__ == "__main__":
    main()
