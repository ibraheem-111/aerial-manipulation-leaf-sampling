"""One-command PX4 + ROS 2 experiment: start MuJoCo + PX4 SITL, the DDS agent and the MPC node, wait for the
node to finish, shut everything down. Run inside `source setup/env_ros.sh`.

    python scripts/px4_experiment.py --task excite --seed 0              # rate-loop identification run
    python scripts/px4_experiment.py --task push --tag ideal             # touch and push
    python scripts/px4_experiment.py --task push --realistic --seed 1 --tag realistic_s1
    python scripts/px4_experiment.py --task reach --viewer               # watch it

Extra node options (e.g. --tau-rate 0.04) are passed through after `--`.
Logs: runs/uam_quad/px4/<task>_<tag|seed>_{sim,agent,node}.log
"""
import argparse
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from uam_quad.paths import RUNS  # noqa: E402

OUT = os.path.join(RUNS, "px4")


def wait_for(path, text, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(path) and text in open(path, errors="ignore").read():
            return True
        time.sleep(0.5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["reach", "excite", "push"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--realistic", action="store_true")
    ap.add_argument("--viewer", action="store_true")
    a, node_extra = ap.parse_known_args()
    node_extra = [x for x in node_extra if x != "--"]
    tag = a.tag or f"seed{a.seed}"
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, f"{a.task}_{tag}")
    px4_log = os.path.join(RUNS, "px4_sitl.log")
    if os.path.exists(px4_log):
        os.remove(px4_log)

    sim_cmd = [sys.executable, os.path.join(HERE, "px4_sim.py"), "--ros", "--seed", str(a.seed),
               "--scene", "push" if a.task == "push" else "empty"]
    if a.realistic:
        sim_cmd.append("--realistic")
    if a.viewer:
        sim_cmd.append("--viewer")
    procs = []
    try:
        procs.append(subprocess.Popen(sim_cmd, stdout=open(stem + "_sim.log", "w"), stderr=subprocess.STDOUT,
                                      cwd=HERE, start_new_session=True))
        if not wait_for(px4_log, "Ready for takeoff", 90):
            sys.exit(f"PX4 not ready after 90 s; see {px4_log} and {stem}_sim.log")
        procs.append(subprocess.Popen(["MicroXRCEAgent", "udp4", "-p", "8888"], stdout=open(stem + "_agent.log", "w"),
                                      stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(4.0)
        node_cmd = [sys.executable, os.path.join(HERE, "mpc_px4_node.py"), "--task", a.task, "--seed", str(a.seed),
                    "--tag", tag, *node_extra]
        with open(stem + "_node.log", "w") as log:
            rc = subprocess.call(node_cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE, timeout=240)
        time.sleep(3.0)                                    # let it land
        print(f"node exit {rc}; logs {stem}_*.log", flush=True)
        for line in open(stem + "_node.log", errors="ignore"):
            if "done, landing" in line:
                print(line.strip(), flush=True)
    finally:
        for p in reversed(procs):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)
        time.sleep(2.0)
        for p in procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
