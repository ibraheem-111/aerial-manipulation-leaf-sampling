"""Parallel tuning of the PID baseline and wind-strength sweep for all controllers.

    python scripts/sweep_wind.py --tune      # PID gain grid at the default wind, 2 seeds -> picks PID_TUNED
    python scripts/sweep_wind.py             # wind sweep: pid, pid_tuned, lqr_sched, mpc x wind levels x 2 seeds
"""
import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_controllers as cc  # noqa: E402

OUT = cc.OUT
KEYS = ("crashed", "ee_rms_arm_phase_mm", "ee_rms_arm_phase_in_plane_mm", "ee_rms_arm_phase_sideways_mm",
        "ee_mean_hold_mm", "base_dev_rms_arm_phase_mm", "max_tilt_deg", "rotor_at_limit_pct")


def job(args):
    ctrl, wind, seed, gains = args
    try:
        r, _, _ = cc.run(ctrl, True, seed=seed, wind_mag=wind, pid_gains=gains)
    except Exception as e:  # keep the sweep going; record the failure
        r = {"crashed": True, "error": repr(e)}
    r.update(controller=ctrl, wind_mean_N=wind, seed=seed, gains=gains)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.tune:
        grid = [dict(pos_wn=wn, pos_ki=ki, i_limit=1.0) for wn, ki in itertools.product((2.0, 2.5, 3.0), (0.6, 0.8, 1.2))]
        jobs = [(c, 0.335, s, g) for g in grid for c in ("pid_tuned", "pid_tuned_noouter") for s in (0, 1)]
    else:
        levels = (0.335, 0.75, 1.25, 1.75, 2.5, 3.5)
        jobs = [(c, w, s, None) for w in levels for c in ("pid", "pid_tuned_noouter", "lqr_sched", "mpc") for s in (0, 1)]
    with Pool(a.workers) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            res.append(r)
            print(f"[{i + 1}/{len(jobs)}]", json.dumps({k: r.get(k) for k in ("controller", "wind_mean_N", "seed", "gains") + KEYS}), flush=True)
    name = "tune.json" if a.tune else "wind_sweep.json"
    json.dump(res, open(f"{OUT}/{name}", "w"), indent=1)
    print("wrote", f"{OUT}/{name}")


if __name__ == "__main__":
    main()
