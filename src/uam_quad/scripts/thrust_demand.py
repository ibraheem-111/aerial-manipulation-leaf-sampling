"""Per-rotor thrust demand in the hardest standalone runs, for motor sizing.

Runs the realistic arm-tracking task (compare_controllers.py, MPC, wind 0.335 N and 1.25 N, seeds 0-1) and
the realistic touch-and-push (push_task.py, seeds 0-3), once with the real E310 ceiling (7.85 N) and once with
the ceiling raised to --cap (default 15 N) in both the plant and the MPC, so the demand is not hidden by
clipping. Thrust is MuJoCo's lagged rotor thrust, logged at 50 Hz.

    python scripts/thrust_demand.py            # -> runs/uam_quad/thrust_demand.json
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from uam_quad.paths import RUNS

JOBS = [("track", s, 0.335) for s in (0, 1)] + [("track", s, 1.25) for s in (0, 1)] + [("push", s, 0.335) for s in range(4)]


def job(a):
    kind, seed, wind, cap = a
    import uam_quad
    orig = uam_quad.f450_stock
    stock = (lambda **kw: orig(**{"thrust_max": cap, **kw})) if cap else orig
    if kind == "track":
        import compare_controllers as cc
        cc.f450_stock = stock
        r, L, _ = cc.run("mpc", True, seed=seed, wind_mag=wind)
    else:
        import push_task as pt
        pt.f450_stock = stock
        r, L, _ = pt.run(True, True, seed=seed)
    f = np.asarray(L["f"])
    return dict(kind=kind, seed=seed, wind=wind, cap=cap or 7.848, crashed=bool(r.get("crashed", False)),
                max_per_rotor=np.round(f.max(0), 2).tolist(), max_any=round(float(f.max()), 2),
                p999=round(float(np.percentile(f, 99.9)), 2), p99=round(float(np.percentile(f, 99)), 2),
                time_above_7p85_pct=round(float(np.mean(f > 7.84) * 100), 2),
                **{k: r[k] for k in ("force_mean_4N", "contact_lost_pct", "ee_rms_arm_phase_mm", "max_tilt_deg") if k in r})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=15.0)
    a = ap.parse_args()
    jobs = [(*j, None) for j in JOBS] + [(*j, a.cap) for j in JOBS]
    with Pool(min(len(jobs), os.cpu_count() or 1)) as p:
        rows = p.map(job, jobs)
    json.dump(rows, open(os.path.join(RUNS, "thrust_demand.json"), "w"), indent=1)
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
