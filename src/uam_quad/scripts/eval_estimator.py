"""Multi-seed evaluation of the Kalman disturbance estimator (uam_quad/estimator.py).

  free flight : arm-driven reach task (compare_controllers.py) under realistic conditions,
                MPC + integrator on the gripper error ("mpc") vs MPC + Kalman estimator ("mpc_kf")
  contact     : touch and push (push_task.py), force-aware MPC, realistic, without the estimator and with it
                in three modes (the admittance loop always owns the board normal):
                  tangential : live estimate on y and z      vertical : live estimate on z only
                  freeze     : free-flight estimate held while touching

    python scripts/eval_estimator.py            # about 6 min on 12 cores -> runs/uam_quad/estimator/results.json
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uam_quad.paths import RUNS  # noqa: E402

SEEDS = (0, 1, 2, 3)


def job(a):
    kind, variant, seed = a
    if kind == "free":
        import compare_controllers as cc
        r, _, _ = cc.run(variant, True, seed=seed)
        keep = ("crashed", "ee_rms_arm_phase_mm", "base_dev_rms_arm_phase_mm", "max_tilt_deg", "d_hat_end_N")
    else:
        import push_task as pt
        r, _, _ = pt.run(True, True, seed=seed, estimator=(variant != "none"), est_axes=variant if variant != "none" else "tangential")
        keep = ("force_mean_2N", "force_mean_4N", "force_rms_err_4N", "contact_lost_pct", "max_tilt_deg", "d_hat_mean_N")
    return dict(kind=kind, variant=variant, seed=seed, **{k: r.get(k) for k in keep})


def main():
    jobs = [("free", v, s) for v in ("mpc", "mpc_kf") for s in SEEDS] + [("contact", v, s) for v in ("none", "tangential", "vertical", "freeze") for s in SEEDS]
    with Pool(min(len(jobs), os.cpu_count() or 1)) as p:
        rows = p.map(job, jobs)
    out = os.path.join(RUNS, "estimator")
    os.makedirs(out, exist_ok=True)
    json.dump(rows, open(os.path.join(out, "results.json"), "w"), indent=1)
    for kind, variant in (("free", "mpc"), ("free", "mpc_kf")) + tuple(("contact", v) for v in ("none", "tangential", "vertical", "freeze")):
        R = [r for r in rows if r["kind"] == kind and r["variant"] == variant]
        num = {k: v for k, v in R[0].items() if isinstance(v, (int, float)) and k != "seed" and not isinstance(v, bool)}
        print(f"{kind:8s} {variant:7s}", {k: round(float(np.mean([r[k] for r in R])), 2) for k in num},
              "crashes" if kind == "free" else "", sum(bool(r.get("crashed")) for r in R) if kind == "free" else "")


if __name__ == "__main__":
    main()
