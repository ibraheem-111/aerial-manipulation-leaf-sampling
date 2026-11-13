"""Compare PX4 touch-and-push runs by tag prefix: ideal seed 0 and realistic seeds 0-3 per variant.

    python scripts/compare_push.py base cc cy cyr truth      # reads runs/uam_quad/px4/push_px4_<prefix>_*.json
"""
import json
import os
import sys

import numpy as np

from uam_quad.paths import RUNS

D = os.path.join(RUNS, "px4")
KEYS = ("force_mean_2N", "force_mean_4N", "force_rms_err_4N", "contact_lost_pct", "max_tilt_deg", "motor_sat_pct")


def load(tag):
    p = os.path.join(D, f"push_px4_{tag}.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))["result"]
    return {k: r.get(k, float("nan")) for k in KEYS}


def mean(v):
    v = [x for x in v if np.isfinite(x)]          # motor_sat_pct is missing in runs logged before it existed
    return float(np.mean(v)) if v else float("nan")


def main():
    print(f"{'run':24s}" + "".join(f"{k:>18s}" for k in KEYS))
    for pre in sys.argv[1:] or ["base", "cc", "cy"]:
        rows = {n: load(f"{pre}_{n}") for n in ["ideal_s0"] + [f"realistic_s{s}" for s in range(4)]}
        for n, r in rows.items():
            if r:
                print(f"{pre + '_' + n:24s}" + "".join(f"{r[k]:18.2f}" for k in KEYS))
        R = [r for n, r in rows.items() if r and n.startswith("realistic")]
        if R:
            print(f"{pre + ' realistic mean':24s}" + "".join(f"{mean([r[k] for r in R]):18.2f}" for k in KEYS))
        print()


if __name__ == "__main__":
    main()
