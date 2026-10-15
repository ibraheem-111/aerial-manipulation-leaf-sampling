"""Grades exercises.py.  Usage:
    source setup/env.sh && cd practice        # from the repo root
    python check.py               # grade your answers
    python check.py --solutions   # grade the reference solutions (should be 12/12)
    python check.py 3 7           # grade only exercises 3 and 7
"""
import importlib
import math
import os
import sys
import tempfile
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
G = 9.81


def checks(mod):
    def c1():
        m = mod.ex01_load_pendulum()
        assert m.nq == 1 and m.nbody == 2, f"expected 1 joint, 2 bodies; got nq={m.nq}, nbody={m.nbody}"

    def c2():
        nq, nv, nu, nb = mod.ex02_sizes()
        assert (nq, nv, nu, nb) == (10, 9, 7, 6), f"got {(nq, nv, nu, nb)}"

    def c3():
        T = mod.ex03_pendulum_period()
        expect = 2 * math.pi * math.sqrt(0.5 / G)
        assert abs(T - expect) / expect < 0.03, f"got {T:.3f} s, expected about {expect:.3f} s"

    def c4():
        m = mod.ex04_total_mass()
        assert abs(m - 1.617) < 0.01, f"got {m:.3f} kg"

    def c5():
        names = list(mod.ex05_actuator_names())
        assert names == ["thrust0", "thrust1", "thrust2", "thrust3", "servo1", "servo2", "servo3"], f"got {names}"

    def c6():
        f = np.asarray(mod.ex06_hover_thrusts())
        assert f.shape == (4,), "return the 4 rotor thrusts"
        assert abs(f.sum() - 1.617 * G) < 0.05, f"thrusts should add up to m g = {1.617 * G:.2f} N, got {f.sum():.2f}"
        assert f[0] > f[1] and f[2] > f[3], "front rotors (0 and 2) should carry more than rear (1 and 3)"

    def c7():
        import mujoco
        v = np.asarray(mod.ex07_read_sensor())
        m = mujoco.MjModel.from_xml_path(mod.F450_XML)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        assert np.allclose(v, d.site_xpos[m.site("ee").id], atol=1e-9), f"got {v}"

    def c8():
        h = mod.ex08_free_fall()
        assert abs(h - 0.5 * G * 0.25) < 0.01, f"got {h:.3f} m, expected {0.5 * G * 0.25:.3f} m"

    def c9():
        a, b = mod.ex09_arm_pose_effect()
        assert a > b + 0.5, f"arm out should need more front thrust than folded: {a:.2f} vs {b:.2f} N"

    def c10():
        e = mod.ex10_hover_controller()
        assert e < 0.01, f"final error {e * 100:.1f} cm (should be < 1 cm)"

    def c11():
        F = mod.ex11_contact_force()
        assert abs(F - 1.0 * G) / G < 0.02, f"got {F:.2f} N, expected about {G:.2f} N"

    def c12():
        p = os.path.join(tempfile.gettempdir(), "uam_practice_render.png")
        if os.path.exists(p):
            os.remove(p)
        mod.ex12_render_png(p)
        assert os.path.exists(p) and os.path.getsize(p) > 10_000, "no PNG written (or it is empty)"

    return [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mod = importlib.import_module("solutions" if "--solutions" in sys.argv else "exercises")
    which = {int(a) for a in args} or set(range(1, 13))
    passed = 0
    for i, c in enumerate(checks(mod), 1):
        if i not in which:
            continue
        try:
            c()
            print(f"ex{i:02d}  PASS")
            passed += 1
        except NotImplementedError:
            print(f"ex{i:02d}  not done yet")
        except AssertionError as e:
            print(f"ex{i:02d}  FAIL  {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ex{i:02d}  ERROR {type(e).__name__}: {e}")
            if "-v" in sys.argv:
                traceback.print_exc()
    print(f"\n{passed}/{len(which)} passed")


if __name__ == "__main__":
    main()
