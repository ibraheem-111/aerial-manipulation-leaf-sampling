"""Plot a PX4 run of mpc_px4_node.py: end-effector and hub tracking, force, admittance offset, MPC commands.

    python scripts/plot_px4_run.py push_px4_cy_ideal_s0     # a push run, by tag
    python scripts/plot_px4_run.py mpc_px4_reach            # the latest reach (overwritten every reach run)
    python scripts/plot_px4_run.py push_px4_cy_ideal_s0 --save    # -> runs/uam_quad/px4/<name>.png instead of a window

State x = [p, q (wxyz), v, w, th, thd (, yaw Pade state)], input u = [T, w_cmd (3, FLU), th_cmd (3)], from the
MPC's point of view (PX4's estimate unless --truth-state). ee_truth is MuJoCo's true end-effector position,
shifted into PX4's local frame by its offset at the first sample (as the node's ee_rms_mm_truth).
"""
import argparse
import warnings
import json
import os

warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from uam_quad.paths import RUNS

D = os.path.join(RUNS, "px4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="file name in runs/uam_quad/px4 without .json")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    d = json.load(open(os.path.join(D, a.name.removesuffix(".json") + ".json")))
    L = d["log"]
    t = np.array([r["t"] for r in L])
    x, u = np.array([r["x"] for r in L]), np.array([r["u"] for r in L])
    ee, ee_ref = np.array([r["ee_truth"] for r in L]), np.array([r["ee_ref"] for r in L])
    base_ref = np.array([r["base_ref"] for r in L])
    ee = ee - (ee[0] - ee_ref[0])      # MuJoCo world -> PX4 local frame: remove the start offset, as the node does
    push = any(r["F_des"] > 0 for r in L)

    fig, ax = plt.subplots(3 if push else 2, 2, figsize=(13, 9 if push else 6.5), sharex=True)
    ax = np.atleast_2d(ax)
    for i, c in enumerate("xyz"):
        ax[0, 0].plot(t, ee[:, i], f"C{i}", label=f"ee {c}")
        ax[0, 0].plot(t, ee_ref[:, i], f"C{i}--", lw=1)
        ax[0, 1].plot(t, x[:, i], f"C{i}", label=f"hub {c}")
        ax[0, 1].plot(t, base_ref[:, i], f"C{i}--", lw=1)
    ax[0, 0].set(ylabel="end effector [m]", title="end effector: true (solid) vs reference (dashed)")
    ax[0, 1].set(ylabel="hub [m]", title="hub: MPC's estimate (solid) vs reference (dashed)")
    err = np.linalg.norm(ee - ee_ref, axis=1) * 1e3
    ax[1, 0].plot(t, err, "k")
    ax[1, 0].set(ylabel="|ee error| [mm]", title=f"end-effector error, RMS after 6 s = "
                 f"{np.sqrt(np.mean(err[t > t[0] + 6] ** 2)):.1f} mm")
    ax[1, 1].plot(t, u[:, 1:4])
    ax2 = ax[1, 1].twinx()
    ax2.plot(t, u[:, 0], "k", lw=1)
    ax2.set_ylabel("T [N] (black)")
    ax[1, 1].legend(["ω_x cmd", "ω_y cmd", "ω_z cmd"], loc="upper left", fontsize=8)
    ax[1, 1].set(ylabel="body-rate cmd [rad/s]", title="MPC output to PX4")
    if push:
        ax[2, 0].plot(t, [r["F_filt"] for r in L], "C0", label="measured (filtered)")
        ax[2, 0].plot(t, [r["F_des"] for r in L], "k--", label="setpoint")
        ax[2, 0].fill_between(t, 0, 5, where=[r["kappa"] for r in L], color="C2", alpha=0.1, label="in contact")
        ax[2, 0].set(ylabel="normal force [N]", xlabel="t [s]", title="contact force")
        ax[2, 1].plot(t, np.array([r["delta"] for r in L]) * 1e3, "C3")
        ax[2, 1].set(ylabel="δ [mm]", xlabel="t [s]", title="admittance offset along n̂ (clamp −10 … 50 mm)")
    for a_ in ax.flat:
        a_.grid(alpha=0.3)
        if a_.get_legend_handles_labels()[0]:
            a_.legend(fontsize=8)
    res = d.get("result", {})
    fig.suptitle(a.name + "   " + ", ".join(f"{k}={v}" for k, v in res.items()
                                             if k in ("force_mean_4N", "force_rms_err_4N", "motor_sat_pct", "solve_ms_median")),
                 fontsize=9)
    fig.tight_layout()
    if a.save:
        out = os.path.join(D, a.name + ".png")
        fig.savefig(out, dpi=110)
        print(out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
