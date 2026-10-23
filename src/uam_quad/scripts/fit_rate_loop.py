"""Identify PX4's closed rate loop from the excitation flights (mpc_px4_node.py --task excite).

Model per body axis i (FLU), with the rate command held between 50 Hz updates (zero-order hold):
    tau_i * d(omega_i)/dt = K_i * omega_cmd,i(t - theta_i) - omega_i
  full     : tau, K, theta fitted (theta on a 0-200 ms grid, tau and K by least squares at each grid point)
  mpc-form : K = 1, theta = 0, tau fitted   (the structure the MPC model uses: w_dot = (w_cmd - w) / tau)
Fit: least squares on the FREE-RUN simulation error (the model is simulated over the whole 8-26 s record from the
measured initial rate, never reset to the data). Training: seeds 0 and 1. Validation: seed 2 (held out).
Score: VAF = 1 - var(omega_meas - omega_sim) / var(omega_meas), and RMS error, on the free-run simulation.
The "measured" rate is MuJoCo's true body rate published by the bridge at 250 Hz (/sim/base_twist).

    python scripts/fit_rate_loop.py      # -> runs/uam_quad/px4/rate_id_fit.json, rate_id_fit.png
"""
import json
import os

import numpy as np
from scipy.optimize import least_squares

from uam_quad.paths import RUNS

DIR = os.path.join(RUNS, "px4")
T0, T1 = 8.0, 26.0
TRAIN, TEST = (0, 1), (2,)
AXES = ("roll", "pitch", "yaw")
THETA_GRID = np.arange(0.0, 0.2001, 0.002)


def load(seed):
    d = json.load(open(os.path.join(DIR, f"rate_id_seed{seed}.json")))
    C, Y = np.array(d["cmd"]), np.array(d["truth"])
    Y = Y[(Y[:, 0] >= T0) & (Y[:, 0] < T1)]
    return C, Y


def simulate(C, Y, ax, tau, K, theta):
    """Free-run simulation on the measurement time grid, exact first-order step between samples."""
    t, y = Y[:, 0], Y[:, 1 + ax]
    idx = np.searchsorted(C[:, 0], t - theta, side="right") - 1
    u = np.where(idx >= 0, C[np.clip(idx, 0, None), 1 + ax], 0.0)
    w = np.empty_like(y)
    w[0] = y[0]
    for k in range(1, len(t)):
        a = np.exp(-(t[k] - t[k - 1]) / tau)
        w[k] = a * w[k - 1] + (1 - a) * K * u[k - 1]
    return w


def residuals(params, data, ax, form):
    tau, K, theta = (params[0], 1.0, 0.0) if form == "mpc" else params
    return np.concatenate([simulate(C, Y, ax, tau, K, theta) - Y[:, 1 + ax] for C, Y in data])


def score(data, ax, tau, K, theta):
    e, y = [], []
    for C, Y in data:
        e.append(simulate(C, Y, ax, tau, K, theta) - Y[:, 1 + ax]); y.append(Y[:, 1 + ax])
    e, y = np.concatenate(e), np.concatenate(y)
    return dict(vaf_pct=round(float(100 * (1 - np.var(e) / np.var(y))), 1), rms_rad_s=round(float(np.sqrt(np.mean(e ** 2))), 3))


def main():
    train = [load(s) for s in TRAIN]
    test = [load(s) for s in TEST]
    out = dict(train_seeds=TRAIN, test_seeds=TEST, window_s=[T0, T1], axes={})
    for ax, name in enumerate(AXES):
        # The ZOH lookup makes the residual piecewise constant in theta (zero gradient), so theta is gridded
        # (2 ms steps, half the 250 Hz sample period) and (tau, K) are fitted by least squares at each theta.
        best = None
        for theta in THETA_GRID:
            r = least_squares(lambda p: residuals([p[0], p[1], theta], train, ax, "full"), x0=[0.05, 1.0],
                              bounds=([0.003, 0.3], [3.0, 3.0]))
            if best is None or r.cost < best[0]:
                best = (r.cost, r.x[0], r.x[1], theta)
        _, tau, K, theta = best
        mpcf = least_squares(residuals, x0=[0.05], args=(train, ax, "mpc"), bounds=([0.003], [3.0]))
        out["axes"][name] = dict(
            full=dict(tau_s=round(float(tau), 4), K=round(float(K), 3), theta_s=round(float(theta), 4),
                      train=score(train, ax, tau, K, theta), test=score(test, ax, tau, K, theta)),
            mpc_form=dict(tau_s=round(float(mpcf.x[0]), 4),
                          train=score(train, ax, mpcf.x[0], 1.0, 0.0), test=score(test, ax, mpcf.x[0], 1.0, 0.0)),
            placeholder_0p05=dict(test=score(test, ax, 0.05, 1.0, 0.0)))
        print(name, json.dumps(out["axes"][name]), flush=True)
    json.dump(out, open(os.path.join(DIR, "rate_id_fit.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C, Y = test[0]
    fig, axs = plt.subplots(3, 1, figsize=(9, 6.5), sharex=False)
    for ax, name in enumerate(AXES):
        a0 = T0 + 6 * ax
        sel = (Y[:, 0] >= a0 + 1.0) & (Y[:, 0] < a0 + 3.0)
        f, m = out["axes"][name]["full"], out["axes"][name]["mpc_form"]
        ws_full = simulate(C, Y, ax, f["tau_s"], f["K"], f["theta_s"])
        ws_mpc = simulate(C, Y, ax, m["tau_s"], 1.0, 0.0)
        cs = (C[:, 0] >= a0 + 1.0) & (C[:, 0] < a0 + 3.0)
        axs[ax].step(C[cs, 0], C[cs, 1 + ax], where="post", color="0.6", lw=1, label="command ω_cmd")
        axs[ax].plot(Y[sel, 0], Y[sel, 1 + ax], color="k", lw=1.4, label="measured ω (true body rate)")
        axs[ax].plot(Y[sel, 0], ws_full[sel], color="C1", lw=1.2, ls="--", label=f"fit τ={f['tau_s']*1e3:.0f} ms, θ={f['theta_s']*1e3:.0f} ms, K={f['K']:.2f}")
        axs[ax].plot(Y[sel, 0], ws_mpc[sel], color="C0", lw=1.0, ls=":", label=f"MPC form τ={m['tau_s']*1e3:.0f} ms")
        axs[ax].set_ylabel(f"{name} rate [rad/s]"); axs[ax].grid(alpha=0.3); axs[ax].legend(fontsize=7, loc="upper right")
    axs[2].set_xlabel("time since arming [s]   (held-out seed 2, free-run simulation)")
    fig.tight_layout(); fig.savefig(os.path.join(DIR, "rate_id_fit.png"), dpi=110)


if __name__ == "__main__":
    main()
