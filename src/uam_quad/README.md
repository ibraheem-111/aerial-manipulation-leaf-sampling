# uam_quad

The Python package for the project: a MuJoCo model of an F450 quadrotor with a 3-DoF pitch arm on top,
its equations of motion, and the controllers (geometric PID cascade, LQR/LQI, whole-body MPC,
admittance force control, Kalman disturbance estimator).

**Start at the [repository README](../../README.md)**: install, how to run everything, notation,
equations for each controller, and a map of where everything lives in the code.

Deeper notes in `docs/`:
- [`docs/eom.md`](docs/eom.md): coupled Newton–Euler equations of motion, checked against MuJoCo
- [`docs/control_fundamentals.md`](docs/control_fundamentals.md): PID, LQR, MPC, force control background
- [`docs/wbmpc.md`](docs/wbmpc.md): whole-body MPC design, tuning lessons and all experiment results
