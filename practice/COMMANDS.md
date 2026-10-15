# MuJoCo + UAM cheat sheet

Everything here runs from a terminal on the desktop, from the repo root. Once per terminal:

```bash
source setup/env.sh          # Python env (MuJoCo 3.3.2, CasADi, ACADOS); first time: bash setup/install_sim.sh
cd src/uam_quad              # our package
```

---

## 1. Look at a model (MuJoCo's own viewer)

```bash
python scripts/export_xml.py                                   # writes models/*.xml from the Python generator
python -m mujoco.viewer --mjcf=models/f450_omx.xml
python -m mujoco.viewer --mjcf=models/f450_omx_push_scene.xml
```

No controller runs here, so with motors at zero the drone falls. That's expected. Use it to look, pause and poke.

**Viewer controls** (press **F1** in the viewer for the full list):

| Action | How |
|---|---|
| Rotate / pan / zoom the camera | Left-drag / right-drag / scroll |
| Pause and resume; step while paused | Space; → |
| Reset the simulation | Backspace |
| Select a body | Double-click it |
| Push the selected body (force) | Ctrl + right-drag |
| Twist the selected body (torque) | Ctrl + left-drag |
| Show / hide the left and right panels | Tab / Shift+Tab |
| Contact points / contact forces | C / F (also in the left panel, under Rendering) |

The left panel's **Control** section has a slider per actuator: `thrust0–3` in newtons, `servo1–3` in radians. Drag the thrust sliders up to about 4 N each and the drone lifts off.

---

## 2. Fly it live (our controller in the loop)

```bash
python scripts/fly.py                                # hover, arm swinging (geometric controller)
python scripts/fly.py --arm still                    # arm held still
python scripts/fly.py --arm reach --waypoints        # fly a 1 m square while the arm moves
python scripts/fly.py --no-com-ff                    # turn off centre-of-mass feedforward: watch the drift
python scripts/fly.py --amp 1.0 --freq 1.0           # aggressive arm swing
python scripts/fly.py --headless --seconds 10        # no window, prints max tilt / hover error
```

In the window, double-click the drone and Ctrl + right-drag to push it, then watch the controller recover. (`fly.py` drives the physics itself, so Space may not pause it. Use the standalone viewer in section 1 for pausing.)

---

## 3. The experiments (each writes a plot, JSON and optionally a video)

| What | Command | Output in `runs/uam_quad/` (repo root) | Time |
|---|---|---|---|
| Arm swing vs geometric controller | `python scripts/arm_swing_test.py --preset f450_mrs --amp 1.0 --freq 1.0` | `arm_swing_*.png` | 20 s |
| MPC reach task | `python scripts/mpc_task.py --video` | `mpc_task_f450_stock_omx.{png,json,mp4}` | 1–2 min |
| MPC with a sideways push | `python scripts/mpc_task.py --push 4` | `…_push4.*` | 1 min |
| Ablation: no arm coupling | `python scripts/mpc_task.py --fast --no-coupling` | `…_nocpl_fast.*` (it flips) | 1 min |
| MPC vs PID, ideal/realistic | `python scripts/compare_controllers.py --video` | `compare/compare.png`, `compare_realistic.mp4` | 3–4 min |
| Four-way video | `python scripts/video_4way.py` | `compare/compare_4way_realistic.mp4` | 4 min |
| Wind sweep, all controllers | `python scripts/sweep_wind.py` | `compare/wind_sweep*.json` | 15–20 min |
| Touch and push | `python scripts/push_task.py --video` | `push/push.png`, `push/push_realistic.mp4` | 3 min |
| Unit tests | `python -m pytest -q tests` | 15 pass | 2 s |

Watch a video: `xdg-open ../../runs/uam_quad/mpc_task_f450_stock_omx.mp4`

---

## 4. Poke at it from Python (`python` or `ipython`)

```python
import mujoco, numpy as np
from uam_quad import f450_stock, with_omx_arm, load_model, hover_trim, GeometricController

model, cfg = load_model(with_omx_arm(f450_stock()))   # MjModel = the fixed description
data = mujoco.MjData(model)                            # MjData  = the changing state

model.nq, model.nv, model.nu                           # 10, 9, 7  (quaternion -> nq > nv)
[model.actuator(i).name for i in range(model.nu)]      # thrust0..3, servo1..3
model.body_subtreemass[model.body("base").id]          # total mass, 1.617 kg

trim = hover_trim(model, cfg, [-0.6, 1.0, 0.4])        # thrusts + servo setpoints that hold still
trim.ctrl[:4]                                          # front 5.1 N, rear 2.8 N

data.qpos[:] = trim.qpos; data.ctrl[:] = trim.ctrl; data.act[:4] = trim.act
mujoco.mj_forward(model, data)                         # compute everything for this state (no time step)
data.site_xpos[model.site("ee").id]                    # end-effector position
data.sensor("gyro").data                               # body rates from the IMU site

ctrl = GeometricController(model, cfg)
for _ in range(1000):                                  # 2 s at 500 Hz
    data.ctrl[:4], _ = ctrl(data, np.array([0.2, 0, 1.0]))
    mujoco.mj_step(model, data)                        # advance time by model.opt.timestep (2 ms)
data.qpos[0:3]                                         # ~0.26 m: heading to x = 0.2, still overshooting at 2 s

data.xfrc_applied[model.body("base").id, :3] = [0, 2, 0]   # external force on the base (wind / push)
```

The vocabulary Das will use:

| Term | Meaning |
|---|---|
| `MjModel` | The fixed description (masses, geometry, actuators) |
| `MjData` | The changing state (`qpos`, `qvel`, `ctrl`, `act`, contacts, sensors) |
| `qpos` / `qvel` | Generalized positions / velocities; free joint = position + quaternion |
| `ctrl` | Actuator commands; here rotor thrust [N] and servo targets [rad] |
| `act` | Actuator internal state; here the lagged rotor thrust |
| `mj_step` | Advance one timestep |
| `mj_forward` | Recompute derived quantities without advancing time |
| `xfrc_applied` | Apply external forces and torques to a body |
| `mjd_transitionFD` | Finite-difference linearization (`uam_quad.linearize`) |
| contacts | `data.ncon`, `data.contact[i]`, and `mujoco.mj_contactForce` for the force |

---

## 5. A 5-minute live demo for a call

1. `python -m mujoco.viewer --mjcf=models/f450_omx.xml`
   - Show the body tree in the right panel and the actuators in the left panel.
   - Drag thrust sliders to about 4 N to show lift; pause with Space.
2. `python scripts/fly.py`
   - Hover with the arm swinging. Push the drone with Ctrl + right-drag and watch it recover.
   - Then rerun with `--no-com-ff` to show what the feedforward does.
3. `xdg-open ../../runs/uam_quad/compare/compare_4way_realistic.mp4`
   - PID / LQR / MPC side by side under noise and wind.
4. `xdg-open ../../runs/uam_quad/push/push_realistic.mp4`
   - Touch and push: the drone leans into the board.
5. If asked about the model: `python -m pytest -q tests -k eom`
   - The hand-derived equations match MuJoCo.

## 6. If something breaks

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: uam_quad` or `acados` | You didn't `source setup/env.sh` |
| Viewer or `Renderer` error about the display | Run from a desktop terminal, or `export DISPLAY=:1` |
| MPC script slow on first run | ACADOS compiles C code once (about 10 s); later runs reuse it |
| Plot or video not where expected | Everything goes to `runs/uam_quad/` at the repo root |
