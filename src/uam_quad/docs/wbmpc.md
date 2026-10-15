# Whole-body MPC for the F450 + 3-DoF arm

Code: `uam_quad/wbmpc.py` (model + OCP), `scripts/mpc_task.py` (task scene, closed loop, metrics, video).
Status (2026-10-15): flying in MuJoCo on the stock F450 with the ROBOTIS OpenMANIPULATOR-X arm.

## Platform used
| Part | Source | Values |
|---|---|---|
| Airframe | DJI F450 spec + DJI E310 propulsion spec; ESC/battery/avionics masses estimated | 1.104 kg airframe, 7.85 N max per rotor (800 g @ 12 V), PX4 quad-X order |
| Arm | ROBOTIS OpenMANIPULATOR-X URDF (Apache-2.0), joints 2–4 only | 0.415 kg moving, links 0.130 / 0.124 / 0.126 m, XM430-W350 (4.1 N·m @ 12 V) |
| Shoulder bracket | OMX link2, merged into the airframe inertia | 0.098 kg |
| Total | | 1.617 kg; hover with arm forward: 5.1 N front, 2.8 N rear |

## Prediction model
State (19): position, quaternion, world velocity, body rates, 3 joint angles, 3 joint rates.
Input (7): four rotor thrusts and three joint commands.

- **Base:** the coupled Newton–Euler equations from `docs/eom.md`, written symbolically in CasADi. They include the CoM offset c(θ), the inertia J_O(θ), and the arm-motion terms ḣ, ċ, c̈ and J̇.
- **Arm:** a critically damped reference model, θ̈ = ω_a²(θ_cmd − θ) − 2ω_a θ̇. In the simulator, stiff joint servos with gravity feedforward track this model's output. So the MPC knows θ̈, and therefore the arm's reaction on the base.
- **Check against MuJoCo:** the CasADi model matches MuJoCo's accelerations to 10⁻⁹ m/s² and 10⁻⁷ rad/s², and the end-effector position to machine precision (`tests/test_model.py::test_wbmpc_prediction_model_matches_mujoco`).

## Optimal control problem (ACADOS)
- **Horizon:** 25 × 40 ms = 1.0 s. RK4 (2 steps per interval), SQP-RTI, Gauss–Newton Hessian, HPIPM QP solver.
- **Rate:** runs at 50 Hz on the true MuJoCo state, with 500 Hz physics and 20 ms motor lag in the plant.
- **Cost:** nonlinear least squares on:
  - end-effector position (400): the objective
  - base-position anchor (2, or 200 when the base must hold still)
  - upright attitude (40, 40, 20)
  - base velocity (4) and body rates (1)
  - arm posture (0.02), joint rates (0.002)
  - rotor thrusts and joint commands near hover (0.02 each)
- **Constraints:** rotor thrust 0–7.85 N, joint limits, |q_x|, |q_y| ≤ 0.6.
- **Reference:** each solve receives a minimum-jerk end-effector trajectory, with base velocity feedforward, sampled across the horizon.

## Tuning lessons
1. **Smooth references, not step targets.** A step target produced bang-bang thrust (0 ↔ 7.85 N) and 33° of tilt. Minimum-jerk references brought this down to 5°.
2. **Keep arm regularization light.** Posture and joint-rate weights of 0.5 / 0.2 made moving the arm "expensive". The MPC then used the base or accepted centimeters of error: 25 mm RMS on a 1 Hz arm sweep. Dropping them to 0.02 / 0.002 gives 0.5–3.5 mm.
3. **Check reachability before blaming the controller.** The first sweep went 8 cm forward, past the OMX workspace with the base fixed (verified by inverse kinematics). The MPC correctly moved the base about 15 mm to cover it.

## Results (`runs/uam_quad/*.json`)
Reach task: hold, then end effector to target A on a post (1.2, 0, 1.35), then B (1.2, 0.45, 1.15), then home.

| Run | EE tracking while moving | Holding at A / B | Max tilt | Rotor thrust | Solve time (median / max) |
|---|---|---|---|---|---|
| Reach | 0.5 mm RMS (2.1 max) | 0.3 / 0.3 mm | 5.2° | 2.7–5.3 N | 2.5 / 3.5 ms |
| Reach, 4 N side push at 12 s | 82 mm max during push | 0.3 / 0.3 mm | 17.4° | 0–7.85 N (brief saturation) | 2.6 / 3.4 ms |
| Fast reach (1 m/s) | 3.5 mm RMS (13.8 max) | 1.8 / 2.0 mm | 14.5° | 2.2–5.9 N | 2.5 / 4.7 ms |

All 1,100 solves converged in every run above.

**Arm sweep:** the base holds still (anchor weight 200, ω_a = 25 rad/s) while the end effector traces a figure-eight in the arm plane (±5 cm in x, ±5 cm in z). Every point is reachable with the base fixed.

| Sweep frequency | With coupling terms | Without coupling terms (ablation) |
|---|---|---|
| 0.2 Hz | 0.5 mm RMS, base ±3 mm, 0.3° | **flips** (tilt > 10° at 3.9 s; joints driven to 5.1 rad/s vs 0.9 with coupling) |
| 0.7 Hz | 1.7 mm RMS, base ±3 mm, 2.7° | flips |
| 1.0 Hz | 3.5 mm RMS, base ±3 mm, 2.9° | flips |

The fast reach task without coupling also flips (154°).

With heavy arm damping (old weights) the ablation made no difference, because the arm never moved fast. The coupling terms are what let the MPC use the arm aggressively and stay stable.

## Run it
```bash
source setup/env.sh && cd src/uam_quad      # from the repo root
python scripts/mpc_task.py --video                       # reach task, plot + json + mp4 in runs/uam_quad (repo root)
python scripts/mpc_task.py --push 4                      # disturbance rejection
python scripts/mpc_task.py --fast [--no-coupling]        # fast moves / ablation
python scripts/mpc_task.py --task sweep --T 13 --arm-wn 25 --w-base 200 --sweep-f 0.7 --video
```

## Limitations and next steps
- **Perfect information:** the simulator gives exact state and the model matches the plant. Next: state-estimation noise, mass and inertia mismatch (±20%), and model error in the arm servos.
- **Thrust as the command:** rotor thrust is commanded directly. PX4 offboard wants thrust plus body rates, so a variant with those inputs is needed for hardware.
- **No contact:** the end effector reaches the pad but contact forces aren't modeled. Next: hold contact against a surface or compliant target (the project's open gap), and rotor downwash on the target.
- **Estimated parameters:** airframe masses beyond the frame and motors are estimates. Weigh the lab F450 and update `f450_stock(...)`.

## MPC vs cascaded PID, arm-driven task (`scripts/compare_controllers.py`, 2026-10-15)
The base parks in front of the post and the arm moves the gripper through points around the pad. The PID baseline is an autopilot-style cascade: position PID → geometric attitude PD at 500 Hz, with arm IK from the measured base pose at 50 Hz. The MPC runs at 50 Hz with integral action on the gripper error.

"Realistic" applies the same conditions to both controllers:
- **Sensor noise:** position 2 mm, attitude 0.3°, gyro 0.02 rad/s, joint encoders 0.002 rad.
- **Delay:** 10 ms on the state the controller sees.
- **Model mismatch:** plant 10% heavier, rotors give 95% of commanded thrust, 30 ms motor lag (model assumes 20 ms).
- **Wind:** Ornstein–Uhlenbeck, mean 0.34 N, σ 0.5 N.

| Gripper error, arm phase (RMS) | PID ideal | MPC ideal | PID realistic | MPC realistic |
|---|---|---|---|---|
| Total | 5.6 mm | 1.1 mm | 39.7 mm | 7.5 mm |
| In the arm plane | 5.6 | 1.1 | 20.2 | 4.9 |
| Sideways (the arm can't reach this direction) | 0.0 | 0.0 | 34.2 | 5.7 |
| Holding on pad (mean) | 0.2 | 0.1 | 37.0 | 8.2 |
| Base off its reference (RMS) | 3.4 | 3.4 | 53.3 | 41.6 |
| Max tilt | 1.0° | 0.9° | 6.8° | 7.1° |

Neither controller saturated the rotors. The MPC solved 100% of problems, at 2.6 ms median.

Caveats:
- **PID gains not retuned:** the baseline uses the geometric controller's default gains (position integral 0.6).
- **Integral action differs:** the MPC has an integrator on the gripper error, while the PID has one on base position. The PID's slow integrator explains most of its sideways error.
- **First PID run was unfair:** run at the MPC's 50 Hz, the PID's attitude loop went unstable under the 10 ms delay. Real autopilots run that loop at 250 Hz–1 kHz.

## Correction and extended comparison (2026-10-15, later)
**Correction:** the realistic default-PID result above (39.7 mm, no crash) was one lucky run. Over three seeds, and with the gust strength changed from 0.500 to 0.5025 N, the default PID + IK crashes every time. Freezing the arm makes it survive all three seeds (68–99 mm gripper error), which pins the cause on the decoupled loop: base wobble → IK from the measured base moves the arm → arm reaction moves the base.

### Controllers
- **PID (default):** geometric cascade, position bandwidth 2.5 rad/s, integral 0.6; arm by IK from the measured base pose.
- **PID (tuned):** chosen by `scripts/sweep_wind.py --tune` (18 settings × 2 seeds). Only 2 of 18 survived both seeds; the best is position bandwidth 3.0 rad/s, integral 0.8, no outer gripper integrator. The other surviving setting was worse (53–85 mm).
- **LQR (fixed):** LQI from `uam_quad/lqr.py`, linearized at hover with the nominal arm pose, with feedback on arm states. **It crashes even in ideal conditions** once the arm swings more than 1 rad from the linearization pose: one gain is valid at one pose.
- **LQR (arm-blind):** the same gain with the arm-state columns zeroed. Ideal: 11 mm.
- **LQR (gain-scheduled):** relinearizes every 0.15 rad of arm motion (33 redesigns in the ideal run), arm states not fed back. With arm-state feedback kept, the scheduled LQR flew but poorly (51 mm).
- **MPC:** whole-body, as above.

### Baseline conditions (mean wind 0.335 N; realistic results over 2 seeds)
| Controller | Ideal: gripper RMS, arm phase | Realistic: gripper RMS, arm phase | Realistic: crashes |
|---|---|---|---|
| PID (default) + IK | 5.6 mm | — | 2/2 (plus 3/3 in the recheck) |
| PID (tuned) + IK | 5.6 mm | 27 mm | 0/2 |
| LQR (fixed) + IK | crash | crash | — |
| LQR (gain-scheduled) + IK | 5.5 mm | 30 mm | 0/2 |
| Whole-body MPC | 1.1 mm | 7 mm | 0/2 |

### Wind sweep (`scripts/sweep_wind.py`, gust σ = 1.5 × mean, 2 seeds per cell)
Mean gripper RMS over the runs that didn't crash; ✗ = crashed runs.

| Mean wind [N] | 0.34 | 0.75 | 1.25 | 1.75 | 2.5 | 3.5 |
|---|---|---|---|---|---|---|
| PID (default) | ✗✗ | ✗✗ | ✗✗ | ✗✗ | ✗✗ | ✗✗ |
| PID (tuned) | 27 mm | ✗✗ | ✗✗ | ✗✗ | ✗✗ | ✗✗ |
| LQR (gain-scheduled) | 30 mm | 78 mm | ✗✗ | ✗✗ | ✗✗ | ✗✗ |
| Whole-body MPC | 7 mm | 21 mm | 59 mm | 89 mm | 139 mm (1 ✗) | 202 mm (1 ✗) |

The drone weighs 15.9 N, so 3.5 N mean with σ 5 N gusts is a severe storm for an F450.

Files: `runs/uam_quad/compare/{tune,wind_sweep,wind_sweep_summary}.json`, `wind_sweep.png`, `compare_4way_realistic.mp4`.

## Touch and push (`scripts/push_task.py`)
**Setup:** a vertical board with a compliant contact (45 ms), and a fingertip sphere on the gripper.
- **Force sensing:** the contact force from MuJoCo, filtered, with 0.1 N noise in realistic mode.
- **Admittance loop:** the fingertip reference moves along the board normal at 0.006 m/s per N of force error.
- **Force-aware MPC:** the MPC model takes an external end-effector force parameter (validated against `mj_applyFT` in the tests). Once in contact, the commanded force is fed forward, so the MPC plans the lean.

**Setpoints:** 2 N from 9 s, 4 N from 15 s, release at 20 s.

| Run | 2 N phase: mean ± RMS error | 4 N phase | Contact lost | Max tilt |
|---|---|---|---|---|
| Force-aware, ideal | 1.97 ± 0.28 N | 3.84 ± 0.29 N | 0% | 13.8° |
| Plain MPC, ideal | 0.38 N | 0.43 N | 70% | 2.9° |
| Force-aware, realistic (4 seeds) | 2.16 N | 3.75 N | 6.6% (3.8–10%) | 18.9° |
| Plain MPC, realistic | 0.50 N | 0.70 N | 60% | 7.1° |

**How realistic contact got fixed (2026-10-15):**
- **First attempt:** before the fix, the force-aware run under realistic conditions lost contact 45% of the time (1.71 / 2.89 N, 31° tilt).
- **Isolating the cause:** testing one effect at a time showed the slow disturbances were to blame: model mismatch lost contact 19% of the time and wind 13%. Sensor noise, delay and force noise each stayed under 3%.
- **What fixed it:** contact detection with hysteresis (`CONTACT_ON` = 0.3 N to enter, under 0.1 N for 0.1 s to leave). The force feedforward had been switching on and off with the noisy force signal. An entry threshold of 0.5 N regressed the ideal case to 1.51 N with 6.7% lost; 0.3 N doesn't.
- **What didn't help:** a stiff 10 ms contact bounced and flipped the drone (hence the compliant contact). Tangential integral action (`TANGENTIAL_INTEGRAL`) and the first velocity observer (`wbmpc.DisturbanceObserver`) didn't help either; the observer fought the admittance loop and mistook motor lag for a disturbance.

## Kalman disturbance estimator (`uam_quad/estimator.py`, `scripts/eval_estimator.py`)
**What it estimates:** one world-frame force d on the hub, per axis a [p, v, d] Kalman filter.
- **Prediction:** the nominal model's acceleration from the measured state, the measured contact force and the motor-lagged thrusts.
- **Measurement:** position and velocity.
- **Use:** d goes into the MPC as parameter `p[3:6]`.

It recovers d_z = −2.0 to −2.5 N for the +10% mass and 95% thrust mismatch (−2.4 N expected).

Realistic results, 4 seeds:

| | Gripper RMS | Base deviation RMS | Max tilt |
|---|---|---|---|
| MPC + gripper-error integrator | 9.1 mm | 41.9 mm | 6.9° |
| MPC + Kalman estimator | 8.9 mm | 21.9 mm | 8.0° |

| Push, estimate in contact | Contact lost | 4 N mean | Max tilt |
|---|---|---|---|
| none | 6.6% | 3.75 N | 18.9° |
| live y, z (`est_axes="tangential"`) | 16.8% | 3.62 N | 22.2° |
| live z only (`"vertical"`) | 9.8% | 3.61 N | 18.5° |
| free-flight value held (`"freeze"`) | 30.9% | 3.33 N | 35.6° |

**Conclusion:** use the estimator in free flight, where it halves base drift, and leave it out while in contact.
- **Likely cause (not tested):** the board adds friction and a torque through the arm that a force-only, hub-located estimate can't represent.
- **Next:** a 6-D wrench estimate with a 3-axis fingertip force sensor.
