# Multirotor control fundamentals: study guide

Built on `docs/eom.md` and the code in this package. Each section ends with an exercise you run on
the F450 + arm model and a "say it in two sentences" check: if you can answer that out loud without
notes, you can defend it in a meeting.

## 1. Underactuation and what it forces on the design
Four independent inputs (thrust + three moments), six DoF. Thrust only points along body $z$, so
horizontal motion must be produced by tilting.

Consequence: you choose four things to track. The standard choice is position $(x, y, z)$ and yaw
$\psi$. These are the **flat outputs** (Mellinger & Kumar 2011): given a smooth trajectory of them,
the attitude, rates, thrust and moments needed are all determined. Roll and pitch are *not* yours to
choose; they follow from the desired acceleration.

**Exercise.** Using `UAMConfig().allocation_matrix()`, check `np.linalg.matrix_rank` for 4 and 6
rotors. Then explain why a hexarotor with upright rotors is still underactuated.

**Two sentences.** *Why can't the drone hold a tilted attitude while hovering in place?*

## 2. Cascaded control and time-scale separation
Because position reaches the moments through four integrators (eom.md §5), controllers are nested
loops, each faster than the one outside it:

```
position (≈1–2 Hz bandwidth) → desired acceleration → desired attitude + thrust
attitude (≈5–10 Hz)           → desired body rates
rates (≈20–50 Hz)             → desired moments → allocation → rotor thrusts
```

PX4 is structured this way: position P → velocity PID → attitude P → rate PID → mixer. The rule of
thumb is that each inner loop should be roughly 3–5× faster than the loop around it. The outer loop
can then treat the inner one as "instant," and each loop can be designed on its own simple model.

**Exercise.** In `controller.py` the default `Gains` have `pos_wn = 2.5` and `att_wn = 18` rad/s. Set
`att_wn = 6` (closer to the position loop) and rerun `scripts/arm_swing_test.py`. Note what happens to
the overshoot. Then try `pos_wn = 6` with `att_wn = 18`.

**Two sentences.** *Why must the attitude loop be faster than the position loop?*

## 3. Attitude control: a double integrator with gains you can compute
Near hover each axis is $J\,\ddot\eta = M$. A PD law $M = -k_p\,e - k_d\,\dot e$ gives
$J s^2 + k_d s + k_p = 0$, so

$$\omega_n = \sqrt{k_p / J}, \qquad \zeta = \frac{k_d}{2\sqrt{k_p J}}
\quad\Longleftrightarrow\quad k_p = J\omega_n^2,\ \ k_d = 2\zeta\omega_n J.$$

That is exactly how `controller.py` sets `kR` and `kW`, using the current inertia $J_O$ from MuJoCo.
For the default pose, $J_O \approx \mathrm{diag}(0.015, 0.039, 0.044)$ kg·m². With $\omega_n = 18$,
$\zeta = 0.8$: $k_p \approx (4.7, 12.8, 14.2)$, $k_d \approx (0.42, 1.13, 1.26)$.

**Exercise.** Compute those numbers by hand from the $J_O$ above, then print `ctrl.mass_props(data)`
inside the controller and compare. Why is roll inertia (~0.015) so much smaller than pitch (~0.039)?
(Hint: where is the arm?)

**Two sentences.** *What do $\omega_n$ and $\zeta$ mean physically for the drone's response to a push?*

## 4. Why geometric (SO(3)) attitude control instead of Euler angles
Euler angles hit singularities (gimbal lock at ±90° pitch), and the error "difference of angles"
isn't a proper rotation error. The geometric controller (Lee et al. 2010) uses

$$e_R = \tfrac12\big(R_d^\top R - R^\top R_d\big)^\vee,$$

which is well defined for any attitude and reduces to the small-angle error near hover. Same PD
structure, valid globally (almost).

**Exercise.** Read the 10 lines of `GeometricController.__call__` that build $R_d$ from the desired
force direction and yaw. Write each line next to its equation in Lee et al. 2010, Section III.

## 5. From position error to thrust and attitude
The position loop outputs a desired *force* $F_d = -k_x e_x - k_v e_v + m g e_3 + m\ddot p_d$.
Its direction becomes the desired body $z$ axis, $b_{3d} = F_d / \lVert F_d\rVert$; yaw fixes the rest
of $R_d$. The thrust actually applied is the projection $T = F_d \cdot R e_3$: if the drone isn't
tilted yet, it doesn't push hard in the wrong direction.

**Two sentences.** *Why is thrust projected onto the current body axis instead of using $\lVert F_d\rVert$?*

## 6. Allocation and saturation
$f = A^{+}[T, M]$ then clip to $[0, f_{\max}]$. Clipping breaks the requested moments, which is how
attitude control degrades under aggressive commands. Real autopilots prioritize attitude over
thrust when desaturating (PX4's "airmode" is one variant).

In our F450 test the arm out front already uses 73% of front-rotor thrust at hover. The ±1 rad, 1 Hz
swing drives commands to 0 / 8 N for 62 ms when the swing stops (`runs/uam_quad/arm_swing_f450.mp4`).

**Exercise.** Lower `thrust_max` in `UAMConfig` to 7 N and rerun the fast swing. At what value does
the drone lose attitude? This is a payload-margin question you should be able to answer for the
real F450 once it's weighed.

## 7. LQR on the linearized model
`linearize(model, hover_trim(...))` gives discrete $(A, B)$ with 22 states. With
`scipy.linalg.solve_discrete_are` you get an LQR gain that handles all the couplings in one shot
(including CoM offset terms at that pose). It is optimal only near the trim point.

**Exercise.** Build the LQR for rotor inputs only (first 4 columns of $B$) with the arm held. Compare
its arm-swing response with the geometric controller. Note that it needs every state, including rotor
thrust states you wouldn't measure on hardware.

## 8. Disturbance rejection: four ways to handle what the model misses
| Approach | Idea | Here |
|---|---|---|
| Integral action | Remove steady error | `pos_ki` in `Gains` |
| Feedforward | Cancel known terms before they act | CoM gravity moment (`com_ff`) |
| Disturbance observer / L1 adaptive | Estimate the unknown wrench online, subtract it | Flying Hand / UMI-on-Air |
| Robust ($H_\infty$) | Design for the worst bounded disturbance | Your existing background |

The arm reaction ($\dot h$ in eom.md) is *known* if you know the commanded arm motion. That makes it
a feedforward candidate, or a term for the MPC model, instead of something to estimate after it
hits.

**Two sentences.** *Why is the arm reaction torque better handled by a model than by an estimator?*

## 9. Where MPC fits
MPC repeatedly solves a short-horizon optimal control problem with the model and constraints
(thrust limits, joint limits, tilt limits), applies the first input, and repeats. Compared with
the cascade it handles the coupling and the saturation explicitly. The cost is computation, and you
need a model good enough to predict with. Sections 4.3–4.5 of eom.md are that model.

## Reading list (in order)
1. Mahony, Kumar & Corke, "Multirotor aerial vehicles," IEEE RAM 2012: modeling and the cascade.
2. Lee, Leok & McClamroch, CDC 2010: geometric control (what `controller.py` implements).
3. D. Mellinger, V. Kumar, "Minimum snap trajectory generation and control for quadrotors," ICRA 2011, pp. 2520–2525: differential flatness and trajectory generation.
4. PX4 docs, "Controller Diagrams": how the cascade looks in a real autopilot.
5. Ruggiero, Lippiello & Ollero, RA-L 2018: aerial manipulation survey, for the arm coupling.
