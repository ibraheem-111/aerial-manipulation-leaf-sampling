# Questions Das may ask (answer out loud first, then check)

**MuJoCo basics**
1. *What is the difference between `MjModel` and `MjData`?*
   The model is the fixed description (geometry, masses, actuators). The data is the time-varying state (`qpos`, `qvel`, `ctrl`, contacts) plus everything computed from it.
2. *Why is `nq` 10 but `nv` 9?*
   The free joint stores orientation as a 4-number quaternion, but it has only 3 rotational degrees of freedom (angular velocity), plus 3 arm joints in each case.
3. *What does `mj_step` do vs `mj_forward`?*
   `mj_step` integrates one timestep (2 ms here). `mj_forward` recomputes kinematics, forces and accelerations for the current state without advancing time.
4. *How are the rotors modeled?*
   Each is a `general` actuator on a site: force along the body z axis, a yaw torque proportional to thrust, and a first-order lag (`dyntype="filter"`, 20–30 ms) standing in for the ESC and motor. There's no blade aerodynamics.
5. *How does the arm get commanded?*
   MuJoCo `position` actuators (PD servos, torque-limited like the XM430) track joint targets. The controller sends joint angles, the same interface a Dynamixel bus would take.
6. *Where does the airflow / downwash come in?*
   It doesn't yet. MuJoCo doesn't simulate air. Rotor forces push only the drone; wind is an external force we apply with `xfrc_applied`.
7. *How do you measure contact force?*
   Loop over `data.contact` and call `mujoco.mj_contactForce`; the first component is the normal force. That's the stand-in for a fingertip force sensor.

**Dynamics and control**
8. *Why can't a quad move sideways without tilting?*
   Upright rotors only push along body z. The allocation matrix has rank 4 (thrust and three moments), so it's underactuated. Sideways force comes from tilting the thrust vector.
9. *What does the arm do to the drone?*
   Three things:
   - It shifts the center of mass (a static pitch moment, 0.8 N·m with the arm forward).
   - It pushes back when it accelerates (reaction torque, the biggest effect).
   - It changes the inertia.

   The equations of motion page derives all of these, and they match MuJoCo.
10. *Why cascaded control?*
    Position responds to moments through four integrators (moment → rate → angle → velocity → position). So inner loops (rates, attitude) run faster than outer ones (position), each designed on a simple model.
11. *PID vs LQR vs MPC, in one sentence each?*
    - **PID:** model-light and easy to tune; what PX4 runs.
    - **LQR:** optimal for one linearized operating point, with no constraints or look-ahead.
    - **MPC:** solves a constrained optimal control problem over a horizon at every step, with the nonlinear model.
12. *Why did fixed-gain LQR crash here?*
    It's linearized at one arm pose. When the arm moved more than 1 rad away, its linear model of the arm's effect was wrong. Gain scheduling (relinearizing as the arm moves) fixed it.
13. *Why did PID + IK crash in realistic conditions?*
    It's a feedback loop: drone wobbles → IK (computed from the measured drone pose) moves the arm → the arm reaction moves the drone. Freezing the arm stopped the crashes. The MPC plans arm and drone together, so it doesn't have this loop.
14. *What is the "coupling" in whole-body MPC, and how do you know it matters?*
    The arm-motion terms in the model (reaction torque, moving center of mass). Without them, the MPC flips the drone during fast arm motion; with them, it tracks to millimeters.
15. *What did "realistic" mean in your tests?*
    - **Sensors:** noise, and a 10 ms delay.
    - **Model mismatch:** airframe 10% heavier, rotors give 95% of commanded thrust, motors 50% slower than the model.
    - **Wind:** gusty, random-walk.
16. *How does touch-and-push work?*
    An admittance loop moves the fingertip reference into the board until the measured force matches the setpoint, and the MPC tracks that reference. The force-aware version tells the MPC the contact force, so it plans the lean. Without it, the drone gets pushed back off the board.
17. *What doesn't work yet?*
    The disturbance estimator in contact. The Kalman estimator (a world-frame force on the hub) halves base drift in free flight. But every way of using it while pushing made contact worse than leaving it out, likely because board friction and the contact torque through the arm aren't a force at the hub. Realistic pushing works without it (6.6% contact lost), thanks to contact detection with hysteresis. Next is a force + torque estimate with a real fingertip force sensor.
18. *What is the disturbance estimator, in which frame?*
    One force vector d = (d_x, d_y, d_z) in newtons, in the world frame (x forward, y left, z up), acting at the hub. It is everything the model gets wrong: wind, the extra mass, weak rotors. A Kalman filter per axis compares measured motion with the model's predicted motion. It found d_z = −2.0 to −2.5 N for a 10% heavier drone with 95% thrust (−2.4 N expected).

**Platform**
19. *Is PX4 in the loop?*
    No. In simulation the controller writes rotor thrusts and joint targets directly to MuJoCo. On hardware, the MPC would send thrust plus body rates to PX4 in offboard mode, and joint targets to the Dynamixel bus.
20. *Where do the F450 numbers come from?*
    - **Sourced:** frame 282 g and E310 propulsion at 7.85 N per rotor (DJI specs).
    - **Estimated:** battery, ESCs and avionics masses. They need to be weighed.
    - **Arm:** ROBOTIS OpenMANIPULATOR-X CAD masses.
