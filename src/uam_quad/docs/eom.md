# Equations of motion: multirotor + 3-DoF arm on top

This derivation matches `uam_quad/eom.py`, which implements it and checks it against MuJoCo
(`python -m pytest tests -k eom`). Numbers are from the default F450 model (typical specs, not
measured on the lab airframe).

## 1. Frames and state

- World frame **W**: $z$ up, gravity $-g\,e_3$.
- Body frame **B**: attached to the base, origin $O$ at the base body origin, $x$ forward, $y$ left, $z$ up.
- $p \in \mathbb{R}^3$: position of $O$ in W. $R \in SO(3)$: maps B vectors to W.
- $\omega \in \mathbb{R}^3$: base angular velocity in B. Attitude kinematics: $\dot R = R\,\hat\omega$
  ($\hat\omega$ is the skew matrix, $\hat\omega v = \omega \times v$).
- $\theta = (\theta_1,\theta_2,\theta_3)$: arm joint angles, all about body $+y$.

Generalized velocity, the same ordering MuJoCo uses for a free joint plus 3 hinges:
$\nu = (\dot p_W,\ \omega_B,\ \dot\theta) \in \mathbb{R}^9$.

## 2. Rotors and the allocation matrix

Rotor $i$ sits at $r_i = (x_i, y_i, z_r)$ in B and pushes along $+z_B$ with thrust $f_i$.
Its drag torque on the body is $-s_i k_m f_i$ about $z_B$ ($s_i=+1$ for a CCW prop, which twists the body
clockwise). The moment of a force along $z$ at $r_i$ is $r_i \times f_i e_3 = (y_i f_i,\ -x_i f_i,\ 0)$,
so the rotor wrench about $O$ is

$$
\begin{bmatrix} T \\ M_x \\ M_y \\ M_z \end{bmatrix}
= \underbrace{\begin{bmatrix} 1 & \cdots & 1 \\ y_1 & \cdots & y_n \\ -x_1 & \cdots & -x_n \\ -s_1 k_m & \cdots & -s_n k_m \end{bmatrix}}_{A\ (4\times n)} f,
\qquad F_B = T e_3 .
$$

For the F450 ($0.225$ m arm radius, X layout, $k_m = 0.016$ m):

```
A = [[ 1       1       1       1     ]
     [ 0.159   0.159  -0.159  -0.159 ]
     [-0.159   0.159   0.159  -0.159 ]
     [-0.016   0.016  -0.016   0.016 ]]
```

**Why it's underactuated.** $A$ has rank 4: the rotors produce thrust along $z_B$ and three moments,
never a force along $x_B$ or $y_B$. Six rigid-body DoFs, four independent inputs. An upright hexarotor
is the same: six columns, still rank 4. Fully actuated frames (Flying Hand) tilt the rotors so the
force rows are non-zero and $A$ reaches rank 6.

Motor dynamics (first order, $\tau_m = 30$ ms): $\tau_m \dot f_i = f_{i,\text{cmd}} - f_i$, with
$0 \le f_i \le f_{\max} = 8$ N.

## 3. The bare quadrotor (arm locked, CoM at O)

Newton for the CoM and Euler for the rotation, in the standard form (Mahony, Kumar & Corke 2012;
Lee et al. 2010):

$$
\dot p = v, \qquad m\dot v = -m g\, e_3 + T R e_3,
$$
$$
\dot R = R\hat\omega, \qquad J\dot\omega + \omega \times J\omega = M .
$$

Read it as: thrust can only point along the body $z$ axis ($Re_3$), so **to accelerate sideways
the vehicle must first rotate**. The rotation is driven by the moments $M$ through the rigid-body
Euler equation; $\omega \times J\omega$ is the gyroscopic term (small near hover).

## 4. Adding the arm

### 4.1 Arm kinematics
With $\phi_k = \theta_1 + \dots + \theta_k$ and $R_y(\phi)$ the rotation about $y$
(it maps $+x$ toward $-z$ for $\phi > 0$, so positive angles pitch the arm down):

$$
P_1 = a_0,\qquad P_{k+1} = P_k + R_y(\phi_k)\,[L_k, 0, 0]^\top,
$$

where $a_0$ is the shoulder mount on top of the base. Body $k$ has mass $m_k$, CoM $r_k = P_k + R_y(\phi_k)\,\bar r_k$
and inertia $I_k = R_y(\phi_k)\,\bar I_k\,R_y(\phi_k)^\top$ (bars: in the link's own frame).

### 4.2 Three arm-dependent quantities (all in B)

$$
m = m_b + \textstyle\sum_k m_k, \qquad
c(\theta) = \tfrac{1}{m}\big(m_b r_b + \textstyle\sum_k m_k r_k\big) \quad\text{(system CoM relative to } O\text{)}
$$
$$
J_O(\theta) = \textstyle\sum_{\text{all bodies}} \big[I_k + m_k(\lVert r_k\rVert^2 \mathbb{I} - r_k r_k^\top)\big]
\quad\text{(system inertia about } O\text{)}
$$
$$
h(\theta,\dot\theta) = \textstyle\sum_k \big[m_k\, r_k \times \dot r_k + I_k\,\dot\phi_k e_2\big]
\quad\text{(angular momentum from the arm moving relative to the base)}
$$

### 4.3 Translation: total linear momentum
The CoM is at $p + R c$ in W. Differentiating twice and writing everything in B with
$a_O = R^\top \ddot p$:

$$
m\big(a_O + \dot\omega\times c + \omega\times(\omega\times c) + 2\,\omega\times\dot c + \ddot c\big)
= T e_3 - m g\,R^\top e_3 .
$$

### 4.4 Rotation: Euler's equation about the moving point O
For a reference point that accelerates, $\sum M_O = \dot H_O + m\, c\times a_O$, with
$H_O = J_O\,\omega + h$ (angular momentum about $O$). Differentiating $H_O$ in the rotating frame
adds $\omega \times H_O$:

$$
J_O\dot\omega + \dot J_O\,\omega + \omega\times J_O\omega + \dot h + \omega\times h + m\,c\times a_O
= M + m\,c\times(-g R^\top e_3).
$$

### 4.5 Coupled form
The two equations are linear in $(a_O, \dot\omega)$:

$$
\begin{bmatrix} m\mathbb{I} & -m\hat c \\ m\hat c & J_O \end{bmatrix}
\begin{bmatrix} a_O \\ \dot\omega \end{bmatrix}
=
\begin{bmatrix} T e_3 - m g R^\top e_3 - m\big(\omega\times(\omega\times c) + 2\omega\times\dot c + \ddot c\big) \\
M + m\,c\times(-gR^\top e_3) - \dot J_O\omega - \omega\times J_O\omega - \dot h - \omega\times h \end{bmatrix}.
$$

This is the top 6 rows of the usual multibody form $M(q)\dot\nu + b(q,\nu) = S^\top u$, written so that
each coupling term has a name. Setting $c = 0$, $h = 0$, $\dot J_O = 0$ gives back Section 3.

### 4.6 What each coupling term is
| Term | Physical meaning | Max error if dropped (aggressive random states) | In the baseline controller? |
|---|---|---|---|
| $m\,c\times(-gR^\top e_3)$, $m\hat c$ blocks | **CoM offset**: arm out front makes gravity pitch the nose down | 26 m/s², 74 rad/s² | Yes: the CoM feedforward cancels the static gravity moment |
| $\dot h + \omega\times h$ | **Arm reaction**: accelerating the arm torques the base the other way | 31 m/s², **650 rad/s²** | No |
| $2\omega\times\dot c + \ddot c$ | **Moving CoM**: the CoM shifts as the arm moves | 44 m/s², 112 rad/s² | No |
| $\dot J_O\,\omega$ | **Changing inertia** while the base rotates | 0.6 m/s², 30 rad/s² | No (uses current $J_O$ only) |

The error column is from `check_against_mujoco(..., drop=(...))` with joint speeds of a few rad/s;
at hover with slow arm motion they are much smaller. The ranking is the point: the arm reaction
dominates rotation, which is why the fast arm swing still tilts the F450 by 10.6° with the CoM
feedforward on.

### 4.7 Arm joints
The servos are position-controlled, so (like Flying Hand) the arm is modeled kinematically with a
first-order lag, $\beta\,\dot\theta + \theta = \theta_{\text{cmd}}$. In the MuJoCo model each joint is a
PD position servo ($k_p = 20$ N·m/rad, $k_v = 0.6$, torque limit 6 N·m). The base equations above
hold for any arm motion; how $\theta$ follows $\theta_\text{cmd}$ is a separate, simpler model.

## 5. Hover and linearization

**Trim.** At rest ($\omega = 0$, $a_O = 0$, $\dot\theta = 0$, level):
$T = mg$ and $M = -m\,c\times(-g e_3) = (m g\,c_y,\ -m g\,c_x,\ 0)$.
With the default arm pose $\theta = (-0.6, 1.0, 0.4)$: $m = 1.84$ kg, $c = (0.046, 0, 0.018)$ m, so
$M_y = -0.825$ N·m. Solving $A f = (mg, 0, -0.825, 0)$: front rotors 5.81 N, rear 3.22 N.
That's 73% of the 8 N limit before any maneuvering.

**Small-angle model** (arm still, roll $\varphi$, pitch $\vartheta$, yaw $\psi$, deviations from hover):

$$
\delta\ddot x \approx g\,\delta\vartheta,\quad \delta\ddot y \approx -g\,\delta\varphi,\quad
\delta\ddot z = \delta T/m,\quad J_O\,\delta\dot\omega \approx \delta M,\quad \delta\dot\eta \approx \delta\omega .
$$

So $x$ responds to a pitch moment through **four integrators** (moment → rate → angle → velocity →
position). That chain is why multirotor control is cascaded: you can't command $\ddot x$ directly,
only the attitude that produces it. `uam_quad.linearize()` gives the full 22-state discrete model
(9 positions, 9 velocities, 4 rotor states) that includes the arm terms; it is controllable
(`test_hover_linearization_is_controllable`).

## 6. Check it yourself

```bash
source setup/env.sh && cd src/uam_quad      # from the repo root
python -m pytest -q tests -k eom
python -c "from uam_quad import load_model; from uam_quad.eom import check_against_mujoco as c; m,cfg=load_model(); print(c(m,cfg,trials=50))"
```

Agreement is about $10^{-5}$ m/s² and $10^{-4}$ rad/s², limited by the finite differences used for
$\dot c$, $\ddot c$, $\dot J_O$, $\dot h$, not by the model (changing the step changes the error).

## References
- T. Lee, M. Leok, N. H. McClamroch, "Geometric tracking control of a quadrotor UAV on SE(3)," IEEE CDC, 2010.
- R. Mahony, V. Kumar, P. Corke, "Multirotor aerial vehicles: Modeling, estimation, and control of quadrotor," IEEE Robotics & Automation Magazine, 19(3), 2012.
- R. Featherstone, *Rigid Body Dynamics Algorithms*, Springer, 2008.
- F. Ruggiero, V. Lippiello, A. Ollero, "Aerial manipulation: A literature review," IEEE Robotics and Automation Letters, 3(3), 2018.
- G. He et al., "Flying Hand," arXiv 2504.10334, 2025 (kinematic arm + L1 for the coupling terms).
