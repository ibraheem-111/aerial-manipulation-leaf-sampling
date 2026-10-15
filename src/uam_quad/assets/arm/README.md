# Planar 3-DoF arm from ROBOTIS OpenMANIPULATOR-X

Real-arm replacement for the placeholder capsule arm in `uam_quad/model.py`. It uses the three parallel
pitch joints of the ROBOTIS OpenMANIPULATOR-X (OMX, RM-X52-TNM): joint2 (shoulder), joint3 (elbow) and
joint4 (wrist). The base yaw joint (joint1) is dropped and held at 0. All joints are DYNAMIXEL XM430-W350-T.

| uam_quad | OMX source | joint→next joint [m] | mass [kg] | range [rad] (uam convention) |
|---|---|---|---|---|
| `arm_l1` / `arm_j1` | link3 / joint2 | 0.130231 (= ‖(0.024, 0, 0.128)‖) | 0.138509 | [-2.8854, 0.1146] |
| `arm_l2` / `arm_j2` | link4 / joint3 | 0.124 | 0.132746 | [-0.1146, 2.7854] |
| `arm_l3` / `arm_j3` | link5 / joint4 (gripper body + gripper servo) | 0.126 to `ee` | 0.143276 | [-1.70, 1.97] |
| `finger_left/right` | gripper palms (frozen) | – | 0.001 each | – |
| `arm_mount` (fixed, on airframe) | link2 (joint2 servo + bracket) | – | 0.098407 | – |

Moving arm mass is 0.41653 kg. With the mount it is 0.51494 kg. CoMs, full inertias, mesh bounds and
sources are in `arm_spec.json`.

Frame convention: each link frame sits on its joint axis, with x pointing to the next joint and y as the
joint axis. Zero angle means the link points along +x, and +q pitches the tip down. OMX link3 is L-shaped,
so the `arm_l1` frame is OMX link3 rotated by Ry(-θ1), where θ1 = atan2(0.128, 0.024) = 1.385448 rad.
Joint angles map as:
`arm_j1 = q2 − θ1`, `arm_j2 = q3 + θ1`, `arm_j3 = q4`.
The OMX "home" pose (all zero) is therefore (−θ1, +θ1, 0).

## Files
- `meshes/*.obj`: OMX STLs converted mm → m. Vertices are already expressed in the uam link frame, so each
  mesh sits at pose = identity. `arm_mount.obj` has its origin on the `arm_j1` axis.
- `arm_spec.json`: machine-readable parameters with source URLs.
- `arm_only.xml`: fixed-base test model. It uses the same joint, body and site names as `model.py`, and
  position servos with kp=20, kv=0.6 and ±4.1 N·m.
- `arm_only.png`: render. Top: q=0. Middle: OMX home pose. Bottom: (-0.6, 1.2, 0.5).
- `build_arm.py`: regenerates everything from a clone of the OMX repo.

## Source and license
Robot description: https://github.com/ROBOTIS-GIT/open_manipulator, branch `main`, commit
`d31000d90c679af9c982e73de8b12d777c5ff7dd`, from
`open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x_arm.urdf.xacro` and
`open_manipulator_description/meshes/open_manipulator_x/`. The ROS 1 `noetic` branch has identical
inertials and meshes, and only its joint limits differ.

Copyright ROBOTIS CO., LTD. Licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0). Modifications: the STL meshes were rescaled to meters,
re-expressed in rotated link frames and converted to OBJ. The inertials were rotated into the same
frames. The joint subset was extracted into MJCF.

Servo data: ROBOTIS e-Manual, XM430-W350-T/R, https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/.
It weighs 82 g and has a 353.5:1 gear ratio. Stall torque is 3.8 / 4.1 / 4.8 N·m and no-load speed is
43 / 46 / 57 rpm, at 11.1 / 12 / 14.8 V. The operating range is 10–14.8 V.
OMX system specs (0.70 kg total, 500 g payload, 380 mm reach) come from
https://emanual.robotis.com/docs/en/platform/openmanipulator_x/specification/.

## Caveats
- Link inertials are ROBOTIS' CAD values. They were not measured. The finger palms use the URDF
  placeholder values (1 g, 1e-6 kg·m²).
- The servo gains kp and kv are carried over from the placeholder and were not identified. The URDF gives
  no damping, friction or armature data that maps to an XM430 (its `damping=0.1` is a ROS sim value), so
  `damping=0.02` and `armature=0.002` are kept from the placeholder.
- `arm_mount` is OMX link2 (the yaw-side bracket that holds the joint2 servo). A real airframe mount
  would be a custom bracket. OMX link1 (the yaw servo and base, 0.0791 kg) is not included.
