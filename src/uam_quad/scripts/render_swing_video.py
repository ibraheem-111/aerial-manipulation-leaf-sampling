"""Render the arm-swing disturbance test as a side-by-side video with live readouts.

Left: slow swing (±0.5 rad @ 0.5 Hz). Right: fast swing (±1.0 rad @ 1 Hz). Same F450 model,
same geometric controller (with CoM feedforward). Overlays show time, arm phase, base tilt and a
bar per rotor showing the controller's thrust command, red when the command is clipped at 0 N or
the thrust limit (the rotors themselves lag the command through a 30 ms motor time constant).

    python scripts/render_swing_video.py      # -> <repo>/runs/uam_quad/arm_swing_f450.mp4 (+ _still.jpg)
"""
import os

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from uam_quad import GeometricController, UAMConfig, f450_mrs, hover_trim, load_model
from uam_quad.paths import RUNS  # noqa: E402
from uam_quad.controller import quat_to_R

OUT = RUNS
Q0 = np.array([-0.6, 1.0, 0.4])
P_D = np.array([0.0, 0.0, 1.0])
T, FPS, W, H = 11.0, 30, 640, 480
PRESET = os.environ.get("UAM_PRESET", "typical")


def font(size, bold=False):
    for p in (f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_BIG, F_MED, F_SMALL = font(22, True), font(18), font(15)


def simulate(amp, freq):
    model, cfg = load_model(f450_mrs() if PRESET == 'f450_mrs' else UAMConfig())
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    n = cfg.n_rotors
    trim = hover_trim(model, cfg, Q0, pos=P_D)
    data = mujoco.MjData(model)
    data.qpos[:], data.ctrl[:] = trim.qpos, trim.ctrl
    data.act[:n] = trim.act
    mujoco.mj_forward(model, data)
    ctrl = GeometricController(model, cfg)
    servo_offset = trim.ctrl[n:] - Q0
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body("base").id
    cam.distance, cam.azimuth, cam.elevation = 1.35, -90.0, -12.0
    frames, every = [], int(round(1 / (FPS * model.opt.timestep)))
    max_tilt, sat_times, sat_frames = 0.0, [], []
    with mujoco.Renderer(model, H, W) as r:
        for k in range(int(T / model.opt.timestep)):
            t = k * model.opt.timestep
            swinging = 3.0 <= t < 8.0
            s = amp * np.sin(2 * np.pi * freq * (t - 3.0)) if swinging else 0.0
            thrusts, _ = ctrl(data, P_D)
            data.ctrl[:n] = thrusts
            data.ctrl[n:] = Q0 + np.array([s, -s, 0.0]) + servo_offset
            mujoco.mj_step(model, data)
            tilt = np.degrees(np.arccos(np.clip(quat_to_R(data.qpos[3:7])[2, 2], -1, 1)))
            max_tilt = max(max_tilt, tilt)
            # the controller's command clipped at a rotor limit (actual thrust lags it by motor_tau)
            sat_now = bool(np.any((thrusts <= 1e-6) | (thrusts >= cfg.thrust_max - 1e-6)))
            if sat_now:
                sat_times.append(t)
            if k % every == 0:
                r.update_scene(data, camera=cam)
                if sat_now:
                    sat_frames.append(len(frames))
                frames.append(overlay(r.render(), t, swinging, tilt, max_tilt, thrusts.copy(), cfg.thrust_max, amp, freq))
    if sat_times:
        print(f'amp {amp}: rotor command clipped at a limit for {len(sat_times)*model.opt.timestep*1000:.0f} ms total, between t={min(sat_times):.2f}-{max(sat_times):.2f} s; max tilt {max_tilt:.1f} deg')
    else:
        print(f'amp {amp}: no rotor saturation; max tilt {max_tilt:.1f} deg')
    return frames, sat_frames


def overlay(rgb, t, swinging, tilt, max_tilt, thrust, tmax, amp, freq):
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 0, W, 92], fill=(19, 23, 31, 215))
    d.text((14, 10), f"Arm swing ±{amp:g} rad @ {freq:g} Hz", font=F_BIG, fill=(238, 236, 230))
    phase = "SWINGING" if swinging else ("hold" if t < 3 else "hold (after)")
    d.text((14, 42), f"t = {t:4.1f} s   arm: {phase}", font=F_MED, fill=(242, 154, 74) if swinging else (201, 204, 212))
    d.text((14, 66), f"base tilt {tilt:4.1f}°   (max so far {max_tilt:4.1f}°)", font=F_MED, fill=(238, 236, 230))
    # rotor thrust bars, bottom-right
    x0, y0, bw, bh = W - 190, H - 128, 34, 90
    d.rectangle([x0 - 12, y0 - 30, W - 8, H - 8], fill=(19, 23, 31, 215))
    d.text((x0 - 4, y0 - 26), f"rotor command (0–{tmax:.1f} N)", font=F_SMALL, fill=(201, 204, 212))
    for i, f in enumerate(thrust):
        x = x0 + i * (bw + 10)
        sat = f <= 1e-6 or f >= tmax - 1e-6
        d.rectangle([x, y0, x + bw, y0 + bh], outline=(120, 125, 135), width=1)
        hgt = bh * np.clip(f / tmax, 0, 1)
        d.rectangle([x, y0 + bh - hgt, x + bw, y0 + bh], fill=(230, 70, 50) if sat else (90, 160, 230))
        d.text((x + 4, y0 + bh + 2), f"{f:3.1f}", font=F_SMALL, fill=(255, 120, 100) if sat else (201, 204, 212))
    if np.any((thrust <= 1e-6) | (thrust >= tmax - 1e-6)):
        d.rectangle([14, 104, 300, 136], fill=(200, 50, 35, 230))
        d.text((24, 109), "COMMAND CLIPPED AT LIMIT", font=F_MED, fill=(255, 255, 255))
    return np.asarray(im)


def main():
    os.makedirs(OUT, exist_ok=True)
    (slow, _), (fast, sat) = simulate(0.5, 0.5), simulate(1.0, 1.0)
    frames = [np.concatenate([a, np.full((H, 8, 3), 245, np.uint8), b], axis=1) for a, b in zip(slow, fast)]
    name = "arm_swing_f450" if PRESET == "typical" else f"arm_swing_{PRESET}"
    path = os.path.join(OUT, name + ".mp4")
    imageio.mimwrite(path, frames, fps=FPS, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=8)
    still = frames[sat[0] if sat else int(6.6 * FPS)]  # first frame with a rotor at its limit
    Image.fromarray(still).save(os.path.join(OUT, name + "_still.jpg"), quality=88)
    print(path, os.path.getsize(path) // 1024, "KB", len(frames), "frames", frames[0].shape)


if __name__ == "__main__":
    main()
