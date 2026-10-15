"""2x2 video of the four controllers on the arm task in realistic conditions (same seed and wind)."""
import os
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_controllers as cc  # noqa: E402

LABELS = {"pid": "PID (default) + arm IK", "pid_tuned_noouter": "PID (tuned) + arm IK",
          "lqr_sched": "LQR (gain-scheduled) + arm IK", "mpc": "Whole-body MPC"}


def main(seed=0, wind=0.335):
    tiles = {}
    for c in LABELS:
        r, _, fr = cc.run(c, True, video=True, seed=seed, wind_mag=wind)
        tiles[c] = fr
        print(c, {k: r[k] for k in ("crashed", "ee_rms_arm_phase_mm", "max_tilt_deg")}, flush=True)
    fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    fn = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    n = min(len(v) for v in tiles.values())
    out = []
    for i in range(n):
        ims = []
        for c in LABELS:
            img, t, ph, e, tilt = tiles[c][i]
            I = Image.fromarray(img).resize((640, 480)); d = ImageDraw.Draw(I, "RGBA")
            d.rectangle([0, 0, 640, 58], fill=(19, 23, 31, 215))
            d.text((10, 5), LABELS[c], font=fb, fill=(238, 236, 230))
            d.text((10, 32), f"{ph} · t {t:4.1f} s · gripper {e * 1000:6.1f} mm · tilt {tilt:5.1f}°", font=fn, fill=(242, 154, 74))
            ims.append(np.asarray(I))
        top = np.concatenate([ims[0], np.full((480, 6, 3), 245, np.uint8), ims[1]], 1)
        bot = np.concatenate([ims[2], np.full((480, 6, 3), 245, np.uint8), ims[3]], 1)
        out.append(np.concatenate([top, np.full((6, top.shape[1], 3), 245, np.uint8), bot], 0))
    path = f"{cc.OUT}/compare_4way_realistic.mp4"
    imageio.mimwrite(path, out, fps=30, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=2)
    print("video", path)


if __name__ == "__main__":
    main()
