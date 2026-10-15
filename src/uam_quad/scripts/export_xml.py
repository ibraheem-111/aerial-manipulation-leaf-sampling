"""Write the generated models to plain MJCF files you can open with MuJoCo's own tools.

    python scripts/export_xml.py
    python -m mujoco.viewer --mjcf=src/uam_quad/models/f450_omx.xml     (from the repo root)
Mesh paths are written relative to the models/ folder, so the files work wherever the repo lives.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uam_quad import UAMConfig, build_mjcf, f450_stock, with_omx_arm  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def main():
    os.makedirs(OUT, exist_ok=True)
    from push_task import SCENE as PUSH_SCENE
    models = {
        "f450_omx.xml": build_mjcf(with_omx_arm(f450_stock())),
        "f450_omx_push_scene.xml": build_mjcf(with_omx_arm(f450_stock()), extra_world=PUSH_SCENE),
        "simple_quad_capsule_arm.xml": build_mjcf(UAMConfig()),
    }
    pkg = os.path.dirname(OUT)
    for name, xml in models.items():
        xml = xml.replace(os.path.join(pkg, "assets") + os.sep, "../assets/")   # absolute -> relative mesh paths
        with open(os.path.join(OUT, name), "w") as f:
            f.write(xml)
        print("wrote", os.path.join(OUT, name))


if __name__ == "__main__":
    main()
