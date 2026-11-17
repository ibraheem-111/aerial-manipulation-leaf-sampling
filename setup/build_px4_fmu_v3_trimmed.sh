#!/usr/bin/env bash
# Builds a trimmed PX4 v1.16.2 for the Cube Black / Pixhawk 2.1 (FMUv3) flight controller:
#   bash setup/build_px4_fmu_v3_trimmed.sh
# Output: firmware/v1.16.2-trimmed/px4_fmu-v3_trimmed.px4  -> flash with QGroundControl
#         (Configure > Firmware > Advanced settings > Custom firmware file).
#
# Why: stock px4_fmu-v3_default v1.16.2 leaves the Cube Black so short of RAM that with an SD card
# mounted PX4 stops accepting USB input (FTP fails with ENOMEM, commands go unacknowledged, motor test
# is ignored). setup/px4_fmu_v3_trim.diff switches off 40 modules a PWM quadrotor with mocap and uXRCE-DDS
# does not use (fixed-wing/VTOL/rover/UUV, UAVCAN, DShot, camera, gimbal, LPE, attitude_estimator_q,
# SIH, test tools...). Measured: flash 2,010,920 -> 1,478,240 B; static SRAM 29,292 -> 24,700 B; with the
# card in, USB input, FTP, command ACKs and the actuator test all work. Heap headroom is still small
# (~3 KB free at idle with TELEM1 MAVLink, uXRCE-DDS on TELEM2 and logging running).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$ROOT/external"
PX4_TAG="v1.16.2"
SRC="$EXT/PX4-Autopilot-$PX4_TAG"            # separate from external/PX4-Autopilot (the SITL build)
TRIM="$ROOT/setup/px4_fmu_v3_trim.diff"
OUT="$ROOT/firmware/$PX4_TAG-trimmed"
command -v arm-none-eabi-gcc >/dev/null || { echo "arm-none-eabi-gcc not found (sudo apt install gcc-arm-none-eabi)"; exit 1; }
mkdir -p "$EXT" "$OUT"

if [ ! -d "$SRC" ]; then
  git clone --branch "$PX4_TAG" --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/PX4/PX4-Autopilot.git "$SRC"
fi
# A shallow clone has no NuttX release tags; PX4's version header needs one. Fetch the real 11.0.0 tag
# (the version the official v1.16.2 FMUv3 image reports).
NUTTX="$SRC/platforms/nuttx/NuttX/nuttx"
git -C "$NUTTX" rev-parse -q --verify refs/tags/nuttx-11.0.0 >/dev/null ||
  git -C "$NUTTX" fetch -q --depth 1 origin "refs/tags/nuttx-11.0.0:refs/tags/nuttx-11.0.0"

# Apply the trim once (skip if already applied).
if git -C "$SRC" apply --check "$TRIM" 2>/dev/null; then
  git -C "$SRC" apply "$TRIM"
elif git -C "$SRC" apply --check --reverse "$TRIM" 2>/dev/null; then
  echo "trim already applied"
else
  echo "board config differs from both stock and trimmed: inspect $SRC/boards/px4/fmu-v3/default.px4board"; exit 1
fi

[ -d "$EXT/.venv-px4" ] || uv venv -q "$EXT/.venv-px4" --python 3.12
source "$EXT/.venv-px4/bin/activate"
# setuptools<70: PX4's generators still import pkg_resources.
uv pip install -q -r "$SRC/Tools/setup/requirements.txt" ninja "setuptools<70"
make -C "$SRC" px4_fmu-v3_default
deactivate

cp "$SRC/build/px4_fmu-v3_default/px4_fmu-v3_default.px4" "$OUT/px4_fmu-v3_trimmed.px4"
(cd "$OUT" && sha256sum px4_fmu-v3_trimmed.px4 > SHA256SUMS)
echo "== $OUT/px4_fmu-v3_trimmed.px4"; cat "$OUT/SHA256SUMS"
