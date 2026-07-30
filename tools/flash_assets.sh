#!/usr/bin/env bash
# Write the two data partitions. Offsets come from firmware/partitions.csv:
#   worm   @ 0x310000 (1536K)  — brain, corpus, learned taste  (~1.02 MB)
#   voices @ 0x490000 (11M)    — every word in Daniel's voice  (~10.47 MB)
# These change far less often than the app, so they are a separate step.
set -euo pipefail
PORT="${1:-/dev/cu.usbmodem101}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

for f in build/liam.hwrm build/voices.hvox; do
    [ -f "$ROOT/$f" ] || { echo "missing $f — run tools/bake.py and tools/voices.py"; exit 1; }
done

esptool.py --chip esp32s3 --port "$PORT" --baud 921600 \
    write_flash --flash_mode qio --flash_size 16MB \
    0x310000 "$ROOT/build/liam.hwrm" \
    0x490000 "$ROOT/build/voices.hvox"
