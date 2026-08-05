#!/usr/bin/env bash
# Write the two data partitions. Offsets come from firmware/partitions.csv:
#   worm   @ 0x190000 (1536K)  — brain, corpus, learned taste  (~1.02 MB)
#   voices @ 0x2D0000 (11M)    — every word in Daniel's voice  (~10.47 MB)
# These change far less often than the app, so they are a separate step.
set -euo pipefail
# The S3 re-enumerates under a different name after a replug, so detect it.
PORT="${1:-$(ls /dev/cu.usbmodem* 2>/dev/null | head -1)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

for f in build/liam.hwrm build/voices.hvox; do
    [ -f "$ROOT/$f" ] || { echo "missing $f — run tools/bake.py and tools/voices.py"; exit 1; }
done

esptool.py --chip esp32s3 --port "$PORT" --baud 921600 \
    write_flash --flash_mode qio --flash_size 16MB \
    0x190000 "$ROOT/build/liam.hwrm" \
    0x2D0000 "$ROOT/build/voices.hvox"
