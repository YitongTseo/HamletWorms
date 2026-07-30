#!/usr/bin/env bash
# Bake every worm in a flask, one asset per board.
#
# flask_1 of gen-0007 holds exactly sixteen worms — Alice through Peggy — which
# is one per board for a sixteen-board wall. Each gets its own seed and its own
# evolved connectome, so sixteen boards show sixteen genuinely different animals
# from the same generation rather than sixteen copies.
#
#   ./tools/bake_flask.sh                 -> build/flask/<Name>.hwrm
#   ./tools/bake_flask.sh flask_2 gen-0007
set -euo pipefail
FLASK="${1:-flask_1}"
GEN="${2:-gen-0007}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/../HamletRNAWorld/v7/data/poetry-2/generations/$FLASK/$GEN"
OUT="$ROOT/build/flask"
mkdir -p "$OUT"

[ -d "$SRC" ] || { echo "no such generation: $SRC"; exit 1; }

n=0
for dir in "$SRC"/*/; do
    worm="$(basename "$dir")"
    [ -f "$dir/weights.json" ] || continue
    printf '%-10s ' "$worm"
    python3 "$ROOT/tools/bake.py" --flask "$FLASK" --gen "$GEN" --worm "$worm" \
        --out "$OUT/$worm.hwrm" | awk '/^worm /{printf "%-9s %-16s ", $4, $5} /TOTAL/{print $2" B"}'
    n=$((n + 1))
done
echo
echo "$n worms -> $OUT"
echo "flash one with:  esptool.py --chip esp32s3 --port <PORT> --baud 921600 \\"
echo "                   write_flash 0x310000 build/flask/<Name>.hwrm"
echo "the voice bank at 0x490000 is identical on every board — flash it once each."
