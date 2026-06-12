#!/usr/bin/env bash
# Emit manifest.txt with one line per input file: "<mode> <abs_path>".
# The array job indexes lines (1-based) via SLURM_ARRAY_TASK_ID.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT="$ROOT/hpc/manifest.txt"

: > "$OUT"
for f in "$ROOT"/data/nue/*.h5; do
    [ -e "$f" ] || continue
    echo "electron $f" >> "$OUT"
done
for f in "$ROOT"/data/bnb/*.h5; do
    [ -e "$f" ] || continue
    echo "pi0 $f" >> "$OUT"
done

N=$(wc -l < "$OUT")
echo "manifest: $N tasks -> $OUT"
