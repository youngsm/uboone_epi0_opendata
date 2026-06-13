#!/usr/bin/env bash
# Emit hpc/manifest_full.txt: one absolute input path per line. Full-event
# mode doesn't care which sample a file came from — class labels come from
# truth — so both nue and bnb files share the same task definition.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT="$ROOT/hpc/manifest_full.txt"

: > "$OUT"
for f in "$ROOT"/data/nue/*.h5 "$ROOT"/data/bnb/*.h5; do
    [ -e "$f" ] || continue
    echo "$f" >> "$OUT"
done

N=$(wc -l < "$OUT")
echo "manifest_full: $N tasks -> $OUT"
