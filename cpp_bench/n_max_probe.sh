#!/bin/bash
# Discover largest admissible N for Path (s) on a fixed grid: N divisible by Px*Pz and Py.
# Usage: ./n_max_probe.sh <binary> <Px> <Py> <Pz> <np> [low] [high]
# Example: ./n_max_probe.sh ./scqr3_full25d_bench 2 2 1 4 1000 120000

set -euo pipefail
BIN=${1:?binary}
PX=${2:?Px}
PY=${3:?Py}
PZ=${4:?Pz}
NP=${5:?np}
LOW=${6:-2048}
HIGH=${7:-200000}
EXTRA=( "${@:8}" )
MPIRUN="mpirun --map-by :OVERSUBSCRIBE -np $NP"

ok() {
  local N=$1
  $MPIRUN "$BIN" --N="$N" --px=$PX --py=$PY --pz=$PZ --passes=2 --no-la "${EXTRA[@]}" >/tmp/nmax_try.log 2>&1 \
    && (grep -q "tmed=" /tmp/nmax_try.log || grep -q "ours_ms=" /tmp/nmax_try.log)
  local st=$?
  if [[ $st -ne 0 ]]; then return 1; fi
  if grep -q "OOM_PROBE" /tmp/nmax_try.log 2>/dev/null; then return 1; fi
  return 0
}

lo=$LOW
hi=$HIGH
best=$LOW
while [[ $lo -le $hi ]]; do
  mid=$(( (lo + hi) / 2 ))
  step=$(( Px * Pz ))
  [[ $step -lt 1 ]] && step=1
  mid=$(( (mid / step) * step ))
  [[ $mid -lt $lo ]] && mid=$lo
  if ok "$mid"; then best=$mid; lo=$((mid + step)); else hi=$((mid - 1)); fi
done
echo "N_max_probe grid=[$PX,$PY,$PZ] best_N=$best (heuristic binary; verify memory on target GPU)"
