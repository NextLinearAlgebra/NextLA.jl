#!/usr/bin/env bash
# Run libcusolverMp Geqrf at fixed (N, MB, Px, Py) and append vendor table lines:
#   N  P  vendor_fp64_ms  vendor_fp32_ms
# Point benches at this file with:  export NEXTLA_VENDOR_METRICS_TABLE=/path/to/this.txt
#
# Usage: ./capture_vendor_table.sh [output.txt]
# Env: MPIRUN (default mpirun -np $P)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
OUT="${1:-$ROOT/vendor_metrics_table.txt}"
BENCH="$ROOT/cusolverMp_geqrf_bench"
if [[ ! -x "$BENCH" ]]; then
  echo "error: $BENCH not found (run ./build.sh)" >&2
  exit 1
fi

extract_med() {
  local d="$1"
  grep 'VENDOR_TABLE_ROW' | grep "dtype=$d" | tail -1 | sed -n 's/.*median_ms=\([0-9.eE+-]*\).*/\1/p'
}

write_row() {
  local N="$1" MB="$2" PX="$3" PY="$4"
  local P=$((PX * PY))
  local MR="${MPIRUN:-mpirun -np $P}"
  set +e
  local out64 out32 rc64 rc32
  out64=$($MR "$BENCH" "$N" "$MB" "$MB" "$PX" "$PY" 2>&1)
  rc64=$?
  out32=$($MR "$BENCH" "$N" "$MB" "$MB" "$PX" "$PY" fp32 2>&1)
  rc32=$?
  set -e
  if [[ $rc64 -ne 0 ]]; then echo "warning: fp64 failed N=$N P=$P rc=$rc64" >&2; echo "$out64" | tail -3 >&2; fi
  if [[ $rc32 -ne 0 ]]; then echo "warning: fp32 failed N=$N P=$P rc=$rc32" >&2; echo "$out32" | tail -3 >&2; fi
  local med64 med32
  med64=$(echo "$out64" | extract_med fp64)
  med32=$(echo "$out32" | extract_med fp32)
  if [[ -n "$med64" && -n "$med32" ]]; then
    echo "$N $P $med64 $med32" >>"$OUT"
    echo "ok N=$N P=$P fp64=$med64 fp32=$med32" >&2
  else
    echo "skip N=$N P=$P (fp64='$med64' fp32='$med32')" >&2
  fi
}

{
  echo "# N P vendor_fp64_ms vendor_fp32_ms  (cusolverMp_geqrf_bench median_ms)"
  echo "# Use: export NEXTLA_VENDOR_METRICS_TABLE=$OUT"
} >"$OUT"

write_row 2048 256 2 2
write_row 1024 256 2 2

echo "Wrote $OUT" >&2
