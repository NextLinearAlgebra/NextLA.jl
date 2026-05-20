#!/bin/bash
# Derived-grid sweep: TeX schedule from auto M (per-rank device memory × frac / σ)
# in the benches, plus cuSOLVERMp baselines (FP64 + FP32).
#
# Usage:
#   ./run_derived_sweep.sh [NP]
#
# Env: SIZES, N_MAX, CONDA_PREFIX (see run_full25d_sweep.sh). Optional: NEXTLA_FASTMEM_FRAC

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export PATH="${CONDA_PREFIX:-/usr}/bin:$PATH"

NP=${1:-4}
SIZES=${SIZES:-"4000 8000"}
if [ -n "${N_MAX:-}" ]; then
  SIZES="$SIZES $N_MAX"
fi

MPIRUN="mpirun --map-by :OVERSUBSCRIBE -np $NP"
HEAD () { printf "\n========================================================================\n  %s\n========================================================================\n" "$*"; }

HEAD "cuSOLVERMp FP64 + FP32 baselines — $NP ranks"
for N in $SIZES; do
  case $NP in
    4) PX=2; PY=2 ;;
    8) PX=4; PY=2 ;;
    *) PX=$NP; PY=1 ;;
  esac
  $MPIRUN ./cusolverMp_geqrf_bench "$N" 256 256 "$PX" "$PY"
  $MPIRUN ./cusolverMp_geqrf_bench "$N" 256 256 "$PX" "$PY" fp32
done

HEAD "Path (s) full25d — TeX-derived grid (omit --px/--py/--pz; M and b from device + §A3b in bench)"
for N in $SIZES; do
  printf "  N=%s  (derived grid)\n" "$N"
  $MPIRUN ./scqr3_full25d_bench --N="$N" --passes=3 --strict-b --no-la
  $MPIRUN ./scqr3_full25d_bench --N="$N" --passes=3 --strict-b
  $MPIRUN ./scqr3_full25d_bench --N="$N" --passes=3 --strict-b --matrix=fp64mp --no-la
  $MPIRUN ./scqr3_full25d_bench --N="$N" --passes=3 --strict-b --matrix=fp64mp_tf32 --no-la
  $MPIRUN ./scqr3_full25d_bench --N="$N" --passes=3 --strict-b --matrix=fp32full --no-la
done

HEAD "Path (s) explicit P=4 grids (ablation / regression)"
for N in $SIZES; do
  $MPIRUN ./scqr3_full25d_bench --N="$N" --passes=2 --px=2 --py=2 --pz=1 --no-la
  $MPIRUN ./scqr3_full25d_bench --N="$N" --passes=2 --px=2 --py=2 --pz=1
  $MPIRUN ./scqr3_full25d_bench --N="$N" --passes=3 --px=1 --py=1 --pz=4 --no-la
  $MPIRUN ./scqr3_full25d_bench --N="$N" --passes=3 --px=1 --py=1 --pz=4
  $MPIRUN ./scqr3_full25d_bench --N="$N" --passes=2 --px=2 --py=2 --pz=1 --matrix=fp64mp_tf32 --no-la
  $MPIRUN ./scqr3_full25d_bench --N="$N" --passes=2 --px=2 --py=2 --pz=1 --matrix=fp32full --no-la
done

HEAD "Postprocess hint: pipe paired medians to aggregate_speedups.py"
echo "  Example:  (vendor_ms ours_ms per line on stdin)"
echo "DONE run_derived_sweep"
