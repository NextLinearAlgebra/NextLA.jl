#!/bin/bash
# Small-N smoke: divisibility, layout sanity, §A3b window, quick orthogonality check.
# Requires built binaries in this directory.
#
# Optional: export NEXTLA_VENDOR_FP64_MS / NEXTLA_VENDOR_FP32_MS so METRICS lines
# include numeric vendor medians (five-column reporting without Python fusion).
#
# Optional: export NEXTLA_VENDOR_METRICS_TABLE to a file built by
#   ./capture_vendor_table.sh
# (defaults to ./vendor_metrics_table.txt if that file exists).
#
# Optional: SKIP_TF32_SMOKE=1 skips --matrix=fp64mp_tf32 cases (toolkit without
# CUBLAS_COMPUTE_32F_FAST_TF32 would otherwise MPI_ABORT).

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
MPIRUN="mpirun --map-by :OVERSUBSCRIBE -np 4"
LOG="${SMOKE_LOG:-$HERE/smoke_metrics.log}"
VENDOR_TABLE="${NEXTLA_VENDOR_METRICS_TABLE:-$HERE/vendor_metrics_table.txt}"
if [[ -f "$VENDOR_TABLE" ]]; then
  export NEXTLA_VENDOR_METRICS_TABLE="$VENDOR_TABLE"
fi
rm -f "$LOG"

echo "== scqr3_full25d tri-mode + TF32 + derived + smoke grid (4 ranks) =="
$MPIRUN ./scqr3_full25d_bench --N=2048 --px=2 --py=2 --pz=1 --passes=2 --strict-b --no-la | tee -a "$LOG"
$MPIRUN ./scqr3_full25d_bench --N=2048 --px=2 --py=2 --pz=1 --passes=2 --strict-b --matrix=fp64mp --no-la | tee -a "$LOG"
if [[ -z "${SKIP_TF32_SMOKE:-}" ]]; then
  $MPIRUN ./scqr3_full25d_bench --N=2048 --px=2 --py=2 --pz=1 --passes=2 --strict-b --matrix=fp64mp_tf32 --no-la | tee -a "$LOG"
else
  echo "== SKIP_TF32_SMOKE: skipping scqr3 fp64mp_tf32 ==" | tee -a "$LOG"
fi
$MPIRUN ./scqr3_full25d_bench --N=2048 --px=2 --py=2 --pz=1 --passes=2 --strict-b --matrix=fp32full --no-la | tee -a "$LOG"
$MPIRUN ./scqr3_full25d_bench --N=2048 --passes=2 --strict-b --no-la | tee -a "$LOG"
$MPIRUN ./scqr3_full25d_bench --N=2048 --smoke --passes=1 --strict-b --no-la | tee -a "$LOG"

echo "== householder path h tri-mode (1D c=P) =="
$MPIRUN ./householder_2p5d_bench --N=2048 --strict-b --matrix=fp64 | tee -a "$LOG"
$MPIRUN ./householder_2p5d_bench --N=2048 --strict-b --matrix=fp64mp | tee -a "$LOG"
if [[ -z "${SKIP_TF32_SMOKE:-}" ]]; then
  $MPIRUN ./householder_2p5d_bench --N=2048 --strict-b --matrix=fp64mp_tf32 | tee -a "$LOG"
else
  echo "== SKIP_TF32_SMOKE: skipping householder fp64mp_tf32 ==" | tee -a "$LOG"
fi
$MPIRUN ./householder_2p5d_bench --N=2048 --strict-b --matrix=fp32full | tee -a "$LOG"

echo "== householder block-cyclic (P=4 = 2×2×1, b|N) =="
$MPIRUN ./householder_2p5d_bench --N=2048 --b=512 --layout=blockcyclic --px=2 --py=2 --pz=1 --strict-b --matrix=fp64 | tee -a "$LOG"

echo "== givens path g tri-mode =="
$MPIRUN ./givens_2p5d_bench --N=2048 --strict-b --matrix=fp64 | tee -a "$LOG"
$MPIRUN ./givens_2p5d_bench --N=2048 --strict-b --matrix=fp64mp | tee -a "$LOG"
if [[ -z "${SKIP_TF32_SMOKE:-}" ]]; then
  $MPIRUN ./givens_2p5d_bench --N=2048 --strict-b --matrix=fp64mp_tf32 | tee -a "$LOG"
else
  echo "== SKIP_TF32_SMOKE: skipping givens fp64mp_tf32 ==" | tee -a "$LOG"
fi
$MPIRUN ./givens_2p5d_bench --N=2048 --strict-b --matrix=fp32full | tee -a "$LOG"

echo "== givens block-cyclic =="
$MPIRUN ./givens_2p5d_bench --N=2048 --b=512 --layout=blockcyclic --px=2 --py=2 --pz=1 --strict-b --matrix=fp64 | tee -a "$LOG"

echo "== qdwh path q (fp64 / fp64mp / fp64mp_tf32 / fp32full) =="
$MPIRUN ./qdwh_2p5d_bench --N=1024 --iters=2 --strict-b --matrix=fp64 | tee -a "$LOG"
$MPIRUN ./qdwh_2p5d_bench --N=1024 --iters=2 --strict-b --matrix=fp64mp | tee -a "$LOG"
if [[ -z "${SKIP_TF32_SMOKE:-}" ]]; then
  $MPIRUN ./qdwh_2p5d_bench --N=1024 --iters=2 --strict-b --matrix=fp64mp_tf32 | tee -a "$LOG"
else
  echo "== SKIP_TF32_SMOKE: skipping qdwh fp64mp_tf32 ==" | tee -a "$LOG"
fi
$MPIRUN ./qdwh_2p5d_bench --N=1024 --iters=2 --strict-b --matrix=fp32full | tee -a "$LOG"

echo "== qdwh block-cyclic fp64 (N=2048, b=512, 2×2×1) =="
$MPIRUN ./qdwh_2p5d_bench --N=2048 --b=512 --iters=2 --strict-b --matrix=fp64 --layout=blockcyclic --px=2 --py=2 --pz=1 | tee -a "$LOG"

echo "METRICS log written to $LOG"
echo "SMOKE OK"
