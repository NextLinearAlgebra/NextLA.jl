#!/bin/bash
# Comprehensive 2.5D QR variant sweep on 4 H200 GPUs, larger matrices.
#
#   Variants compared:
#     - Householder (Path h): vanilla, +LA, +IR=1, +MP+LA
#     - sCQR3 (Path s, passes=3): vanilla, +LA, +IR=1, +MP, +MP+LA
#     - CQR2  (Path s, passes=2): vanilla, +LA, +IR=1
#     - cuSOLVERMp baseline (NVIDIA's distributed QR via libcusolverMp)
#
#   N ∈ {8000, 16000, 32000, 64000}
#   c=4 (4 GPUs)
#
#   Output goes to stdout. Designed to be invoked as
#     srun -p dev -N1 -n1 --gres=gpu:4 -t 02:00:00 ./run_all_large.sh

set -e
cd /home/ftome_local/comparative-bench/NextLA.jl/cpp_bench
export PATH=/home/ftome_local/miniforge3/bin:$PATH
export LD_LIBRARY_PATH=/home/ftome_local/miniforge3/lib:${LD_LIBRARY_PATH}

NP=${NP:-4}
SIZES=${SIZES:-"8000 16000 32000 64000"}

MPIRUN="mpirun --map-by :OVERSUBSCRIBE -np $NP"
HEAD () { printf "\n========================================================================\n  %s\n========================================================================\n" "$*"; }

# cuSOLVERMp baseline ------------------------------------------------------
HEAD "cuSOLVERMp baseline (NVIDIA, libcusolverMp v0.8) — c=$NP"
for N in $SIZES; do
  case $NP in
    4) PX=2; PY=2; MB=256 ;;
    8) PX=4; PY=2; MB=256 ;;
    *) PX=$NP; PY=1; MB=256 ;;
  esac
  $MPIRUN ./cusolverMp_geqrf_bench $N $MB $MB $PX $PY
done

# sCQR3 / CQR2 variants ----------------------------------------------------
HEAD "Path-s variants — sCQR3 / CQR2 with the schedule from qr_schur_xpartition.tex §A.3"
for N in $SIZES; do
  printf "\n----- N=$N -----\n"
  for PASSES in 3 2; do
    $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=$PASSES
    $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=$PASSES --la
  done
  $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=3 --ir=1
  $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=3 --mp
  $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=3 --mp --la
done

# Householder variants ----------------------------------------------------
HEAD "Path-h variants — Householder + WY (cuSolver geqrf+orgqr per panel)"
for N in $SIZES; do
  printf "\n----- N=$N -----\n"
  $MPIRUN ./householder_2p5d_bench --N=$N --no-la
  $MPIRUN ./householder_2p5d_bench --N=$N --la
  $MPIRUN ./householder_2p5d_bench --N=$N --no-la --ir=1
  $MPIRUN ./householder_2p5d_bench --N=$N --mp --la
done

HEAD "DONE"
