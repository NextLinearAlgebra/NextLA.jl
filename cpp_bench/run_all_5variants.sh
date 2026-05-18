#!/bin/bash
# Full 5-variant sweep on 4 H200 GPUs:
#   Path s sCQR3 / CQR2  (with passes=3, passes=2, LA, MP, IR combinations)
#   Path h Householder   (with LA, IR, MP+LA)
#   Path g Givens        (with LA, IR)
#   Path q QDWH          (with LA, iters=6)
#   cuSOLVERMp baseline
#
# Usage:
#   srun -p large -N1 -n1 --gres=gpu:4 -t 03:00:00 ./run_all_5variants.sh

set -e
cd /home/ftome_local/comparative-bench/NextLA.jl/cpp_bench
export PATH=/home/ftome_local/miniforge3/bin:$PATH
export LD_LIBRARY_PATH=/home/ftome_local/miniforge3/lib:${LD_LIBRARY_PATH}

NP=${NP:-4}
SIZES=${SIZES:-"8000 16000 32000"}
SIZES_QDWH=${SIZES_QDWH:-"8000 16000"}   # QDWH does 6 inner QRs per call; cap at 16K
SIZES_GIVENS=${SIZES_GIVENS:-"4000 8000"}   # Givens panel is sequential in (j,i); cap at 8K

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

# Path-s variants ----------------------------------------------------------
HEAD "Path-s variants — sCQR3 / CQR2"
for N in $SIZES; do
  printf "\n----- N=$N -----\n"
  $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=3
  $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=3 --la
  $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=2
  $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=2 --la
  $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=3 --ir=1
  $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=3 --mp --la
done

# Path-h variants ----------------------------------------------------------
HEAD "Path-h variants — Householder + WY"
for N in $SIZES; do
  printf "\n----- N=$N -----\n"
  $MPIRUN ./householder_2p5d_bench --N=$N
  $MPIRUN ./householder_2p5d_bench --N=$N --la
  $MPIRUN ./householder_2p5d_bench --N=$N --ir=1
  $MPIRUN ./householder_2p5d_bench --N=$N --mp --la
done

# Path-g variants — Givens (slower; cap at SIZES_GIVENS) -------------------
HEAD "Path-g variants — Givens"
for N in $SIZES_GIVENS; do
  printf "\n----- N=$N -----\n"
  $MPIRUN ./givens_2p5d_bench --N=$N
  $MPIRUN ./givens_2p5d_bench --N=$N --la
done

# Path-q variants — QDWH polar (cap at SIZES_QDWH) -------------------------
HEAD "Path-q variants — QDWH polar"
for N in $SIZES_QDWH; do
  printf "\n----- N=$N -----\n"
  $MPIRUN ./qdwh_2p5d_bench --N=$N --iters=6
  $MPIRUN ./qdwh_2p5d_bench --N=$N --iters=6 --la
done

HEAD "DONE"
