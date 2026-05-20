#!/bin/bash
# Full 5-variant sweep on 8 H200 GPUs (1D row distribution, c=8).
# cuSOLVERMp uses 4x2 process grid.
#
# Usage: srun -p full-node -N1 -n1 --gres=gpu:8 -t 03:00:00 ./run_all_5variants_8gpu.sh

set -e
cd /home/ftome_local/comparative-bench/NextLA.jl/cpp_bench
export PATH=/home/ftome_local/miniforge3/bin:$PATH
export LD_LIBRARY_PATH=/home/ftome_local/miniforge3/lib:${LD_LIBRARY_PATH}

NP=8
SIZES=${SIZES:-"8000 16000 32000"}
SIZES_QDWH=${SIZES_QDWH:-"8000 16000"}
SIZES_GIVENS=${SIZES_GIVENS:-"4000 8000"}

MPIRUN="mpirun --map-by :OVERSUBSCRIBE -np $NP"
HEAD () { printf "\n========================================================================\n  %s\n========================================================================\n" "$*"; }

HEAD "cuSOLVERMp baseline — c=$NP (4x2 grid)"
for N in $SIZES; do
  $MPIRUN ./cusolverMp_geqrf_bench $N 256 256 4 2
done

HEAD "Path-s variants — 8-GPU c=8"
for N in $SIZES; do
  printf "\n----- N=$N -----\n"
  $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=3
  $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=3 --la
  $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=2
  $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=2 --la
done

HEAD "Path-h variants — 8-GPU"
for N in $SIZES; do
  printf "\n----- N=$N -----\n"
  $MPIRUN ./householder_2p5d_bench --N=$N --no-la
  $MPIRUN ./householder_2p5d_bench --N=$N --la
done

HEAD "Path-g variants — 8-GPU"
for N in $SIZES_GIVENS; do
  printf "\n----- N=$N -----\n"
  $MPIRUN ./givens_2p5d_bench --N=$N --no-la
  $MPIRUN ./givens_2p5d_bench --N=$N --la
done

HEAD "Path-q variants — 8-GPU"
for N in $SIZES_QDWH; do
  printf "\n----- N=$N -----\n"
  $MPIRUN ./qdwh_2p5d_bench --N=$N --iters=6 --no-la
  $MPIRUN ./qdwh_2p5d_bench --N=$N --iters=6 --la
done

HEAD "DONE"
