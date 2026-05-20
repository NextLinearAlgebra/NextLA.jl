#!/bin/bash
# 8-GPU sweep using 1D row distribution (c=8). cuSOLVERMp uses 4x2 grid.
set -e
cd /home/ftome_local/comparative-bench/NextLA.jl/cpp_bench
export PATH=/home/ftome_local/miniforge3/bin:$PATH
export LD_LIBRARY_PATH=/home/ftome_local/miniforge3/lib:${LD_LIBRARY_PATH}

NP=8
SIZES=${SIZES:-"8000 16000 32000 64000"}

MPIRUN="mpirun --map-by :OVERSUBSCRIBE -np $NP"
HEAD () { printf "\n========================================================================\n  %s\n========================================================================\n" "$*"; }

HEAD "cuSOLVERMp baseline — c=$NP (4x2 grid)"
for N in $SIZES; do
  $MPIRUN ./cusolverMp_geqrf_bench $N 512 512 4 2
done

HEAD "Path-s variants — 8-GPU c=8 (1D row distribution)"
for N in $SIZES; do
  printf "\n----- N=$N -----\n"
  $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=2
  $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=2 --la
  $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=3 --la
  $MPIRUN ./scqr3_2p5d_variants --N=$N --passes=3 --ir=1
done

HEAD "Path-h variants — 8-GPU Householder"
for N in $SIZES; do
  printf "\n----- N=$N -----\n"
  $MPIRUN ./householder_2p5d_bench --N=$N --no-la
  $MPIRUN ./householder_2p5d_bench --N=$N --la
done

HEAD "DONE"
