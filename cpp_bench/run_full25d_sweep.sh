#!/bin/bash
# Full 2.5D Conflux-style sweep.
#   4 GPUs:  [Px=2, Py=2, Pz=1]   (pure 2D, c=1, no replication)
#   4 GPUs:  [Px=1, Py=1, Pz=4]   (degenerate, c=4, max replication)
#   8 GPUs:  [Px=2, Py=2, Pz=2]   (true 2.5D, c=2, c=floor(P^{1/3}))
#   8 GPUs:  [Px=1, Py=1, Pz=8]   (degenerate, c=8, max replication)
#
# Usage:
#   4 GPUs:  srun -p large     -N1 -n1 --gres=gpu:4 -t 02:00:00 ./run_full25d_sweep.sh 4
#   8 GPUs:  srun -p full-node -N1 -n1 --gres=gpu:8 -t 02:00:00 ./run_full25d_sweep.sh 8

set -e
cd /home/ftome_local/comparative-bench/NextLA.jl/cpp_bench
export PATH=/home/ftome_local/miniforge3/bin:$PATH
export LD_LIBRARY_PATH=/home/ftome_local/miniforge3/lib:${LD_LIBRARY_PATH}

NP=${1:-4}
SIZES=${SIZES:-"8000 16000 32000"}
if [ -n "${N_MAX:-}" ]; then
  SIZES="$SIZES $N_MAX"
fi

MPIRUN="mpirun --map-by :OVERSUBSCRIBE -np $NP"
HEAD () { printf "\n========================================================================\n  %s\n========================================================================\n" "$*"; }

# cuSOLVERMp baseline ------------------------------------------------------
HEAD "cuSOLVERMp baseline (NVIDIA, libcusolverMp v0.8) — $NP GPUs"
for N in $SIZES; do
  case $NP in
    4) PX=2; PY=2 ;;
    8) PX=4; PY=2 ;;
    *) PX=$NP; PY=1 ;;
  esac
  $MPIRUN ./cusolverMp_geqrf_bench $N 256 256 $PX $PY
  $MPIRUN ./cusolverMp_geqrf_bench $N 256 256 $PX $PY fp32
done

# Full 2.5D Path-s: pure 2D (c=1), degenerate (c=P), and true 2.5D (8 GPUs only)
HEAD "Full 2.5D Path-s — pure 2D [Px=$( [ "$NP" -eq 4 ] && echo 2 || echo 4 ), Py=2, Pz=1], c=1"
case $NP in
  4) PX=2; PY=2 ;;
  8) PX=4; PY=2 ;;
esac
for N in $SIZES; do
  # Default lookahead is ON (TeX §A1); use --no-la for ablation.
  $MPIRUN ./scqr3_full25d_bench --N=$N --passes=2 --px=$PX --py=$PY --pz=1 --no-la
  $MPIRUN ./scqr3_full25d_bench --N=$N --passes=2 --px=$PX --py=$PY --pz=1
  $MPIRUN ./scqr3_full25d_bench --N=$N --passes=3 --px=$PX --py=$PY --pz=1 --no-la
  $MPIRUN ./scqr3_full25d_bench --N=$N --passes=3 --px=$PX --py=$PY --pz=1
  $MPIRUN ./scqr3_full25d_bench --N=$N --passes=2 --px=$PX --py=$PY --pz=1 --matrix=fp64mp --no-la
  $MPIRUN ./scqr3_full25d_bench --N=$N --passes=2 --px=$PX --py=$PY --pz=1 --matrix=fp64mp_tf32 --no-la
  $MPIRUN ./scqr3_full25d_bench --N=$N --passes=2 --px=$PX --py=$PY --pz=1 --matrix=fp32full --no-la
done

if [ "$NP" -eq 8 ]; then
  HEAD "Full 2.5D Path-s — TRUE 2.5D [Px=2, Py=2, Pz=2], c=2 (the Conflux sweet spot for P=8)"
  for N in $SIZES; do
    $MPIRUN ./scqr3_full25d_bench --N=$N --passes=2 --px=2 --py=2 --pz=2 --no-la
    $MPIRUN ./scqr3_full25d_bench --N=$N --passes=2 --px=2 --py=2 --pz=2
    $MPIRUN ./scqr3_full25d_bench --N=$N --passes=3 --px=2 --py=2 --pz=2 --no-la
    $MPIRUN ./scqr3_full25d_bench --N=$N --passes=3 --px=2 --py=2 --pz=2
    $MPIRUN ./scqr3_full25d_bench --N=$N --passes=2 --px=2 --py=2 --pz=2 --matrix=fp64mp_tf32 --no-la
  done
fi

HEAD "Full 2.5D Path-s — degenerate [Px=1, Py=1, Pz=$NP], c=$NP (max replication)"
for N in $SIZES; do
  $MPIRUN ./scqr3_full25d_bench --N=$N --passes=2 --px=1 --py=1 --pz=$NP --no-la
  $MPIRUN ./scqr3_full25d_bench --N=$N --passes=2 --px=1 --py=1 --pz=$NP
  $MPIRUN ./scqr3_full25d_bench --N=$N --passes=3 --px=1 --py=1 --pz=$NP --no-la
  $MPIRUN ./scqr3_full25d_bench --N=$N --passes=3 --px=1 --py=1 --pz=$NP
  $MPIRUN ./scqr3_full25d_bench --N=$N --passes=2 --px=1 --py=1 --pz=$NP --matrix=fp64mp_tf32 --no-la
done

HEAD "DONE"
