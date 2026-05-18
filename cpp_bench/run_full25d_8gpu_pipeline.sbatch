#!/bin/bash
#SBATCH --job-name=full25d_8gpu
#SBATCH --partition=full-node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=02:30:00
#SBATCH --output=/home/ftome_local/comparative-bench/NextLA.jl/logs/full25d_8gpu_%j.log
#
# Full-node pipeline: smoke [Px,Py,Pz]=[2,2,2] then the same sweep as
# run_full25d_sweep.sh 8 (cuSOLVERMp 4x2 + pure 2D + true 2.5D + degenerate).
#
# Queue behind another user, e.g.:
#   sbatch --dependency=afterany:JOBID run_full25d_8gpu_pipeline.sbatch

set -euo pipefail
cd /home/ftome_local/comparative-bench/NextLA.jl/cpp_bench
export PATH=/home/ftome_local/miniforge3/bin:${PATH}
export LD_LIBRARY_PATH=/home/ftome_local/miniforge3/lib:${LD_LIBRARY_PATH}

MPI="mpirun --map-by :OVERSUBSCRIBE -np 8"

echo "========================================================================"
echo " Smoke: 8 GPUs grid [2,2,2]  N=4000  (true 2.5D, c=2)"
echo "========================================================================"
${MPI} ./scqr3_full25d_bench --N=4000 --passes=2 --px=2 --py=2 --pz=2
${MPI} ./scqr3_full25d_bench --N=4000 --passes=3 --px=2 --py=2 --pz=2
${MPI} ./scqr3_full25d_bench --N=4000 --passes=2 --px=2 --py=2 --pz=2 --la
${MPI} ./scqr3_full25d_bench --N=4000 --passes=3 --px=2 --py=2 --pz=2 --la

echo "========================================================================"
echo " Full sweep: run_full25d_sweep.sh 8"
echo "========================================================================"
bash ./run_full25d_sweep.sh 8

echo "DONE pipeline"
