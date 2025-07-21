#!/bin/bash
#SBATCH --job-name=test.sh
#SBATCH --output=./test.sh.log-%j
#SBATCH --gres=gpu:volta:1
#SBATCH --partition=xeon-g6-volta
#SBATCH --cpus-per-task=20


source /etc/profile

module load julia/1.11.3
module load cuda/12.6

echo "running code"

julia --project=. test.jl "$1" "$2" "$3"

echo "This is outer precision $2 and inner precision $3 saved to $1"
echo "finished"