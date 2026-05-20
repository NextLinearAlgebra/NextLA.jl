#!/bin/bash
# Build CUDA MPI benches. Requires nvcc, OpenMPI (mpicc), NCCL, cuSOLVER, cuBLAS.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

: "${CUDA_HOME:=/usr/local/cuda}"
: "${MPI_INCLUDE:=}"
if [[ -z "$MPI_INCLUDE" ]]; then
  if command -v mpicc &>/dev/null; then
    MPI_INCLUDE="$(mpicc --showme:compile 2>/dev/null | tr ' ' '\n' | grep '^-I' | head -1 | cut -c3-)"
  fi
fi
if [[ -z "$MPI_INCLUDE" || ! -f "$MPI_INCLUDE/mpi.h" ]]; then
  echo "Set MPI_INCLUDE to a directory containing mpi.h (e.g. export MPI_INCLUDE=\$CONDA_PREFIX/include)" >&2
  exit 1
fi

NVCC="$CUDA_HOME/bin/nvcc"
INCLUDES="-I$MPI_INCLUDE -I$ROOT"
LIBS="-L${CUDA_HOME}/lib64 -L${CONDA_PREFIX:-}/lib -lmpi ${NCCL_LIB:+-L$NCCL_LIB} -lnccl -lcusolver -lcublas -lcudart"

$NVCC -std=c++17 -O2 $INCLUDES $LIBS scqr3_full25d_bench.cu -o scqr3_full25d_bench
$NVCC -std=c++17 -O2 $INCLUDES $LIBS householder_2p5d_bench.cu -o householder_2p5d_bench
$NVCC -std=c++17 -O2 $INCLUDES $LIBS givens_2p5d_bench.cu -o givens_2p5d_bench
$NVCC -std=c++17 -O2 $INCLUDES $LIBS qdwh_2p5d_bench.cu -o qdwh_2p5d_bench

mpicxx -std=c++17 -O2 -I"$CUDA_HOME/include" $LIBS cusolverMp_geqrf_bench.cpp -o cusolverMp_geqrf_bench \
  -L"$CUDA_HOME/lib64" -Wl,-rpath,"$CUDA_HOME/lib64" -lcusolverMp || \
  { rm -f cusolverMp_geqrf_bench; echo "Note: cusolverMp_geqrf_bench not built (need libcusolverMp, e.g. under CONDA_PREFIX/lib)." >&2; }

echo "Built: scqr3_full25d_bench householder_2p5d_bench givens_2p5d_bench qdwh_2p5d_bench (and cusolverMp_geqrf_bench if libcusolverMp was found)"
echo "Tip: export LD_LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}\" before mpirun if libmpi is not found."
