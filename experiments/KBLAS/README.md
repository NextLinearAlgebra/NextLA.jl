# KBLAS experiments

This folder contains the KBLAS-side launcher for the fixed-rank strong-scaling
benchmark. It runs both KBLAS TLR output modes:

- `lld`: TLR × TLR → dense
- `lll`: TLR × TLR → TLR

It intentionally uses a standalone CUDA
executable instead of a Julia FFI layer. The KBLAS API contains C++ overloads,
and the standalone executable is the simplest reproducible boundary on an HPC
system.

## Supercomputer setup

Load a CUDA toolkit and a compiler compatible with it, then clone KBLAS:

```bash
git clone https://github.com/ecrc/kblas-gpu.git
export KBLAS_ROOT=$PWD/kblas-gpu
```

KBLAS TLR builds require MAGMA in this configuration. Load or build MAGMA and
set:

```bash
export CUDA_ROOT=/path/to/cuda
export MAGMA_ROOT=/path/to/magma
```

Before building KBLAS, check `kblas-gpu/make.inc` and set the GPU architecture
(`_CUDA_ARCH_`) to the target machine. The important options are:

```make
_SUPPORT_TLR_=TRUE
_SUPPORT_SVD_=TRUE
_USE_MAGMA_=TRUE
```

Then build the library:

```bash
cd "$KBLAS_ROOT"
make -j \
    _CUDA_ROOT_="$CUDA_ROOT" \
    _CUDA_ARCH_=80 \
    _MAGMA_ROOT_="$MAGMA_ROOT" \
    _USE_MAGMA_=TRUE \
    _SUPPORT_TLR_=TRUE \
    _SUPPORT_SVD_=TRUE
```

The build should produce:

```text
$KBLAS_ROOT/lib/libkblas-gpu.a
```

The launcher also needs the CUDA runtime libraries, cuBLAS, OpenBLAS, and
MAGMA available at link/runtime time.

## Run

`benchmark_tlr_dense.cu` is retained as the standalone KBLAS comparison source.
There is deliberately no repository shell launcher; compile it directly for
the local KBLAS, MAGMA, CUDA, and BLAS installation.

This benchmark covers KBLAS `Float32` and `Float64` fixed-rank TLR GEMM.
KBLAS does not currently provide a linked variable-rank implementation for the
`CompressedFTLRMatrix` distributions, nor FP16/BF16/TF32 TLR entry points.
Those cases remain separate from the KBLAS comparison.
