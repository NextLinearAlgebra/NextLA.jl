# Contributing

Contributions are welcome through pull requests on
[GitHub](https://github.com/NextLinearAlgebra/NextLA.jl).

## Repository layout

```
src/
├── NextLA.jl                module entry point, includes every feature
├── streams.jl               GPU streams and events (used by TLR compression)
├── dense/                   dense matrices in one precision
│   ├── core/                NextLAMatrix.jl (wrapper type routing LinearAlgebra calls to NextLA)
│   ├── blas/
│   │   ├── gemm/            GEMM: extended, batched, grouped, recursive
│   │   ├── trsm/            TRSM/TRMM kernels, unified_rectrxm!, batched TRSM
│   │   ├── syrk/            SYRK: dispatch, batched, portable kernel, recursive
│   │   └── axpy.jl, gerc.jl (BLAS level 1 and 2)
│   └── lapack/
│       ├── cholesky/        potrf!, recursive, blocked and batched Cholesky
│       ├── lu/              pivoted LU (lu_pivot.jl), recursive and tiled unpivoted LU
│       ├── qr/              Householder reflectors, tile QR kernels and their PLASMA
│       │                    helpers parfb, pamm, pemv (one per file)
│       └── rotations/       Givens rotations
├── mxp/                     submodule NextLA.MixedPrecision
│   ├── core/                mxp.jl (module definition), the mixed-precision containers,
│   │                        adaptive precision selection and quantization
│   ├── blas/                gemm/, trsm/, syrk/
│   └── lapack/              cholesky/, lu/
└── tlr/                     submodule NextLA.TileLowRank (tile low-rank matrices)
    ├── core/                tlr.jl (module definition), TLRMatrix, CompressedFTLRMatrix,
    │                        compression/ (adaptive randomized approximation)
    ├── blas/                gemm/ (TLR GEMM into dense or compressed results)
    └── lapack/              (empty for now)
ext/
├── NextLACUDAExt.jl     + ext/cuda/      cuBLAS / cuSOLVER methods, streams, batched SVD
├── NextLAAMDGPUExt.jl   + ext/amdgpu/    rocBLAS / rocSOLVER methods, streams, batched SVD
├── NextLAoneAPIExt.jl   + ext/oneapi/    oneMKL methods
└── NextLAMetalExt.jl    + ext/metal/     MPS methods
test/                    mirrors src/ (dense/, mxp/, tlr/): the test for src/<path>.jl is
                         test/<path>.jl; shared helpers in test/setup/
docs/                    this documentation (Documenter.jl)
```

How to run all or some of the tests, on the CPU or a GPU: [`test/README.md`](test/README.md).

Each of `dense/`, `mxp/` and `tlr/` has the same three folders: `core/` for the types, data
structures and module definition the rest builds on, `blas/` for BLAS-style routines and `lapack/`
for factorizations and LAPACK-style routines. Each routine-group folder has one entry file named
after the folder (`gemm/gemm.jl`, `trsm/trsm.jl`, `qr/qr.jl`, ...) that includes every other file
in it and nothing else, and `src/NextLA.jl` includes only those entry files. Exports stay in the
file that defines the symbol.

### Modules

| Module                  | Folder      | Depends on            | Loaded with                     |
|:------------------------|:------------|:----------------------|:--------------------------------|
| `NextLA`                | `src/dense` | nothing in the others | `using NextLA`                  |
| `NextLA.MixedPrecision` | `src/mxp`   | `NextLA`              | `using NextLA.MixedPrecision`   |
| `NextLA.TileLowRank`    | `src/tlr`   | `NextLA`              | `using NextLA.TileLowRank`      |

`MixedPrecision` and `TileLowRank` are submodules: they live inside `NextLA`, can use anything from it
through `..NextLA`, and are not re-exported by it. The dependency runs one way only. Code in
`src/dense` never refers to a submodule, so `NextLA` builds and works with the `include` line of a
submodule removed. TLR code lives in `src/tlr` and follows the same rule.
