```@meta
CurrentModule = NextLA
```

# Backends and GPUs

NextLA writes its kernels once with
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and picks the backend from
the array it is given (`KernelAbstractions.get_backend(A)`). Vendor-specific code lives in
[package extensions](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions))
under `ext/`, which Julia loads automatically when the matching GPU package is loaded:

| Extension         | Loaded by      | Vendor libraries used          |
|:------------------|:---------------|:-------------------------------|
| `NextLACUDAExt`   | `using CUDA`   | cuBLAS, cuSOLVER               |
| `NextLAAMDGPUExt` | `using AMDGPU` | rocBLAS, rocSOLVER             |
| `NextLAoneAPIExt` | `using oneAPI` | oneMKL                         |
| `NextLAMetalExt`  | `using Metal`  | Metal Performance Shaders      |

Without any GPU package, every routine runs on the CPU, calling `LinearAlgebra.BLAS`/`LAPACK`
where they cover the element type and portable Julia kernels otherwise.

## Usage

```julia
using NextLA, CUDA          # or AMDGPU, oneAPI, Metal

using LinearAlgebra
A = rand(Float32, 100, 100)
S = CuArray(A * A' + 100I)   # move the matrix to the GPU once

potrf_recursive!(S)          # runs on the GPU
```

The call is the same as on the CPU. Keep all operands on the same device: NextLA does not move data
between host and device for you.

## What runs where

The table lists the main entry points. "Portable" means the generic KernelAbstractions or recursive
Julia implementation; "vendor" means the call is forwarded to the vendor library for the supported
element types (other types fall back to the portable path where one exists).

| Routine                                  | CPU         | CUDA     | AMDGPU   | oneAPI   | Metal    |
|:-----------------------------------------|:------------|:---------|:---------|:---------|:---------|
| [`gemm!`](@ref), [`recgemm!`](@ref)      | BLAS        | `mul!`   | `mul!`   | `mul!`   | `mul!`   |
| [`gemmEx!`](@ref) (mixed types)          | portable    | vendor   | vendor   | —        | —        |
| [`gemm_batched!`](@ref)                  | loop        | vendor   | vendor   | vendor   | vendor   |
| [`trsm!`](@ref), [`unified_rectrxm!`](@ref) | portable | vendor + portable | vendor + portable | portable | portable |
| [`trsm_batched!`](@ref)                  | loop        | vendor   | vendor   | vendor   | vendor   |
| [`syrk!`](@ref)                          | BLAS        | vendor   | vendor   | vendor   | vendor   |
| [`recsyrk!`](@ref), [`SYRK_KERNEL!`](@ref) | portable  | portable | portable | portable | portable |
| [`potrf!`](@ref)                         | LAPACK      | vendor   | vendor   | —        | vendor   |
| [`potrf_batched!`](@ref)                 | loop        | vendor   | vendor   | vendor   | vendor   |
| [`potrf_recursive!`](@ref)               | recursive   | recursive | recursive | needs `potrf!` | recursive |
| [`lu_recursive!`](@ref) (leaf)           | `lu!(NoPivot())` | cuSOLVER `getrf` | rocSOLVER `getrf_npvt` | — | — |
| [`cholesky_blocked!`](@ref)              | —           | portable | portable | portable | portable |
| [`syrk_batched!`](@ref)                  | loop        | loop     | vendor   | vendor (strided), loop (pointer) | via [`gemm_batched!`](@ref) |
| [`lu_base!`](@ref)                       | portable    | portable | portable | portable | portable |
| [`tile_lu_factor!`](@ref)                | —           | portable | portable | portable | portable |
| QR tile kernels (`geqrt!`, `tsqrt!`, `ttqrt!`, `unmqr!`, `tsmqr!`, `ttmqr!`, …) | portable | portable | portable | portable | portable |

Notes:

- [`gemmEx!`](@ref) with differing storage and compute types is a CUDA and AMDGPU feature; the CPU
  accepts any combination and accumulates in `compute_type`. oneAPI and Metal throw.
- [`trsm!`](@ref) reaches cuBLAS/rocBLAS `trsm!` for `Float32`, `Float64`, `ComplexF32` and
  `ComplexF64` dense matrices; `Float16`, views and the other backends take the portable recursion.
- [`potrf!`](@ref) has no generic device fallback, so on a backend without a wrapper (oneAPI) it
  throws instead of silently copying to the host. [`potrf_recursive!`](@ref) calls it at its leaves.
- `Float16` has no vendor SYRK, so recursive Cholesky in half precision uses [`recsyrk!`](@ref).
- [`cholesky_blocked!`](@ref) is GPU-only: on the CPU it throws and points to [`potrf!`](@ref).
  [`tile_lu_factor!`](@ref) refuses the CPU because its kernels give wrong results there.
- cuBLAS has no batched SYRK, so [`syrk_batched!`](@ref) loops single calls on CUDA; Metal has no SYRK
  at all, and its single-matrix [`syrk!`](@ref) can overwrite the other triangle.
- The AMDGPU fast path of [`unified_rectrxm!`](@ref) compiles and is selected for `ROCMatrix`
  arguments, but has not been run on AMD hardware.
- The QR tile kernels are not exported (call them as `NextLA.geqrt!` and so on). Their CUDA runs
  pass the test suite on Julia 1.10.

## Element types

| Type                     | Notes                                                                   |
|:-------------------------|:------------------------------------------------------------------------|
| `Float64`, `Float32`     | Supported by every routine below.                                       |
| `ComplexF64`, `ComplexF32` | Supported by every routine below; `cholesky_lower!` gives wrong results. |
| `Float16`                | Portable kernels; GEMM accumulates in `Float32`. Not in the CPU batched routines or `getrf2!`. |
| `BFloat16`               | From [BFloat16s.jl](https://github.com/JuliaMath/BFloat16s.jl). As `Float16`, and no Cholesky (`potrf!`, `potrf_recursive!`). |
| `Float8_E5M2`            | Storage type for mixed-precision off-diagonal blocks.                   |

Measured on the CPU: each routine runs on 32×32 inputs and its result is compared with a
`Float64`/`ComplexF64` reference. ✓ correct result, ✗ wrong result, — no method for that type,
*GPU only* / *refused* throws by design. On GPUs the vendor libraries decide which types the vendor
paths take; other types use the portable code.

| Routine | `Float64` | `Float32` | `Float16` | `BFloat16` | `ComplexF64` | `ComplexF32` |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| `gemm!` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `gemmEx!` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `recgemm!` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `gemm_batched!` | ✓ | ✓ | — | — | ✓ | ✓ |
| `trsm!` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `trmm!` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `unified_rectrxm!` (solve) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `unified_rectrxm!` (multiply) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `trsm_batched!` | ✓ | ✓ | — | — | ✓ | ✓ |
| `syrk!` | ✓ | ✓ | — | — | ✓ | ✓ |
| `recsyrk!` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SYRK_KERNEL!` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `syrk_batched!` | ✓ | ✓ | — | — | ✓ | ✓ |
| `potrf!` | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| `potrf_recursive!` | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| `cholesky_blocked!` | GPU only | GPU only | GPU only | GPU only | GPU only | GPU only |
| `cholesky_lower!` | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| `potrf_batched!` | ✓ | ✓ | — | — | ✓ | ✓ |
| `getrf2!` (pivoted LU) | ✓ | ✓ | — | — | ✓ | ✓ |
| `lu_recursive!` (no pivoting) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `lu_recursive_nopiv!` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `lu_base!` (pivoted, n ≤ 32) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `geqrt!` + `unmqr!` (QR of a tile) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `tsqrt!` + `tsmqr!` (triangle on square) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ttqrt!` + `ttmqr!` (triangle on triangle) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `geru!` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `gerc!` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `axpy!` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `lartg` (Givens) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## GEMM compute modes

For reduced-precision inputs, [`gemmEx!`](@ref) and its batched variants take a `compute_type`
that sets the accumulation precision. [`default_compute_type`](@ref) picks one from the storage
types of `A`, `B` and `C`, and [`GEMM_COMPUTE_TYPES`](@ref) documents the supported behaviour per
backend. The precision-policy entry points such as [`precision_gemm_batched!`](@ref) take a compute
*mode* instead, where [`TF32`](@ref) requests TensorFloat-32 computation for `Float32` operands.

Some vendor GEMMs need aligned leading dimensions; [`is_gemm_aligned`](@ref),
[`gemm_alignment_quantum`](@ref) and [`aligned_leading_dimension`](@ref) help allocate arrays that
qualify.
