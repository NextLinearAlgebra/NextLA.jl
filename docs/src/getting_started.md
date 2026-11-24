```@meta
CurrentModule = NextLA
```

# Getting started

Install NextLA, then follow four short workflows, one for each part of the library: dense matrices,
mixed precision, tile low-rank and the GPU. Each one chains several routines and ends with a check
that prints `true`; the CPU ones are run as doctests every time the documentation is built.

## Installation

```julia
using Pkg
Pkg.add("NextLA")                                                # latest registered release
Pkg.add(url = "https://github.com/NextLinearAlgebra/NextLA.jl")  # development version (main)
```

NextLA needs Julia 1.10 or later and is tested on 1.10 and nightly. For GPU support, also add the
backend package you want: `CUDA`, `AMDGPU`, `oneAPI` or `Metal`. To check an installation, run the
test suite as described in [Running the tests](dev/testing.md).

In-place routines follow the BLAS/LAPACK argument conventions, including the character flags
`side`, `uplo`, `trans` and `diag`.

## Usage

### Dense matrices: solve a symmetric positive definite system

[`syrk!`](@ref) builds `S = A Aᵀ + n I`, [`potrf_recursive!`](@ref) factors it as `S = L Lᵀ`, two
calls to [`trsm!`](@ref) solve `S X = B`, and [`gemm!`](@ref) computes the residual.

```jldoctest
using NextLA, LinearAlgebra

n = 2048
A = rand(n, n)
B = rand(n, 256)

S = Matrix{Float64}(n * I, n, n)
syrk!('L', 'N', 1.0, A, 1.0, S)          # lower triangle of S = A Aᵀ + n I
potrf_recursive!(S)                      # Cholesky: L in the lower triangle of S

X = copy(B)
trsm!('L', 'L', 'N', 'N', 1.0, S, X)     # L Y = B
trsm!('L', 'L', 'T', 'N', 1.0, S, X)     # Lᵀ X = Y

R = copy(B)
gemm!(R, A * A' + n * I, X; alpha = -1.0, beta = 1.0)   # R = B - (A Aᵀ + n I) X
norm(R) / norm(B) < 1e-10

# output

true
```

For a general matrix, `lu!(NextLAMatrix{Float64}(A))` gives a pivoted LU that solves with `\`; for
least squares, the QR kernels `NextLA.geqrt!` and `NextLA.unmqr!` factor `A` and apply `Qᵀ`. The
[LU](api/dense/lapack/lu.md) and [QR](api/dense/lapack/qr.md) API pages show both.

### Mixed precision: the same solve with fewer bits

[`adaptive_precisions`](@ref NextLA.MixedPrecision.adaptive_precisions) picks a precision for each level of the block tree,
[`SymmMixedPrec`](@ref NextLA.MixedPrecision.SymmMixedPrec) stores `S` that way, [`potrf_recursive!`](@ref) factors it in place, and
[`unified_rectrxm!`](@ref) solves with the mixed-precision factor.

```jldoctest
using NextLA, NextLA.MixedPrecision, LinearAlgebra

n = 2048
A = rand(n, n)
S = A * A' + n * I                                 # symmetric positive definite
B = rand(n, 256)

p = adaptive_precisions(S, DataType[Float16, Float32, Float64], 256, 1e-6)
M = SymmMixedPrec(S, 'L'; precisions = p)          # off-diagonal blocks in lower precisions
potrf_recursive!(M)                                # S = L Lᵀ, L stored in M
L = TriMixedPrec(M)                                # the factor, without copying

X = copy(B)
unified_rectrxm!('L', 'L', 'N', 1.0, 'S', L, X)    # L Y = B
unified_rectrxm!('L', 'L', 'T', 1.0, 'S', L, X)    # Lᵀ X = Y
norm(X - S \ B) / norm(S \ B) < 1e-5               # close to the Float64 solve

# output

true
```

### Tile low-rank: compress, then multiply

[`TLRMatrix`](@ref NextLA.TileLowRank.TLRMatrix) stores the off-diagonal tiles as low-rank
factors, [`ranks`](@ref NextLA.TileLowRank.ranks) shows how much was kept, and `NextLA.gemm!`
multiplies with the compressed matrix.

```jldoctest
using NextLA, NextLA.TileLowRank, LinearAlgebra, Random
Random.seed!(1)                                    # compression is randomized

n = 2048
A = [1 / (1 + abs(i - j)) for i in 1:n, j in 1:n] + I   # low-rank off-diagonal tiles

T = TLRMatrix(A, 256; maxrank = 64, tol = 1e-8, rel = true)   # 8 × 8 tiles of 256
stored = sum(ranks(T)) * 2 * 256 + 8 * 256^2       # numbers kept: factors plus dense diagonal

C = zeros(n, n)
NextLA.gemm!(C, T, A; workspace = gemm_maximum_workspace_bytes(T, T))   # C = A * A
stored < n^2 / 2 && norm(C - A * A) / norm(A * A) < 1e-7

# output

true
```

### On a GPU

Build the matrices on the GPU and call the same routines; dense routines reach the vendor library
(here cuBLAS and cuSOLVER), and mixed-precision and tile low-rank containers keep their data on the
device. This
example is not run when the documentation is built, since the build machine has no GPU.

```julia
using NextLA, NextLA.MixedPrecision, NextLA.TileLowRank, CUDA, LinearAlgebra

n = 4096
A = CUDA.rand(Float64, n, n)
S = A * A' + n * I                                 # on the GPU
B = CUDA.rand(Float64, n, 256)

F = potrf_recursive!(copy(S))                      # Cholesky on the GPU
X = trsm!('L', 'L', 'N', 'N', 1.0, F, copy(B))     # L Y = B
trsm!('L', 'L', 'T', 'N', 1.0, F, X)               # Lᵀ X = Y

D = CUDA.rand(Float64, n, n) + n * I               # diagonally dominant
M = FullMixedPrec(D; precisions = DataType[Float16, Float32, Float64])
lu_recursive!(M)                                   # mixed-precision LU, blocks on the GPU
G = Array(reconstruct_matrix(M))

K = CuArray([1 / (1 + abs(i - j)) for i in 1:n, j in 1:n])           # low-rank off-diagonal tiles
T = TLRMatrix(K, 256; maxrank = 64, tol = 1e-8, rel = true)          # compressed on the GPU
C = CUDA.zeros(Float64, n, n)
NextLA.gemm!(C, T, T; workspace = gemm_maximum_workspace_bytes(T, T))   # C = K * K

S * X ≈ B && isapprox(UnitLowerTriangular(G) * UpperTriangular(G), Array(D); rtol = 1e-3) &&
    norm(C - K * K) / norm(K * K) < 1e-6                             # true
```

## Getting help

Questions, bug reports and feature requests go to the
[GitHub issues](https://github.com/NextLinearAlgebra/NextLA.jl/issues).

## Where to go next

- [Backends and GPUs](@ref): which routine runs on which backend, and with which element types.
- [Mixed precision](@ref): the full guide to `NextLA.MixedPrecision`.
- [Tile low-rank](@ref): the full guide to `NextLA.TileLowRank`.
- [Benchmarks](@ref): speed against the vendor libraries.
- [API reference](api/dense/core.md): every routine, organised like the source; each page ends with
  more worked examples.
