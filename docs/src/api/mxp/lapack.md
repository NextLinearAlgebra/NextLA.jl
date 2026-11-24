```@meta
CurrentModule = NextLA
```

# Mixed-precision LAPACK

The recursive Cholesky and LU methods of `NextLA.MixedPrecision`, in `src/mxp/lapack`. Load them
with `using NextLA.MixedPrecision`.

```@autodocs
Modules = [MixedPrecision]
Pages = [
    "recpotrf_mixedprec.jl",
    "reclu_mixedprec.jl",
]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Usage examples

Each example stands on its own and ends with a check that prints `true`: the relative error of
the mixed-precision result against the same computation in `Float64`, with a tolerance that
matches the coarsest precision used.

### Cholesky with precisions from `adaptive_precisions`, then a solve

`potrf_recursive!` factors a `SymmMixedPrec` in place, and `TriMixedPrec(M)` hands the factor to
`unified_rectrxm!` for the two triangular solves.

```jldoctest
using NextLA, NextLA.MixedPrecision, LinearAlgebra

n = 2048
A = rand(n, n)
S = A * A' + n * I                                 # symmetric positive definite

p = adaptive_precisions(S, DataType[Float16, Float32, Float64], 256, 1e-6)
M = SymmMixedPrec(S, 'L'; precisions = p)
potrf_recursive!(M)                                # S = L * L', L stored in M

L = TriMixedPrec(M)                                # the factor, without copying
B = rand(n, 256)
X = copy(B)
unified_rectrxm!('L', 'L', 'N', 1.0, 'S', L, X)    # L * Y = B
unified_rectrxm!('L', 'L', 'T', 1.0, 'S', L, X)    # L' * X = Y

relerr(X, Y) = norm(X - Y) / norm(Y)
relerr(reconstruct_matrix(L), cholesky(S).L) < 1e-5 && relerr(X, S \ B) < 1e-5

# output

true
```

### LU without pivoting, then a solve

`lu_recursive!` stores `L` (unit diagonal) and `U` in the same `FullMixedPrec`, and
`unified_rectrxm!` reads either triangle of it.

```jldoctest
using NextLA, NextLA.MixedPrecision, LinearAlgebra

n = 2048
A = rand(n, n) + n * I                             # diagonally dominant: no pivoting needed
M = FullMixedPrec(A; precisions = DataType[Float32, Float64])
lu_recursive!(M)                                   # A = L * U, both stored in M

B = rand(n, 256)
X = copy(B)
unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', M, X)   # L * Y = B, unit diagonal
unified_rectrxm!('L', 'U', 'N', 'N', 1.0, 'S', M, X)   # U * X = Y

norm(X - A \ B) / norm(A \ B) < 1e-5

# output

true
```

### On the GPU

`FullMixedPrec` built from a `CuArray` stores its blocks as `CuArray`s, and `lu_recursive!` and
`reconstruct_matrix` work on it unchanged (this block is not run when the documentation is built).

```julia
using CUDA, NextLA, NextLA.MixedPrecision, LinearAlgebra

n = 4096
A = CUDA.rand(Float64, n, n) + n * I               # diagonally dominant, on the GPU
M = FullMixedPrec(A; precisions = DataType[Float16, Float32, Float64])
lu_recursive!(M)

F = Array(reconstruct_matrix(M))                   # reconstruct_matrix returns a CuMatrix
isapprox(UnitLowerTriangular(F) * UpperTriangular(F), Array(A); rtol = 1e-3)   # true
```
