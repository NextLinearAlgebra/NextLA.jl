```@meta
CurrentModule = NextLA
```

# Mixed-precision BLAS

The recursive GEMM, triangular solve/multiply and SYRK methods of `NextLA.MixedPrecision`,
in `src/mxp/blas`. Load them with `using NextLA.MixedPrecision`.

```@autodocs
Modules = [MixedPrecision]
Pages = [
    "recgemm_mixedprec.jl",
    "rectrxm_mixed.jl",
    "recsyrk_mixedprec.jl",
]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Usage examples

Each example stands on its own and ends with a check that prints `true`: the relative error of
the mixed-precision result against the same computation in `Float64`, with a tolerance that
matches the coarsest precision used.

### GEMM and SYRK updates into mixed-precision containers

`recgemm!` updates a `FullMixedPrec` and `recsyrk!` the stored triangle of a `SymmMixedPrec`, from
ordinary dense factors.

```jldoctest
using NextLA, NextLA.MixedPrecision, LinearAlgebra

n, k = 2048, 512
A = rand(n, k)
B = rand(k, n)
C = rand(n, n)

F = FullMixedPrec(C; precisions = DataType[Float32, Float64])
recgemm!(-1.0, A, B, 1.0, F)                       # F = C - A * B

S = SymmMixedPrec(C + C', 'L'; precisions = DataType[Float32, Float64])
recsyrk!(1.0, A, 0.5, S)                           # S = A * A' + (C + C') / 2

relerr(X, Y) = norm(X - Y) / norm(Y)
relerr(reconstruct_matrix(F), C - A * B) < 1e-5 &&
    relerr(reconstruct_matrix(S), A * A' + (C + C') / 2) < 1e-5

# output

true
```
