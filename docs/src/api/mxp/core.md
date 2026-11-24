```@meta
CurrentModule = NextLA
```

# Mixed-precision core

The `NextLA.MixedPrecision` module, its containers, precision selection and quantization, in
`src/mxp/core`. Load it with `using NextLA.MixedPrecision`; see the [Mixed precision](@ref) guide
for concepts and examples.

```@docs
MixedPrecision
```

## Containers

```@autodocs
Modules = [MixedPrecision]
Pages = ["fullmixedprec.jl", "symmmixedprec.jl", "trimixedprec.jl", "tiledtrimixedprec.jl"]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Precision selection and quantization

```@autodocs
Modules = [MixedPrecision]
Pages = ["adaptive.jl", "quantize.jl"]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Usage examples

Each example stands on its own and ends with a check that prints `true`: the relative error of
the mixed-precision result against the same computation in `Float64`, with a tolerance that
matches the coarsest precision used.

### Building the containers from one `precisions` list

`precisions[1]` is used for the top-level off-diagonal blocks, `precisions[2]` for the next
level, and `precisions[end]` for the diagonal leaves, which is also the element type.

```jldoctest
using NextLA, NextLA.MixedPrecision, LinearAlgebra

A = rand(2048, 2048)
precisions = DataType[Float16, Float32, Float64]   # level 0, level 1, leaves

F = FullMixedPrec(A; precisions)                   # all of A
S = SymmMixedPrec(A + A', 'L'; precisions)         # lower triangle of a symmetric matrix
T = TriMixedPrec(A, 'U'; precisions)               # upper triangle of A

relerr(X, Y) = norm(X - Y) / norm(Y)
eltype(F.A21) == Float16 && eltype(F.A11.A21) == Float32 && eltype(F) == Float64 &&
    relerr(reconstruct_matrix(F), A) < 1e-3 &&
    relerr(reconstruct_matrix(S), A + A') < 1e-3 &&
    relerr(reconstruct_matrix(T), triu(A)) < 1e-3

# output

true
```

### Storing large values in `Float16` with `quantize`

`quantize` divides a block by a `Float32` scale when its values exceed the range of the storage
type, and `dequantize` multiplies it back.

```jldoctest
using NextLA, NextLA.MixedPrecision, LinearAlgebra

A = 1e6 .* randn(2048, 2048)                       # far past floatmax(Float16) = 65504
Q, s = quantize(A, Float16)                        # A ≈ s * Q, with Q in Float16
B = dequantize(Q, s, Float64)

eltype(Q) == Float16 && s > 1 && norm(B - A) / norm(A) < 1e-3

# output

true
```
