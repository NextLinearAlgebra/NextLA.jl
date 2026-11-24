```@meta
CurrentModule = NextLA
```

# Dense core

Types and helpers in `src/dense/core` that the rest of NextLA builds on.

## `NextLAMatrix`

`NextLAMatrix{T}(data::Matrix{T})` wraps a dense matrix so that `LinearAlgebra.lu!` on it runs
NextLA's own pivoted LU ([`getrf2!`](@ref)) and `LinearAlgebra.BLAS.axpy!` runs NextLA's `axpy!`.
The other `LinearAlgebra.BLAS` calls on it (`gemv!`, `gemm!`, `trmm!`, `trsm!`, `syrk!`, `herk!`,
`syr2k!`, `her2k!`) pass the wrapped matrix to the system BLAS.

```jldoctest
using NextLA, LinearAlgebra

A = rand(1024, 1024)
F = lu!(NextLAMatrix{Float64}(copy(A)))   # NextLA's getrf, with partial pivoting
F.L * F.U ≈ A[F.p, :]

# output

true
```

```@autodocs
Modules = [NextLA]
Pages = ["NextLAMatrix.jl"]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Machine parameters

```@autodocs
Modules = [NextLA]
Pages = [joinpath("src", "NextLA.jl")]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Usage examples

The example stands on its own and ends with a check that prints `true`.

### Solving a system through `NextLAMatrix`

`lu!` on a `NextLAMatrix` returns an ordinary `LinearAlgebra.LU`, so `\` solves with it.

```jldoctest
using NextLA, LinearAlgebra

n = 2048
A = rand(n, n)
B = rand(n, 256)

F = lu!(NextLAMatrix{Float64}(copy(A)))   # NextLA's getrf, with partial pivoting
X = F \ B
A * X ≈ B

# output

true
```
