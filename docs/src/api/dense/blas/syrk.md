```@meta
CurrentModule = NextLA
```

# Symmetric rank-k update (SYRK)

`C := α op(A) op(A)ᵀ + β C` on one triangle of `C`.

- [`syrk!`](@ref) — BLAS on the CPU, vendor SYRK on GPUs.
- [`syrk_batched!`](@ref) — many independent updates.
- [`SYRK_KERNEL!`](@ref) — a portable tiled KernelAbstractions SYRK for backends or element types
  no vendor library covers (e.g. `Float16`).
- [`recsyrk!`](@ref) — recursive SYRK; its mixed-precision method is under
  [Mixed-precision BLAS](@ref).

```@autodocs
Modules = [NextLA]
Pages = [
    joinpath("syrk", "syrk_dispatch.jl"),
    joinpath("syrk", "syrk_batched.jl"),
    joinpath("syrk", "syrk_kernel.jl"),
    joinpath("syrk", "recsyrk.jl"),
]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Usage examples

With `trans = 'T'` and `beta = 1`, [`syrk!`](@ref) adds `AᵀA` into the chosen triangle of an
existing `C` and leaves the other triangle exactly as it was.

```jldoctest
using NextLA, LinearAlgebra

A = rand(4096, 2048)
C = rand(2048, 2048)
C0 = copy(C)

syrk!('U', 'T', 1.0, A, 1.0, C)          # upper triangle: C += Aᵀ A
UpperTriangular(C) ≈ UpperTriangular(transpose(A) * A + C0) && tril(C, -1) == tril(C0, -1)

# output

true
```

[`recsyrk!`](@ref) gives the same triangle as [`syrk!`](@ref), and `Symmetric` turns that triangle
into the full product without copying.

```jldoctest
using NextLA, LinearAlgebra

A = rand(2048, 1024)
C1 = zeros(2048, 2048)
C2 = zeros(2048, 2048)

syrk!('L', 'N', 1.0, A, 0.0, C1)
recsyrk!(1.0, A, 0.0, C2)                 # same update, blocked recursively
LowerTriangular(C1) ≈ LowerTriangular(C2) && Symmetric(C1, :L) ≈ A * A'

# output

true
```

[`recsyrk!`](@ref) also takes `Float16` storage with a `Float32` result, accumulating in `Float32`
where BLAS has no half-precision SYRK.

```jldoctest
using NextLA, LinearAlgebra

A = rand(Float16, 2048, 1024)
C = zeros(Float32, 2048, 2048)

recsyrk!(1.0, A, 0.0, C)                 # lower triangle
isapprox(LowerTriangular(C), LowerTriangular(Float64.(A) * Float64.(A)'); rtol = 1e-4)

# output

true
```

On the GPU, [`syrk!`](@ref) calls cuBLAS SYRK, and [`recsyrk!`](@ref) covers `Float16`, which
cuBLAS SYRK does not. This block is not run when the documentation is built.

```julia
using NextLA, CUDA, LinearAlgebra

A = CUDA.rand(Float32, 4096, 2048)
C = CUDA.zeros(Float32, 4096, 4096)
syrk!('L', 'N', 1f0, A, 0f0, C)                 # cuBLAS SYRK

# cuBLAS has no Float16 SYRK; recsyrk! accumulates in Float32 through GEMMEx
A16 = CUDA.rand(Float16, 4096, 2048)
C32 = CUDA.zeros(Float32, 4096, 4096)
recsyrk!(1f0, A16, 0f0, C32)

isapprox(tril(C), tril(A * A'); rtol = 1e-4) &&
    isapprox(tril(C32), tril(Float32.(A16) * Float32.(A16)'); rtol = 1e-4)
```
