```@meta
CurrentModule = NextLA
```

# Matrix multiply (GEMM)

`C := α op(A) op(B) + β C` in several forms:

- [`gemm!`](@ref) — the dense entry point, in output-first (`gemm!(C, A, B; alpha, beta)`) or BLAS
  (`gemm!(transA, transB, alpha, A, B, beta, C)`) argument order.
- [`gemmEx!`](@ref) — storage and accumulation types chosen separately.
- [`gemm_batched!`](@ref) and [`gemmEx_batched!`](@ref) — many independent products in one call,
  from a vector of matrices or a 3-D strided array.
- Grouped GEMM — batches of products with different shapes.
- [`recgemm!`](@ref) — recursive GEMM; its mixed-precision method is documented under
  [Mixed-precision BLAS](@ref).
- [`GEMM_ADD!`](@ref) / [`GEMM_SUB!`](@ref) — the tiled KernelAbstractions kernel.

## Dense and extended GEMM

```@autodocs
Modules = [NextLA]
Pages = [joinpath("gemm", "gemmEx.jl"), joinpath("gemm", "recgemm.jl")]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Compute types and alignment

```@autodocs
Modules = [NextLA]
Pages = [joinpath("gemm", "gemm_types.jl"), joinpath("gemm", "gemm_precision.jl")]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Batched GEMM

```@autodocs
Modules = [NextLA]
Pages = [joinpath("gemm", "gemm_batched.jl")]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Grouped GEMM

```@autodocs
Modules = [NextLA]
Pages = [joinpath("gemm", "gemm_grouped.jl")]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Tiled kernel

```@autodocs
Modules = [NextLA]
Pages = [joinpath("gemm", "matmul.jl")]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Usage examples

Both argument orders of [`gemm!`](@ref) make the same update; the BLAS order takes the transposes
as flags, the output-first order as `transpose`/`adjoint` wrappers.

```jldoctest
using NextLA, LinearAlgebra

A = rand(2048, 1024)
B = rand(2048, 1024)
C1 = rand(1024, 1024)
C0 = copy(C1)
C2 = copy(C1)

gemm!('T', 'N', 2.0, A, B, 1.0, C1)                    # BLAS order: C1 = 2 Aᵀ B + C1
gemm!(C2, transpose(A), B; alpha = 2.0, beta = 1.0)    # output first, same update
C1 ≈ C2 ≈ 2.0 * transpose(A) * B + C0

# output

true
```

[`gemmEx!`](@ref) multiplies `Float16` matrices with a `Float32` accumulation and rounds once on
store; [`recgemm!`](@ref) picks that accumulation by itself.

```jldoctest
using NextLA, LinearAlgebra

A = rand(Float16, 2048, 2048)
B = rand(Float16, 2048, 2048)
C1 = zeros(Float16, 2048, 2048)
C2 = zeros(Float16, 2048, 2048)

gemmEx!('N', 'N', 1.0, A, B, 0.0, C1; compute_type = Float32)   # accumulate in Float32
recgemm!(1.0, A, B, 0.0, C2)                                    # chooses Float32 by itself
C1 ≈ C2 && isapprox(C1, Float64.(A) * Float64.(B); rtol = 1e-3)

# output

true
```

[`gemm_batched!`](@ref) takes a 3-D strided batch, where an operand with batch count 1 is shared
by every member, or a vector of matrices.

```jldoctest
using NextLA, LinearAlgebra

A = rand(1024, 1024, 8)
B = rand(1024, 1024, 1)        # batch count 1: one B shared by every member
C = zeros(1024, 1024, 8)
gemm_batched!('N', 'N', 1.0, A, B, 0.0, C)

As = [rand(1024, 1024) for _ in 1:4]
Bs = [rand(1024, 1024) for _ in 1:4]
Cs = [zeros(1024, 1024) for _ in 1:4]
gemm_batched!('N', 'T', 1.0, As, Bs, 0.0, Cs)

all(C[:, :, i] ≈ A[:, :, i] * B[:, :, 1] for i in 1:8) &&
    all(Cs[i] ≈ As[i] * transpose(Bs[i]) for i in 1:4)

# output

true
```

Grouped GEMM runs products of different shapes in one call, described by
[`GroupedGemmTask`](@ref)s (a loop on the CPU, one grouped call on CUDA).

```jldoctest
using NextLA, LinearAlgebra

shapes = [(1024, 2048, 512), (2048, 1024, 1024), (4096, 512, 1024)]   # (m, n, k) per member
tasks = [NextLA.GroupedGemmTask('N', 'N', 1.0, rand(m, k), rand(k, n), 0.0, zeros(m, n))
         for (m, n, k) in shapes]

NextLA.precision_gemm_grouped!(tasks, NextLA.GEMMCompute{Float64}())
all(t.C ≈ t.A * t.B for t in tasks)

# output

true
```

On the GPU the same calls reach cuBLAS: GEMMEx for mixed precision, one strided-batched call for a
3-D batch, and the TF32 tensor cores through [`TF32`](@ref) (CUDA only). This block is not run when
the documentation is built.

```julia
using NextLA, CUDA

# Float16 storage, Float32 accumulation: one cuBLAS GEMMEx call
A = CUDA.rand(Float16, 4096, 4096)
B = CUDA.rand(Float16, 4096, 4096)
C = CUDA.zeros(Float16, 4096, 4096)
gemmEx!('N', 'N', 1f0, A, B, 0f0, C; compute_type = Float32)

# a 3-D strided batch: one cuBLAS strided-batched call
Ab = CUDA.rand(Float32, 1024, 1024, 16)
Bb = CUDA.rand(Float32, 1024, 1024, 16)
Cb = CUDA.zeros(Float32, 1024, 1024, 16)
gemm_batched!('N', 'N', 1f0, Ab, Bb, 0f0, Cb)

# Float32 operands on the TF32 tensor cores (CUDA only)
X = CUDA.rand(Float32, 4096, 4096)
Y = CUDA.rand(Float32, 4096, 4096)
Z = CUDA.zeros(Float32, 4096, 4096)
NextLA.precision_gemm!('N', 'N', 1f0, X, Y, 0f0, Z, TF32())

isapprox(C, Float32.(A) * Float32.(B); rtol = 1e-3) &&
    Cb[:, :, 16] ≈ Ab[:, :, 16] * Bb[:, :, 16] &&
    isapprox(Z, X * Y; rtol = 1e-2)
```
