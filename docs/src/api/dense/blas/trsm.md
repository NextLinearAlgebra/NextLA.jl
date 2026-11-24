```@meta
CurrentModule = NextLA
```

# Triangular solve and multiply

- [`trsm!`](@ref) — triangular solve in BLAS argument order, `op(A) X = α B` or `X op(A) = α B`.
- [`trmm!`](@ref) — triangular multiply in BLAS argument order.
- [`unified_rectrxm!`](@ref) — the recursive engine behind both, selected with `func = 'S'`
  (solve) or `'M'` (multiply). Its mixed-precision method is under [Mixed-precision BLAS](@ref).
- [`trsm_batched!`](@ref) — many independent solves.
- `LeftLowerTRSM!`, `RightUpperTRMM!`, ... — the KernelAbstractions base-case kernels.

## Recursive TRSM / TRMM

```@autodocs
Modules = [NextLA]
Pages = [joinpath("trsm", "rectrxm.jl")]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## BLAS-style entry points and kernels

```@autodocs
Modules = [NextLA]
Pages = [joinpath("trsm", "trsm_kernels.jl"), joinpath("trsm", "trmm.jl")]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Batched TRSM

```@autodocs
Modules = [NextLA]
Pages = [joinpath("trsm", "trsm_batched.jl")]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Usage examples

### Right-side solve with the conjugate transpose

`side = 'R'` with `transa = 'C'` solves `X * U' = α B` for a complex upper triangle `U`.

```jldoctest
using NextLA, LinearAlgebra

n = 2048
A = rand(ComplexF64, n, n) + n * I
B = rand(ComplexF64, 512, n)

X = trsm!('R', 'U', 'C', 'N', 0.5, A, copy(B))    # X * U' = 0.5 * B
X * UpperTriangular(A)' ≈ 0.5 * B

# output

true
```

### Unit diagonal

With `diag = 'U'` the stored diagonal is never read, so here it holds `NaN`; small off-diagonal
entries keep the unit triangle well conditioned.

```jldoctest
using NextLA, LinearAlgebra

n = 2048
A = rand(n, n) / n
A[diagind(A)] .= NaN
B = rand(n, 256)

X = trsm!('L', 'L', 'T', 'U', 1.0, A, copy(B))    # transpose(L) * X = B, L unit lower
transpose(UnitLowerTriangular(A)) * X ≈ B

# output

true
```

### Multiply into a separate output

The eight-argument [`trmm!`](@ref) writes `alpha * op(A) * B` into `C` and leaves `B` unchanged.

```jldoctest
using NextLA, LinearAlgebra

n = 2048
A = rand(n, n)
B = rand(n, 256)
C = similar(B)

trmm!('L', 'U', 'N', 'N', 1.0, A, B, C)           # C = U * B
C ≈ UpperTriangular(A) * B

# output

true
```

### Batched solve over a vector of matrices

[`trsm_batched!`](@ref) also takes vectors of matrices, and an optional `alpha` last.

```jldoctest
using NextLA, LinearAlgebra

n = 1024
A = [rand(n, n) + n * I for _ in 1:4]
B = [rand(n, 128) for _ in 1:4]

X = trsm_batched!('L', 'U', 'N', 'N', A, deepcopy(B), 2.0)
all(UpperTriangular(A[k]) * X[k] ≈ 2 * B[k] for k in 1:4)

# output

true
```

### On the GPU

The same calls on `CuArray`s reach cuBLAS `trsm!` and `trmm!` (not run when the documentation is built).

```julia
using NextLA, LinearAlgebra, CUDA

n = 4096
A = CUDA.rand(Float32, n, n) + n * I
B = CUDA.rand(Float32, n, 512)

X = trsm!('L', 'L', 'N', 'N', 1.0f0, A, copy(B))  # solve L * X = B
C = trmm!('L', 'L', 'N', 'N', 1.0f0, A, copy(X))  # multiply back: C = L * X
C ≈ B                                             # true
```
