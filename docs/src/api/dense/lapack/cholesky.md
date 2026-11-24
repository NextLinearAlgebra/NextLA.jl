```@meta
CurrentModule = NextLA
```

# Cholesky

Factorizations `A = L Lᵀ` or `A = Uᵀ U` of symmetric positive definite matrices.

| Routine                        | Algorithm                                                  |
|:-------------------------------|:-----------------------------------------------------------|
| [`potrf!`](@ref)               | single call to LAPACK / the vendor solver                  |
| [`potrf_recursive!`](@ref)     | recursive 2×2 blocking over `potrf!`, TRSM and SYRK        |
| [`potrf_batched!`](@ref)       | many independent matrices                                  |
| [`cholesky_blocked!`](@ref)    | blocked KernelAbstractions factorization                   |
| [`cholesky_lower!`](@ref)      | single-kernel KernelAbstractions factorization             |

The mixed-precision method of `potrf_recursive!` for `SymmMixedPrec` is under
[Mixed-precision LAPACK](@ref).

## Factorizations

```@autodocs
Modules = [NextLA]
Pages = [
    joinpath("cholesky", "potrf.jl"),
    joinpath("cholesky", "recpotrf.jl"),
    joinpath("cholesky", "potrf_batched.jl"),
    joinpath("cholesky", "potrf_blocked.jl"),
    joinpath("cholesky", "potrf_kernel.jl"),
    joinpath("cholesky", "potrf_register.jl"),
]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Usage examples

### Solve `S * X = B`

Factor `S = L * Lᵀ` with [`potrf!`](@ref), then solve with `L` and `Lᵀ` using two calls to
[`trsm!`](@ref).

```jldoctest
using NextLA, LinearAlgebra

n = 2048
M = rand(n, n)
S = M * M' + n * I
B = rand(n, 256)

F, info = potrf!('L', copy(S))                    # S = L * L'
X = copy(B)
trsm!('L', 'L', 'N', 'N', 1.0, F, X)              # L * Y = B
trsm!('L', 'L', 'T', 'N', 1.0, F, X)              # L' * X = Y
S * X ≈ B

# output

true
```

### Upper factor

With `uplo = 'U'`, [`potrf_recursive!`](@ref) writes `U` with `S = Uᵀ * U` and leaves the
strict lower triangle as it was.

```jldoctest
using NextLA, LinearAlgebra

n = 2048
M = rand(n, n)
S = M * M' + n * I

F = potrf_recursive!(copy(S), 512; uplo = 'U')    # block_size 512
U = UpperTriangular(F)
U' * U ≈ S && tril(F, -1) == tril(S, -1)

# output

true
```

### Matrices that are not positive definite

[`potrf!`](@ref) reports the order of the first leading minor that is not positive definite
through `info`; [`potrf_recursive!`](@ref) throws `PosDefException` with the same order.

```jldoctest
using NextLA, LinearAlgebra

n = 1024
M = rand(n, n)
S = M * M' + n * I
S[700, 700] = -1.0                                # leading minor of order 700 is indefinite

_, info = potrf!('L', copy(S))
err = try
    potrf_recursive!(copy(S), 256)
catch e
    e
end
info == 700 && err == PosDefException(700)

# output

true
```

### On the GPU

On a `CuArray`, [`potrf!`](@ref) calls cuSOLVER, [`potrf_recursive!`](@ref) stays on the
device, and [`cholesky_blocked!`](@ref) runs its own kernels (not run when the documentation is
built).

```julia
using NextLA, LinearAlgebra, CUDA

n = 4096
M = CUDA.rand(Float64, n, n)
S = M * M' + n * I
B = CUDA.rand(Float64, n, 256)

F, info = potrf!('L', copy(S))
F2 = potrf_recursive!(copy(S), 1024)
F3 = cholesky_blocked!(copy(S))

X = trsm!('L', 'L', 'N', 'N', 1.0, F, copy(B))
trsm!('L', 'L', 'T', 'N', 1.0, F, X)
info == 0 && tril(F2) ≈ tril(F) && tril(F3) ≈ tril(F) && S * X ≈ B    # true
```
