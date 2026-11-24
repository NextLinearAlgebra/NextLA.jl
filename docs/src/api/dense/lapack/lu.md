```@meta
CurrentModule = NextLA
```

# LU

| Routine                                | Pivoting  | Notes                                            |
|:---------------------------------------|:----------|:-------------------------------------------------|
| `LinearAlgebra.lu!(::NextLAMatrix)`    | partial or complete | Julia implementation of LAPACK `getrf`  |
| [`getrf2!`](@ref)                      | partial   | recursive LAPACK `getrf2`                        |
| [`lu_recursive!`](@ref)                | none      | recursive, square matrices                       |
| [`lu_recursive_nopiv!`](@ref)          | none      | recursive, any shape                             |
| [`lu_base!`](@ref)                     | none      | single-workgroup kernel for one block            |
| [`tile_lu_factor!`](@ref)              | none      | tiled driver over `lu_base!`                     |

!!! warning
    The unpivoted routines require a matrix that needs no pivoting, e.g. a diagonally dominant
    one. A zero pivot throws `ZeroPivotException`; small pivots are not checked.

The mixed-precision method of `lu_recursive!` for `FullMixedPrec` is under
[Mixed-precision LAPACK](@ref).

## Pivoted LU (LAPACK style)

```@autodocs
Modules = [NextLA]
Pages = [joinpath("lu", "lu_pivot.jl")]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Recursive and tiled LU

```@autodocs
Modules = [NextLA]
Pages = [joinpath("lu", "reclu.jl"), joinpath("lu", "lu_base.jl"), joinpath("lu", "lu_tiled.jl")]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Usage examples

Each example stands on its own and ends with a check that prints `true`.

Pivoted LU of a `NextLAMatrix`, then a solve with several right-hand sides through the returned
`LinearAlgebra.LU` object:

```jldoctest
using NextLA, LinearAlgebra

A = rand(2048, 2048)
B = rand(2048, 16)

F = lu!(NextLAMatrix{Float64}(copy(A)))   # partial pivoting, computed by getrf2!
X = F \ B
A * X ≈ B

# output

true
```

[`getrf2!`](@ref) called directly returns LAPACK's pivot vector; `laswp` applies those row
interchanges to the right-hand side before the two triangular solves:

```jldoctest
using NextLA, LinearAlgebra

n = 2048
A = rand(n, n)
b = rand(n, 1)

F = copy(A)
ipiv = zeros(Int, n)
_, _, info = getrf2!(F, ipiv, 0)

x = copy(b)
laswp(x, 1, n, ipiv, 1)                  # x := P * b
x = UpperTriangular(F) \ (UnitLowerTriangular(F) \ x)
info == 0 && A * x ≈ b

# output

true
```

Complete pivoting with [`getc2!`](@ref), which permutes both rows and columns,
`A[p, q] = L * U`:

```jldoctest
using NextLA, LinearAlgebra

A = rand(1024, 1024)
F = getc2!(NextLAMatrix{Float64}(copy(A)), CompletePivoting())

p = LinearAlgebra.LU(F.factors, F.ipiv, 0).p   # row permutation from ipiv
q = LinearAlgebra.LU(F.factors, F.jpiv, 0).p   # column permutation from jpiv
UnitLowerTriangular(F.factors) * UpperTriangular(F.factors) ≈ A[p, q]

# output

true
```

Unpivoted recursive LU of a diagonally dominant matrix, with leaves of at most 512 rows, used to
solve a system:

```jldoctest
using NextLA, LinearAlgebra

n = 4096
A = rand(n, n) + n * I            # diagonally dominant: no pivoting needed
F = lu_recursive!(copy(A), 512)

b = rand(n)
x = UpperTriangular(F) \ (UnitLowerTriangular(F) \ b)
A * x ≈ b

# output

true
```

[`lu_recursive_nopiv!`](@ref) factors a tall panel, here 4096 by 1024; `L` is unit lower
trapezoidal and `U` square:

```jldoctest
using NextLA, LinearAlgebra

m, n = 4096, 1024
A = rand(m, n)
A[1:n, :] += m * I                # needs no pivoting

F = lu_recursive_nopiv!(copy(A), 256)
L = tril(F, -1) + Matrix(I, m, n)
U = triu(F[1:n, :])
L * U ≈ A

# output

true
```

On a GPU the same calls take device arrays. [`tile_lu_factor!`](@ref) runs on a GPU only, with
tiles of at most 32. This block is not run when the documentation is built:

```julia
using NextLA, CUDA, LinearAlgebra

n = 4096
A = rand(n, n) + n * I
dF = lu_recursive!(CuArray(A), 1024)       # the leaves go to cuSOLVER, unpivoted
F = Array(dF)
UnitLowerTriangular(F) * UpperTriangular(F) ≈ A

B = rand(2048, 2048)
dG, p, info = tile_lu_factor!(CuArray(B), 32)   # pivoting within each 32×32 tile
G = Array(dG)
info == 0 && UnitLowerTriangular(G) * UpperTriangular(G) ≈ B[p, :]
```
