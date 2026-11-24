```@meta
CurrentModule = NextLA.MixedPrecision
```

# Mixed precision

The `NextLA.MixedPrecision` submodule stores a matrix as a **tree of blocks, each level in its own
floating-point type**, and runs recursive GEMM, triangular solve/multiply, SYRK, Cholesky and LU
directly on that tree. Off-diagonal blocks, which usually carry less of the matrix's norm, can be
stored in `Float16`, `BFloat16` or even 8-bit floats, while the diagonal leaves stay in `Float32`
or `Float64`.

The submodule is not re-exported by NextLA. Load it explicitly:

```julia
using NextLA
using NextLA.MixedPrecision
```

## The recursive splitting

All containers split an `n × n` matrix the same way: at `n ÷ 2` when `n` is a power of two, and
otherwise at the largest power of two below `n`. The diagonal blocks `A11` and `A22` are split
again; the off-diagonal blocks are stored as dense arrays.

```
        level 0                          after level 1
┌─────────┬─────────┐            ┌────┬────┬─────────┐
│         │         │            │ L  │ o1 │         │
│   A11   │   A12   │            ├────┼────┤   o0    │
│         │         │            │ o1 │ L  │         │
├─────────┼─────────┤    →       ├────┴────┼────┬────┤
│         │         │            │         │ L  │ o1 │
│   A21   │   A22   │            │   o0    ├────┼────┤
│         │         │            │         │ o1 │ L  │
└─────────┴─────────┘            └─────────┴────┴────┘

o0: precisions[1]   o1: precisions[2]
L:  leaves, precisions[end]
```

The `precisions` vector gives one type per level, from the top down:

- `precisions[1]` is used for the off-diagonal blocks at level 0,
- `precisions[2]` for the off-diagonal blocks at level 1, and so on,
- `precisions[end]` is used for the leaves (the diagonal blocks where recursion stops) and is the
  container's element type `T_Base`.

Recursion stops when one precision is left or a block is `1 × 1`. So
`precisions = DataType[Float16, Float32, Float64]` stores the two largest off-diagonal blocks in
`Float16`, the next four in `Float32`, and four `Float64` leaves of size about `n/4`.

### Rules for `precisions`

- Allowed types: `Float8_E5M2`, `BFloat16`, `Float16`, `Float32`, `Float64`.
- The leaf type `precisions[end]` must be `Float32` or `Float64`.
- The input matrix must be square, non-empty and finite.

## Containers

| Container                | Stores                                  | Typical use                          |
|:-------------------------|:----------------------------------------|:-------------------------------------|
| [`FullMixedPrec`](@ref)  | a dense square matrix (`A11`, `A12`, `A21`, `A22`) | GEMM output, LU                 |
| [`SymmMixedPrec`](@ref)  | one triangle of a symmetric matrix       | SYRK output, Cholesky               |
| [`TriMixedPrec`](@ref)   | one triangle of a triangular matrix      | triangular solve and multiply       |
| [`TiledTriMixedPrec`](@ref) | a triangular matrix as a flat grid of tiles | tile-based algorithms           |

All of them are `AbstractMatrix` subtypes, so `size`, `M[i, j]` and `eltype` work, and
[`reconstruct_matrix`](@ref) copies the container back into a dense matrix of type `T_Base`.
`transpose(M)` and `M'` return a lazy transposed wrapper.

Each example on this page stands on its own and ends with a check that prints `true`. They are run
as doctests every time the documentation is built.

```jldoctest
using NextLA, NextLA.MixedPrecision

A = rand(1024, 1024)
F = FullMixedPrec(A; precisions = DataType[Float16, Float64])  # off-diagonal Float16, leaves Float64

B = reconstruct_matrix(F)           # back to a dense Float64 matrix
isapprox(B, A; rtol = 1e-3)         # equal up to Float16 rounding

# output

true
```

## Scaling

Narrow types have little range: `floatmax(Float16)` is 65504. To store large values, every block
carries a `Float32` **scale factor** `s`. A block whose largest magnitude exceeds `floatmax` of its
storage type is stored as `A_block / s`, and reads return `stored * s`. Unscaled blocks have
`s = 1`.

- Scale factors are always `Float32`.
- A block that would need a scale beyond `floatmax(Float32)` throws an `ArgumentError`, rather than
  being stored as `Inf`.
- Updates (GEMM, SYRK, solves) that push a block past the range of its type raise its scale.
- Leaves of [`lu_recursive!`](@ref) and [`potrf_recursive!`](@ref) must be unscaled, since a single
  scale cannot represent both factors; a scaled leaf throws an `ArgumentError`.

The same block quantization is available on its own:

```jldoctest
using NextLA.MixedPrecision

A = 1e6 .* rand(1024, 1024)         # too large for Float16 (max 65504)

Q, s = quantize(A, Float16)         # Q = A / s in Float16, s is a Float32 scale
isapprox(dequantize(Q, s, Float64), A; rtol = 1e-3)

# output

true
```

## Choosing precisions automatically

[`adaptive_precisions`](@ref) picks the coarsest type per level that keeps the error below a target
`epsilon`, from the norms of the off-diagonal blocks at each level relative to the whole matrix:

```jldoctest
using NextLA.MixedPrecision, LinearAlgebra

A = rand(1024, 1024)
S = A * A' + 1024I

U = DataType[Float16, Float32, Float64]          # candidate types
p = adaptive_precisions(S, U, 16, 1e-8; uplo = 'L')
M = SymmMixedPrec(S, 'L'; precisions = p)        # p is ready to use as `precisions`
p[end]                                           # the leaves keep the input's precision

# output

Float64
```

- `U` lists the candidate types in any order.
- `n_min` (here `16`) stops splitting before any block would be smaller than this.
- `epsilon` is the target accuracy; a larger `epsilon` allows coarser types.
- `uplo` is `'L'` or `'U'` for a symmetric or triangular matrix (only that triangle is read), and
  `'F'` for a full one.

The leaf type is the coarsest type in `U` that is at least as fine as `eltype(A)` and has the range
of `Float32`, so the result is always a valid `precisions` argument when `U` contains `Float32` or
`Float64`.

## Routines

The routines below extend the dense NextLA functions of the same name with methods for the
containers. They walk the block tree, call the dense routines on the leaves and GEMM on the
off-diagonal blocks, and run independent halves on separate threads for large matrices.

| Routine                                                      | Container argument    | Computes                                      |
|:-------------------------------------------------------------|:----------------------|:----------------------------------------------|
| [`recgemm!(α, A, B, β, C)`](@ref recgemm!)                   | `C::FullMixedPrec`    | `C = αAB + βC`                                |
| [`recsyrk!(α, A, β, C)`](@ref recsyrk!)                      | `C::SymmMixedPrec`    | `C = αAAᵀ + βC` on `C.uplo`                   |
| [`potrf_recursive!(A)`](@ref potrf_recursive!)               | `A::SymmMixedPrec`    | Cholesky, in place                            |
| [`lu_recursive!(A)`](@ref lu_recursive!)                     | `A::FullMixedPrec`    | unpivoted LU, in place                        |
| [`unified_rectrxm!(side, uplo, trans, α, func, A, B)`](@ref unified_rectrxm!) | `A::FullMixedPrec`, `SymmMixedPrec`, `TriMixedPrec` | triangular solve (`'S'`) or multiply (`'M'`) |

### Example: Cholesky, then a triangular solve

```jldoctest
using NextLA, NextLA.MixedPrecision, LinearAlgebra

A = rand(1024, 1024)
S = A * A' + 1024I

M = SymmMixedPrec(S, 'L'; precisions = DataType[Float32, Float64])
potrf_recursive!(M)                    # M now holds the factor L

L = TriMixedPrec(M)                    # view the factor as triangular, no copy
b = rand(1024, 1)
x = unified_rectrxm!('L', 'L', 'N', 1.0, 'S', L, copy(b))   # solve L x = b
isapprox(reconstruct_matrix(L) * x, b; rtol = 1e-6)

# output

true
```

### Example: LU

```jldoctest
using NextLA, NextLA.MixedPrecision, LinearAlgebra

D = rand(1024, 1024) + 1024I           # diagonally dominant, so no pivoting is needed

F = FullMixedPrec(D; precisions = DataType[Float32, Float64])
lu_recursive!(F)
LU = reconstruct_matrix(F)             # L below the diagonal, U on and above
isapprox(UnitLowerTriangular(LU) * UpperTriangular(LU), D; rtol = 1e-5)

# output

true
```

## Pitfalls

- **No pivoting.** [`lu_recursive!`](@ref) on a `FullMixedPrec` does not pivot; use matrices that
  need none (e.g. diagonally dominant).
- **`uplo` must match.** For `SymmMixedPrec` and `TriMixedPrec`, `unified_rectrxm!` requires
  `uplo == A.uplo` and throws an `ArgumentError` otherwise, since only that triangle is stored.
- **Transposes.** To solve with `Aᵀ`, pass `trans = 'T'` rather than `transpose(A)`.
- **Recovering a Cholesky factor.** After `potrf_recursive!(M)`, use
  `reconstruct_matrix(TriMixedPrec(M))` for the dense factor; `reconstruct_matrix(M)` would mirror
  it into both triangles.
- **`TiledTriMixedPrec`** is a storage format with `precisions` of at most two types (off-diagonal
  tiles, diagonal tiles) and a `threshold` for the tile size; the recursive routines above do not
  take it.

The docstrings are on the [Mixed-precision core](@ref), [Mixed-precision BLAS](@ref) and
[Mixed-precision LAPACK](@ref) pages.
