```@meta
CurrentModule = NextLA
```

# Tile low-rank

Many large matrices, such as covariance matrices and discretized integral operators, have
off-diagonal blocks of low numerical rank. A **tile low-rank (TLR)** matrix splits the matrix into
tiles and stores each such tile as two thin factors `U * V'`, which saves memory and makes
products cheaper. `NextLA.TileLowRank` provides TLR matrices, their compression from dense
matrices, and matrix multiplication with them. Load it with `using NextLA.TileLowRank`; it is not
re-exported by `NextLA`.

Each example on this page stands on its own and ends with a check that prints `true`. They are run
as doctests every time the documentation is built.

## Two containers

| Container | Diagonal tiles | Off-diagonal tiles | Use it when |
|:--|:--|:--|:--|
| [`TLRMatrix`](@ref NextLA.TileLowRank.TLRMatrix) | dense | `U * V'` | the diagonal tiles are full rank (the usual case) |
| [`CompressedFTLRMatrix`](@ref NextLA.TileLowRank.CompressedFTLRMatrix) | `U * V'` | `U * V'` | every tile, diagonal included, is low rank |

Both store the factors of all tiles packed into one array, and both are built by compressing a
dense matrix.

## Compressing a matrix

```jldoctest
using NextLA, NextLA.TileLowRank, LinearAlgebra, Random
Random.seed!(1)                             # compression is randomized

n = 2048
A = [1 / (1 + abs(i - j)) for i in 1:n, j in 1:n] + I

C = TLRMatrix(A, 256; maxrank = 64, tol = 1e-8, rel = true)
B = uncompress!(similar(A), C)              # back to a dense matrix
norm(B - A) / norm(A) < 1e-7

# output

true
```

- **`tile_size`** (here `256`) is the side of a square tile, or a tuple `(bm, bn)`. A size that does
  not divide `n` leaves a smaller last row and column of tiles.
- **`maxrank`** caps the rank of every tile.
- **`tol`** is the error allowed per tile in the Frobenius norm: absolute, or relative to the tile's
  norm with `rel = true`. Each tile gets the smallest rank that meets it, up to `maxrank`. With the
  default `tol = 0` nothing may be dropped, so tiles are kept at rank `maxrank`.
- Compression uses a randomized range finder and draws from Julia's global random number
  generator; seed it for reproducible ranks.

## Looking inside

```jldoctest
using NextLA, NextLA.TileLowRank, LinearAlgebra, Random
Random.seed!(1)

n = 2048
A = [1 / (1 + abs(i - j)) for i in 1:n, j in 1:n] + I
C = TLRMatrix(A, 256; maxrank = 64, tol = 1e-8, rel = true)

grid_size(C)                    # (8, 8) tiles
r = ranks(C)                    # rank of each tile, row by row; 0 for the dense diagonal
U, V = get_factors(C, 1, 2)     # the factors of tile (1, 2): U * V' approximates it
stored = sum(r) * 2 * 256 + 8 * 256^2      # numbers stored: factors plus dense diagonal
size(U, 2) == r[2] && stored < n^2 / 2     # about a fifth of the dense matrix

# output

true
```

[`ranks`](@ref NextLA.TileLowRank.ranks) gives the rank of each tile as a vector, row by row, and
[`maxrank`](@ref NextLA.TileLowRank.maxrank) the largest of them (not the `maxrank` cap given to
the constructor). [`get_factors`](@ref NextLA.TileLowRank.get_factors) returns the factors of one
tile and [`dense_diag`](@ref NextLA.TileLowRank.dense_diag) the dense diagonal tiles.

## Multiplying

`NextLA.gemm!(C, A, B; workspace)` computes `C = A * B` into a dense `C` when `A`, `B` or both are
TLR matrices; the other operand may be dense. It takes `alpha`, `beta`, `transA` and `transB` like
the dense [`gemm!`](@ref). `workspace` is a byte count;
[`gemm_maximum_workspace_bytes`](@ref NextLA.TileLowRank.gemm_maximum_workspace_bytes) is enough
for the fastest schedule and
[`gemm_minimum_workspace_bytes`](@ref NextLA.TileLowRank.gemm_minimum_workspace_bytes) the least
that works.

```jldoctest
using NextLA, NextLA.TileLowRank, LinearAlgebra, Random
Random.seed!(1)

n = 2048
A = [1 / (1 + abs(i - j)) for i in 1:n, j in 1:n] + I
T = TLRMatrix(A, 256; maxrank = 64, tol = 1e-8, rel = true)

C = zeros(n, n)
NextLA.gemm!(C, T, T; workspace = gemm_maximum_workspace_bytes(T, T))   # TLR × TLR
D = zeros(n, n)
NextLA.gemm!(D, T, A; workspace = gemm_maximum_workspace_bytes(T, T))   # TLR × dense
norm(C - A * A) / norm(A * A) < 1e-7 && norm(D - A * A) / norm(A * A) < 1e-7

# output

true
```

`NextLA.gemm(A, B; maxrank, tol, rel)` multiplies two `CompressedFTLRMatrix` operands and returns
the product compressed, as another `CompressedFTLRMatrix`, without forming it densely.

## Limitations

- TLR GEMM takes `Float64`, `Float32`, `Float16` and `BFloat16` operands; the low-precision types
  accumulate in `Float32`. The examples here use `Float64`.
- Compression can keep tiles at a higher rank than needed; check the result with `ranks`. Keep
  `maxrank` below the tile size: equal to it, the factors come out wrong.
- GPU arrays work too: compressing a `CuArray` keeps the factors on the device, and `NextLA.gemm!`
  multiplies there (checked with CUDA).
- There are no TLR factorizations (Cholesky, LU) yet.

The docstrings are on the [Tile low-rank core](@ref) and [Tile low-rank BLAS](@ref) pages.
