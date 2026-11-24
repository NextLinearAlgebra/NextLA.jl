# NextLA TODO

What NextLA.jl already does and what is still to build, grouped like BLAS and LAPACK.

✅ done · 🟡 partly done · 🔄 in progress on a branch · ⬜ to do

## Summary

| Area | Done | ✅ | 🟡 | 🔄 | ⬜ |
|:--|:-:|:-:|:-:|:-:|:-:|
| BLAS level 3 (matrix–matrix) | 4 of 7 | 4 |  |  | 3 |
| BLAS level 2 (matrix–vector) | 2 of 6 | 2 |  |  | 4 |
| BLAS level 1 (vector) | 1 of 3 | 1 | 1 |  | 1 |
| Cholesky (symmetric / Hermitian positive definite) | 1 of 6 | 1 |  |  | 5 |
| LU (general matrices) | 3 of 8 | 3 | 2 |  | 3 |
| QR and LQ | 4 of 10 | 4 | 1 | 1 | 4 |
| Least squares and triangular | 1 of 4 | 1 |  |  | 3 |
| Eigenvalues | 0 of 4 |  |  | 1 | 3 |
| Singular values | 0 of 3 |  |  | 3 |  |
| Auxiliary | 2 of 3 | 2 |  |  | 1 |
| Mixed precision (NextLA.MixedPrecision) | 4 of 6 | 4 |  | 1 | 1 |
| Tile low-rank (NextLA.TileLowRank) | 2 of 3 | 2 |  |  | 1 |
| **All** | **24 of 63** | 24 | 4 | 6 | 29 |

## BLAS level 3 (matrix–matrix)

| | Routine | In NextLA |
|:-:|:--|:--|
| ✅ | gemm | `gemm!`, `recgemm!`, `gemmEx!`, `gemm_batched!`, `matmul!` (tiled kernel), `precision_gemm_grouped!` (unexported) |
| ✅ | trsm | `trsm!`, `unified_rectrxm!`, `trsm_batched!` |
| ✅ | trmm | `trmm!`, `unified_rectrxm!` |
| ✅ | syrk | `syrk!`, `recsyrk!`, `SYRK_KERNEL!`, `syrk_batched!` |
| ⬜ | herk | — |
| ⬜ | symm / hemm | — |
| ⬜ | syr2k / her2k | — |

## BLAS level 2 (matrix–vector)

| | Routine | In NextLA |
|:-:|:--|:--|
| ✅ | ger / geru | `geru!` |
| ✅ | gerc | `gerc!` (unexported) |
| ⬜ | gemv | — |
| ⬜ | trmv / trsv | — |
| ⬜ | symv / hemv, syr / her, syr2 / her2 | — |
| ⬜ | banded and packed (gbmv, sbmv, tbsv, spmv, …) | — |

## BLAS level 1 (vector)

| | Routine | In NextLA |
|:-:|:--|:--|
| 🟡 | axpy | `axpy!` (unexported) |
| ✅ | rotg | `lartg` (unexported) |
| ⬜ | dot, nrm2, asum, iamax, scal, copy, swap, rot | — |

## Cholesky (symmetric / Hermitian positive definite)

| | Routine | In NextLA |
|:-:|:--|:--|
| ✅ | potrf | `potrf!`, `potrf_recursive!`, `potrf_batched!`, `cholesky_lower!`, `cholesky_blocked!` |
| ⬜ | potrs (solve with the factor) | — |
| ⬜ | potri (inverse) | — |
| ⬜ | pocon (condition number) | — |
| ⬜ | pstrf (pivoted Cholesky) | — |
| ⬜ | pbtrf / pptrf (banded, packed) | — |

## LU (general matrices)

| | Routine | In NextLA |
|:-:|:--|:--|
| 🟡 | getrf, partial pivoting | `getrf2!`, `lu!(::NextLAMatrix)`, `lu_base!` (n ≤ 32), `tile_lu_factor!` (GPU, pivots within a tile) |
| ✅ | getrf, no pivoting | `lu_recursive!`, `lu_recursive_nopiv!` |
| ✅ | getc2 (complete pivoting) | `getc2!` |
| ✅ | laswp (row swaps) | `laswp` |
| 🟡 | tile LU kernels (getrf_incpiv, gessm, ssssm) | `dgetrf_tiled!`, `dgessm_kernel_L_inv!`, `dssssm_kernel!` (unexported) |
| ⬜ | getrs (solve), gesv (driver) | — |
| ⬜ | getri (inverse), gecon (condition) | — |
| ⬜ | gbtrf (banded) | — |

## QR and LQ

| | Routine | In NextLA |
|:-:|:--|:--|
| ✅ | geqrt (QR of a tile, compact WY) | `geqrt!`, `geqr2!` (unexported) |
| ✅ | larfg / larf / larft / larfb | `larfg!`, `larf!`, `larft!`, `larfb!` (unexported) |
| ✅ | unmqr / ormqr (apply Q) | `unmqr!` (unexported) |
| ✅ | tsqrt / tsmqr, ttqrt / ttmqr (tile QR) | `tsqrt!`, `tsmqr!`, `ttqrt!`, `ttmqr!`, `parfb!`, `pamm!`, `pemv!` (unexported) |
| ⬜ | tiled QR driver over a whole matrix | — |
| 🟡 | geqrf (blocked QR with τ) | `geqrt!` per tile |
| ⬜ | orgqr / ungqr (form Q explicitly) | — |
| ⬜ | gelqt, tslqt, ttlqt (LQ) | — |
| ⬜ | geqp3 (QR with column pivoting) | — |
| 🔄 | CAQR / TSQR, 2.5D QR, CholeskyQR3 | — |

## Least squares and triangular

| | Routine | In NextLA |
|:-:|:--|:--|
| ⬜ | gels (least squares) | — |
| ⬜ | trtri (triangular inverse) | — |
| ✅ | trtrs (triangular solve) | `trsm!` |
| ⬜ | trcon, lauum | — |

## Eigenvalues

| | Routine | In NextLA |
|:-:|:--|:--|
| ⬜ | sytrd / hetrd (tridiagonal reduction) | — |
| ⬜ | steqr / stedc, syev / heev / syevd | — |
| 🔄 | gehrd, hseqr, geev (nonsymmetric) | — |
| ⬜ | sytrf / hetrf (symmetric indefinite) | — |

## Singular values

| | Routine | In NextLA |
|:-:|:--|:--|
| 🔄 | gebrd / gbbrd (bidiagonal reduction) | — |
| 🔄 | bdsqr / bdsdc (bidiagonal SVD) | — |
| 🔄 | gesvd / gesdd (driver) | — |

## Auxiliary

| | Routine | In NextLA |
|:-:|:--|:--|
| ✅ | lamch (machine parameters) | `lamch` (unexported) |
| ✅ | lartg (Givens) | `lartg` (unexported) |
| ⬜ | lange / lansy (norms), lacpy / laset | — |

## Mixed precision (NextLA.MixedPrecision)

| | Routine | In NextLA |
|:-:|:--|:--|
| ✅ | containers | `FullMixedPrec`, `SymmMixedPrec`, `TriMixedPrec`, `TiledTriMixedPrec` |
| ✅ | precision choice and quantization | `adaptive_precisions`, `quantize`, `dequantize`, `reconstruct_matrix` |
| ✅ | GEMM, TRSM/TRMM, SYRK | `recgemm!`, `unified_rectrxm!`, `recsyrk!` (mixed methods) |
| ✅ | Cholesky, LU | `potrf_recursive!`, `lu_recursive!` (mixed methods) |
| 🔄 | mixed-precision solve / iterative refinement | — |
| ⬜ | mixed-precision QR | — |

## Tile low-rank (NextLA.TileLowRank)

| | Routine | In NextLA |
|:-:|:--|:--|
| ✅ | TLR matrices and compression | `TLRMatrix`, `CompressedFTLRMatrix`, `uncompress!` |
| ✅ | TLR matrix multiply | `NextLA.gemm!` and `NextLA.gemm` with TLR operands |
| ⬜ | TLR factorizations (Cholesky, LU) | — |
