# Benchmarks

Speed of the recursive routines, dense and mixed precision, against the vendor libraries on one
GPU. Measured on an NVIDIA GB10 (NVIDIA DGX Spark) with Julia 1.10.14, 1.13.4, last run 2026-11-24.

- Numbers are **GFLOP/s** (higher is better): the routine's standard flop count divided by the
  fastest of up to five timed calls, after an untimed warm-up.
- Mixed-precision columns use the containers of `NextLA.MixedPrecision`. `F16/F32` means the
  off-diagonal blocks are stored in `Float16` and the diagonal leaves in `Float32`. The leaves are
  4096×4096, so at n = 4096 there is no split and the matrix is one leaf in the leaf type.
- The GB10 runs `Float64` at about 400 GFLOP/s on every library, so lower precisions are where
  the speed is.
- — means not measured.

## Speed (GFLOP/s)

### GEMM

`C = A * B`, square, `2n³` flops.

| n | FP64 cuBLAS | FP64 NextLA | FP32 cuBLAS | FP32 NextLA | F32/F64 | F16/F64 | F16/F32 | F16 dense, F32 accum. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 398 | 402 | 19 297 | 18 974 | 400 | 400 | 17 934 | 62 974 |
| 8192 | 405 | 405 | 18 384 | 19 210 | 885 | 868 | 1 928 | 84 496 |
| 16384 | — | — | 15 764 | 15 134 | 1 693 | 2 123 | 1 547 | 66 222 |
| 32768 | — | — | 14 776 | 15 480 | — | — | — | 95 696 |

### SYRK

`C = A * Aᵀ`, lower triangle, `n³` flops.

| n | FP64 cuBLAS | FP64 NextLA | FP32 cuBLAS | FP32 NextLA | F32/F64 | F16/F64 | F16/F32 | F16 dense, F32 accum. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 392 | 385 | 13 618 | 1 388 | 384 | 384 | 1 370 | 34 246 |
| 8192 | 392 | 414 | 8 637 | 1 362 | 744 | 738 | 1 357 | 56 901 |
| 16384 | 402 | 586 | 14 217 | 1 815 | 2 685 | 2 604 | 1 546 | 95 374 |
| 32768 | — | — | 14 030 | — | 4 631 | 3 259 | — | 91 820 |

### TRSM

`L \ B`, `n` right-hand sides, `n³` flops.

| n | FP64 cuBLAS | FP64 NextLA | FP32 cuBLAS | FP32 NextLA | F32/F64 | F16/F64 | F16/F32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 345 | 345 | 8 395 | 8 228 | 222 | 222 | 665 |
| 8192 | 374 | 374 | 12 121 | 12 274 | 432 | 440 | 1 285 |
| 16384 | 389 | 390 | 13 569 | 14 436 | 839 | 864 | 2 543 |
| 32768 | — | — | 14 231 | 14 530 | — | — | 4 836 |

### TRMM

`L * B`, `n` columns, `n³` flops.

| n | FP64 cuBLAS | FP64 NextLA | FP32 cuBLAS | FP32 NextLA | F32/F64 | F16/F64 | F16/F32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 374 | 390 | 8 833 | 11 276 | 231 | 229 | 394 |
| 8192 | 395 | 400 | 6 699 | 14 844 | 451 | 455 | 770 |
| 16384 | 402 | 403 | 6 768 | 14 734 | 866 | 894 | 1 523 |
| 32768 | — | — | 6 668 | 15 158 | — | — | — |

### Cholesky

`S = L Lᵀ`, `n³/3` flops.

| n | FP64 cuSOLVER | FP64 NextLA | FP32 cuSOLVER | FP32 NextLA | F32/F64 | F16/F64 | F16/F32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 375 | 375 | 6 902 | 6 521 | 374 | 375 | 6 524 |
| 8192 | 397 | 374 | 10 031 | 9 591 | 292 | 237 | 1 705 |
| 16384 | 403 | 386 | 10 674 | 10 396 | 403 | 428 | 2 387 |
| 32768 | — | — | 11 410 | 13 433 | — | — | 3 909 |

### LU

`2n³/3` flops; NextLA does not pivot, cuSOLVER `getrf!` does.

| n | FP64 cuSOLVER | FP64 NextLA | FP32 cuSOLVER | FP32 NextLA | F32/F64 | F16/F64 | F16/F32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 330 | 393 | 3 022 | 2 954 | 381 | 381 | 2 862 |
| 8192 | 373 | 420 | 6 327 | 2 269 | 301 | 302 | 1 278 |
| 16384 | 390 | 432 | 9 997 | 2 072 | 435 | 463 | 1 825 |
| 32768 | — | — | 12 164 | 1 990 | — | — | 3 162 |

## Accuracy

Error of the mixed-precision result against the same computation in `Float64`, at n = 8192
(one split: off-diagonal blocks in the first type, leaves in the second). For the
factorizations it is the solve residual `‖Ax − b‖ / (‖A‖ ‖x‖)`.

| Routine | Measures | F32/F64 | F16/F64 | F16/F32 |
|:---|:---|---:|---:|---:|
| GEMM | relative error of `C` | 8.4e-07 | 1.6e-04 | 1.6e-04 |
| SYRK | not measured on the GPU | — | — | — |
| TRSM | relative error of `X` | 1.6e-09 | 4.1e-08 | 4.1e-08 |
| TRMM | relative error of `B` | 1.1e-07 | 3.1e-06 | 3.1e-06 |
| Cholesky | solve residual | 5.4e-10 | 2.2e-08 | 3.2e-08 |
| LU | solve residual | 5.4e-10 | 2.9e-08 | 3.0e-08 |

## Choosing precisions automatically

[`adaptive_precisions`](@ref NextLA.MixedPrecision.adaptive_precisions) picks the storage type of each
level from a tolerance `ε`. It was run on 8 test matrices of size n = 16384 (kernel matrices, block-graded
matrices and randsvd matrices with condition numbers 10² to 10¹⁰), with 4096×4096 leaves.
Speed-up is the vendor `Float64` time divided by NextLA's (median over the matrices); error is the
largest over the matrices (solve residual for the factorizations, relative error for GEMM).

| ε | Usual choice | Cholesky: speed-up vs cuSOLVER, error | LU: speed-up vs cuSOLVER, error | GEMM: speed-up vs cuBLAS, error |
|---:|:---|:---|:---|:---|
| 1e-04 | F16/F16/F64 | 1.1×, 1e-07 (3 of 8 not SPD) | 1.2×, 4e-07 | 5.3×, 3e-05 |
| 1e-08 | F32/F32/F64 | 1.0×, 9e-10 (1 of 8 not SPD) | 1.1×, 9e-10 | 7.7×, 5e-07 |
| 1e-12 | F64/F64/F64 | 0.8×, 8e-17 | 0.9×, 8e-17 | 1.5×, 7e-14 |

Cholesky can fail with `PosDefException` when `Float16` (or, for the worst-conditioned matrix,
`Float32`) off-diagonal blocks are chosen: `adaptive_precisions` bounds the storage error in norm,
but rounding can still make the matrix lose positive definiteness. LU and GEMM do not have this problem.
