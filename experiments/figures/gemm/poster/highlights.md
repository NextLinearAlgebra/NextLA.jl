# Poster-ready GEMM highlights

All NextLA figures use median numerical time from the stored CSV rows.

## Memory--performance

- At `N=16384`, FP16, skewed TLR x TLR, `b=N/8`: **6.34x speedup at 17.8% total memory** (1.699 ms, target runs=2).
- At the minimum measured memory point (9.0%), NextLA still achieves **2.80x speedup**.

## Precision scaling

- Best relative result: **20.54x FP32 speedup** at `N=65536`, reaching 92.1% of its 22.29x executed-FLOP ceiling while using 17.3% total memory.
- At `N=65536`: FP16 9.57x (81.8 ms), BF16 9.11x (84.5 ms), and TF32 11.80x (141.8 ms).
- FP32 has the largest relative speedup; FP16/BF16 have the lowest absolute time.

## KBLAS comparison

- Application/API comparison: NextLA skewed ranks are **1.75--2.84x faster** than KBLAS padded uniformly to `r=b/16`. This is not a constant-work comparison.
- Controlled `b=N/16`, `r=b/16` comparison using the fastest tuned NextLA workspace: **5/5 wins, 1.74x geometric mean** (1.17--2.58x).
- Across the complete 20-case constant-rank grid, fastest-workspace NextLA wins **19/20** with a **1.44x geometric mean**.

## Sources

- `experiments/results/gemm/nextla/memory_pareto_fp16_n16384_rank_b32_b16.csv`
- `experiments/results/gemm/nextla/precision_scaling_skewed_rank_b32_b16_best_workspace.csv`
- `experiments/results/gemm/kblas/constant_rank_fp32_rank_b16_b8.csv`
- `experiments/results/gemm/comparisons/constant_rank_fp32_best_nextla_vs_kblas.csv`
