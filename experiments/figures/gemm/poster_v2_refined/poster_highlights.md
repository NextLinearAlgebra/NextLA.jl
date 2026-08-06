# Poster-ready GEMM highlights

All NextLA figures use median numerical time from the stored CSV rows.

## Memory–performance

- At N=16384, FP16, skewed TLR×TLR, b=N/8: 6.34× speedup at 17.8% total memory (1.699 ms, runs=2).
- At the minimum measured memory point (9.0%), NextLA still achieves 2.80× speedup.

## Precision scaling

- Best relative result: 20.54× FP32 speedup at N=65536, reaching 92.1% of its 22.29× executed-FLOP ceiling while using 17.3% total memory.
- At N=65536: FP16 9.57× (81.8 ms), BF16 9.11× (84.5 ms), and TF32 11.80× (141.8 ms).
- FP32 has the largest relative speedup; FP16/BF16 have the lowest absolute time.

## KBLAS comparison

- Application/API comparison: NextLA skewed ranks are 1.75–2.84× faster than KBLAS padded uniformly to r=b/16. This is not a constant-work comparison.
- Controlled constant-rank comparison at equal or lower total memory: 16/20 wins, 1.09× geometric mean.
- For N≥8192: 15/16 wins, 1.16× geometric mean.
- Memory-matched constraint violations: 0.

## Peak-workspace context (not the fair-memory headline)

- Fastest-workspace NextLA wins 19/20 with a 1.44× geometric mean, but uses more memory than KBLAS in 20/20 cases.

## Sources

- `experiments/results/gemm/v2/memory_v2.csv`
- `experiments/results/gemm/v2/skewed_data__winners.csv`
- `experiments/results/gemm/kblas_tlr_tlr_fp32_fixed_rank_b16_b8.csv`
- `experiments/results/gemm/v2/kblas_comparison_v2__memory_matched_vs_kblas.csv`
- `experiments/results/gemm/v2/kblas_comparison_v2__winners__vs_kblas.csv`
