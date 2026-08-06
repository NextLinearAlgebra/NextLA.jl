# Poster GEMM experiments

This directory contains only the benchmark and plotting pipeline used for the
final dense-output TLR GEMM poster figures.

## Layout

- `gemm/`: NextLA benchmark, selection, comparison, validation, and plotting code.
- `kblas/`: standalone KBLAS fixed-rank benchmark and suite launcher.
- `results/gemm/nextla/`: raw NextLA sweeps and selected workspace winners.
- `results/gemm/kblas/`: KBLAS measurements.
- `results/gemm/comparisons/`: derived NextLA/KBLAS joins.
- `figures/gemm/poster/`: the four final experiment figures.
- `method/` and `figures/method/`: the dynamic-programming schedule illustration.

## Environment

From the repository root:

```bash
julia --project=experiments -e 'using Pkg; Pkg.instantiate()'
python3 -m venv .plenv
.plenv/bin/pip install -r experiments/requirements-plot.txt
```

## Reproduce the NextLA suite

Choose a new output directory; the runner refuses to overwrite it:

```bash
bash experiments/gemm/run_poster_suite.sh \
    experiments/results/gemm/runs/my_run
```

The stored configuration uses one warmup and four measured repetitions. It
runs the FP16 memory Pareto sweep, skewed multiprecision workspace tuning, and
FP32 constant-rank workspace tuning, then selects workspace winners and joins
the constant-rank cases with the checked-in KBLAS results.

The skewed rank interval is

```text
r_ij in [b/32, b/16]
```

with normalized ranks generated as `U^2`, `U ~ Uniform(0,1)` (a discretized,
scaled `Beta(1/2,1)` distribution). The controlled comparison uses constant
rank `r=b/16` or `r=b/8`.

## Regenerate poster figures

```bash
.plenv/bin/python experiments/gemm/build_poster_figures.py
```

The plotter deterministically updates the PNG, PDF, SVG, and `highlights.md`
files under `experiments/figures/gemm/poster/`. Measurement runners never
overwrite an existing CSV.

See `gemm/README.md` and `kblas/README.md` for the exact grids and standalone
commands.
