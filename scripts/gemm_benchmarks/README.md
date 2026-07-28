# GEMM benchmark suite

The three related GEMM studies are kept here and use one configuration entry
point. Common settings (`backend`, `reps`, `warmup`, `seed`, output directory,
precision, and sharding) therefore behave the same for every study. Both
`float32` and `float64` run by default.

From the repository root, a CPU smoke run is:

```bash
scripts/gemm_benchmarks/run_gemm_benchmark.sh \
  --benchmark tlr-output --backend cpu --reps 1 --warmup 0
```

On a GPU node, point `JULIA_PROJECT` at the Julia environment containing CUDA
and NextLA, then select CUDA:

```bash
JULIA_PROJECT=/path/to/gpuenv \
  scripts/gemm_benchmarks/run_gemm_benchmark.sh \
  --benchmark dense --backend cuda --shard-count 8 --shard-index "$SLURM_ARRAY_TASK_ID"
```

Set dimensions, tile sizes, ranks, and precisions by editing the `DEFAULT_*`
values in `config.jl`. Float64 needs more memory and is typically slower, so
use smaller dimensions or more job-array shards if GPU memory is limited.

The default output directory is `scripts/gemm_benchmarks/results/`; use
`--output-dir` to put results on scratch storage. Existing case IDs are
skipped, so rerunning a failed job resumes safely. Use `--help` for all
options. The same options can be supplied with `NEXTLA_GEMM_*` environment
variables, which is convenient in a batch script.

For a site-wide setup, edit the defaults in `config.jl`. The implementation
files are intentionally free of command-line and environment parsing.
