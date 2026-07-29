# NextLA experiments

From the repository root:

```bash
julia --project=experiments -e 'using Pkg; Pkg.instantiate()'
julia --project=experiments experiments/run_experiments.jl
```

The all-campaign runner starts fresh Julia processes for:

- `dense_output/run_experiments.jl`: dense output from padded and packed
  (variable-rank compressed) FTLR operands, using FP16/FP32, BF16/FP32, and
  FP32/TF32. Padded operands are included only in strong scaling as the
  old-format comparison; the other sweeps use compressed operands only;
- `padded_ftlr_output/run_experiments.jl`: padded FTLR output from padded FTLR
  operands, using full FP32.

Each experiment's sweep parameters are at the top of its own file. Shared
precision, repetition, seed, workspace, and output settings are at the top of
the corresponding `run_experiments.jl`. Results are written below that
campaign's `results/` directory after every completed configuration. Restarting
a campaign preserves those files and skips configurations already recorded.

Dense-output correctness checks retain a dense reference on the GPU. Set
`CHECK_RESULTS = false` in a campaign runner only when the largest case cannot
fit the reference alongside the measured operands.
