# NextLA experiments

From the repository root:

```bash
julia --project=experiments -e 'using Pkg; Pkg.instantiate()'
julia --project=experiments experiments/run_experiments.jl
```

The all-campaign runner starts fresh Julia processes for:

- `dense/run_experiments.jl`: padded and variable-rank compressed inputs,
  dense output;
- `PackedFTLR/run_experiments.jl`: the asymmetric-rank padded-input cases;
- `tlr_output/run_experiments.jl`: direct TLR output and dense recompression.

Each experiment's sweep parameters are at the top of its own file. Shared
precision, repetition, seed, workspace, and output settings are at the top of
the corresponding `run_experiments.jl`. Results are written below that
campaign's `results/` directory.

Dense-output correctness checks retain a dense reference on the GPU. Set
`CHECK_RESULTS = false` in a campaign runner only when the largest case cannot
fit the reference alongside the measured operands.
