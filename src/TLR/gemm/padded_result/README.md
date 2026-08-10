# PaddedFTLR-result GEMM

This directory implements only

```text
PaddedFTLRMatrix × PaddedFTLRMatrix → PaddedFTLRMatrix.
```

It is separate from dense accumulation because its output is built by adaptive
randomized approximation rather than by accumulating into dense tiles.

- `operands.jl` exposes padded interior factor panels.
- `run_coupling.jl` implements the batched implicit product operators used by
  the rolling sampler.
- `rolling_schedule.jl` admits and retires output tiles under a capacity bound.
- `workspace.jl` owns `TLRGemmWorkspace` and the phase arenas.
- `driver.jl` validates the canonical storage contract and owns public dispatch.

The mathematical derivation and stopping rule are in [`algorithm.tex`](algorithm.tex).
Dense-result compressed metadata, schedules, analyses, and workspaces are not
used by this subsystem.
