# Compressed-output TLR GEMM

This directory implements the allocation-returning operation

```text
CompressedFTLRMatrix × CompressedFTLRMatrix → CompressedFTLRMatrix.
```

Output ranks are not known before ARA converges, so finalized packed storage
cannot be supplied as an in-place destination. The driver creates private
uniform factor staging, runs the rolling ARA scheduler, allocates final offsets
from the discovered rank grid, and copies only active factor columns.

Inputs and output use complementary packing: outer factors are tile-row-major
and inner factors tile-column-major. This makes the logical factor orders
transpose-invariant and supports `NN`, `NT`, `TN`, and `TT` with the same
sampling-side cost comparison.

- `run_coupling.jl` packs run-local factor panels and implements the
  fixed-row/fixed-column implicit product.
- `rolling_schedule.jl` greedily admits pending tiles as converged slots retire.
- `workspace.jl` owns numerical ARA arenas and scheduler scratch.
- `driver.jl` validates the contract, stages operands, runs ARA, and packs the
  allocation-returning result.

Boundary tiles remain supported by the container, dense compression, and
dense-output GEMM. Compressed-output GEMM currently rejects ragged grids at its
API boundary. The derivation and stopping rule are in
[`algorithm.tex`](algorithm.tex).
