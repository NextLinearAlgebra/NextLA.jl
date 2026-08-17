# Canonical (padded-result) TLR-output GEMM

This directory implements only

```text
CompressedFTLRMatrix × CompressedFTLRMatrix → CompressedFTLRMatrix,
```

where `C` is constructed in a reserved-capacity degenerate mode (uniform
`execution_ranks`, logical `ranks` all zero until the run discovers real
content) — the direct analogue of the fixed-`maxrank` container this
subsystem used before `PaddedFTLRMatrix` was collapsed into
`CompressedFTLRMatrix`. It is separate from dense accumulation
(`dense_result/`) because its output is built by adaptive randomized
approximation rather than by accumulating into dense tiles.

`C`, `A`, and `B` must all use the default *complementary packing* (`outer`
row-major, `inner` column-major — see `_require_complementary_packing` in
`driver.jl`). That packing's logical outer/inner order is transpose-invariant,
which is what lets a single code path serve all four transpose combinations
(`NN`, `NT`, `TN`, `TT`) — `choose_tlr_sampling_side` is a pure cost
comparison between a zero-copy right-sampling stack (from `A`) and a
zero-copy left-sampling stack (from `B`), never gated on which transpose flag
is set. `TN` was unsupported under the old single-fixed-physical-layout
`PaddedFTLRMatrix` regime (neither stack was zero-copy for that combination);
it works here for free.

- `operands.jl` exposes interior factor panels (`InteriorOperand`,
  `LogicalTLROperands`) built from `CompressedFTLRMatrix`'s packed storage via
  a uniform-capacity reshape, and `_beta_tile_factors` for reading `C`'s own
  (uniform-width) prior content when `beta != 0`.
- `run_coupling.jl` implements `RunCoupling{Fixed}` (`Fixed` is `:column` or
  `:row`) — the batched implicit product operator used by the rolling
  sampler, one type for both fixed-axis traversals (a third, `:row_right`,
  historically existed but is provably unreachable once B's inner factor is
  always zero-copy — see the struct's docstring).
- `rolling_schedule.jl` admits and retires output tiles under a capacity
  bound, dispatching `_finalize_wave!`/the hot sampler on `run`'s `Fixed`
  type parameter rather than branching at the call site.
- `workspace.jl` owns `TLRGemmWorkspace` and the phase arenas.
- `driver.jl` validates the complementary-packing contract and owns public
  dispatch.

The mathematical derivation and stopping rule are in [`algorithm.tex`](algorithm.tex).
Dense-result compressed metadata, schedules, analyses, and workspaces are not
used by this subsystem.
