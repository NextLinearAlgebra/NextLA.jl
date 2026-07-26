# TLR GEMM — worklog and roadmap

The target design is [algorithm.tex](algorithm.tex): a blocked adaptive
randomized range finder (ARA) applied per output tile, with the reduction kept
as an implicit factor list instead of a dense tile.

Reference: Boukaram, Turkiyyah & Keyes, *Randomized GPU algorithms for the
construction of hierarchical matrices from matrix-vector operations*, SIAM J.
Sci. Comput. 41(4):C339–C366, 2019, Algorithm 2.3.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done.

## Design decisions

### ARA orthogonalization

Each sample block uses `(projection · Cholesky-QR)²`, matching Algorithm 2.3.
The interleaving is required in finite precision: the first normalization can
amplify a near-null column's overlap with the existing basis, so the second
projection must happen after that amplification.

The factorization is unshifted. A shift places a data-independent floor under
the triangular diagonal used by the stopping test and can make a requested
tolerance impossible to reach. With an unshifted promoted Gram, factorization
breakdown is treated as a rank signal and the valid leading block is retained.
The supported tolerance floor is `sqrt(eps(Tgram)/2)`.

The stopping signal is the diagonal of the composite triangular factor, not
the original sample-column norm. Random combinations of a small residual range
can all have large column norms while becoming dependent within the block; the
triangular diagonal detects that dependence.

### Active packing

R3 uses a dense active prefix. After a tile converges, its run-local state is
swap-removed into a retired suffix. Subsequent GEMM, Gram, factorization, and
triangular-solve calls operate on `1:nactive`, without masked batch slots or
device pointer lists containing holes.

Output layout selects the run family:

- tile-column-major output uses fixed-column runs, applies `XΩ`, and hoists the
  reusable right-side contraction;
- tile-row-major output will use the symmetric fixed-row runs, applies `XᵀΩ`,
  and hoists the reusable left-side contraction.

Ranks size runs but do not select a different fold per tile. Per-tile fold
selection would fragment the batch and require active repacking.

### Truncation

After range finding, one co-range apply forms `Z = XᵀQ`. A thin SVD of `Z`
provides the optimal represented-matrix truncation because `Q` is orthonormal:
the singular values of `Z` are the singular values of `QZᵀ`. The reported
residual is therefore the achieved discarded energy, not an estimate.

## Workspace contract

Compression allocates its reusable numerical storage up front. The live ARA
set is:

1. persistent basis `Q` and one Gaussian block;
2. one block sample and BCGS2 coefficient buffer;
3. one promoted panel, one promoted Gram, and two triangular factors;
4. one block of convergence scalars plus status/cut vectors;
5. co-range storage, written directly into the output factor panel;
6. rank and error diagnostics.

The removed shift machinery and wide one-shot sketch buffers are not part of
the workspace. Sampler, sample-output, basis, co-range, and triangular batch
views are built with the workspace. Packed execution reuses these arrays and
only restricts operations to the current active prefix.

The remaining material allocation is final truncation: backend thin-SVD APIs
currently return fresh factor arrays and may allocate solver workspaces.
Making truncation consume caller-owned storage is part of R4.
An empty co-range returns empty factors directly and never enters LAPACK or a
GPU solver.

## Cleanup before R3b/R4 — 2026-07-27

The unsupported shared-basis TLR-result implementation and dense-slab
recompression fallback were removed together with their tests and obsolete
design document. The `gemm!(::TLRMatrix, ::TLRMatrix, ::TLRMatrix)` method is
absent until the ARA implementation is integrated at R5.

The old general orthogonalization and coordinate-pruning frameworks were also
removed. Active ARA now owns one small `ARACholeskyWorkspace` and one
`ara_cholesky_pass!` primitive that performs only promoted copy, batched Gram,
unshifted batched Cholesky, upper-factor copy, and batched triangular solve.
ARA calls the primitive twice with projection interleaved. Shift buffers,
factor-output construction, rank-policy code, and the pre-Gram column-norm
buffers are gone.

This cleanup deliberately leaves dense-output GEMM, ARA compression, the R1/R2
factor-list primitives, and the R3 packed column-family implementation intact.

Focused verification:

- CPU: numerical primitives 4/4; ARA bookkeeping 25/25, basis 24/24,
  truncation 34/34, and interleaving 6/6; compression 110/110; dense-output
  GEMM 66/66; R1 23/23, R2 20/20, and packed R3 40/40.
- CUDA through `../gpuenv`: numerical primitive 1/1; ARA bookkeeping 2/2,
  basis 10/10, and truncation 9/9; R1 2/2, R2 3/3, packed R3 40/40; focused
  packed compression reconstruction passed.
- Whole TLR suite intentionally not run unless focused failures require it

## Roadmap

- [x] **A0 — convergence bookkeeping.** Per-member sample counts, running
  maximum, consecutive-small count, detected rank, and breakdown masking.
- [x] **A1 — batched ARA core.** Reusable workspace and interleaved two-pass
  projection/normalization.
- [x] **A2 — final truncation.** Batched thin SVD, optimal rank selection,
  achieved residual, and zeroed factor tails.
- [x] **A3 — compression integration.** Dense-tile sampler using ARA rather
  than a wide one-shot sketch.
- [x] **R1 — factor-list sampler.** Coupling prologue plus fused right/left
  application for one output tile.
- [x] **R2 — one-tile range finder.** ARA over the implicit factor list.
- [~] **R3 — batch across a run.** Packed-active column-family implementation
  is complete. Next: symmetric fixed-row implementation for tile-row-major
  output.
- [ ] **R4 — scheduler and arena.** Rank-metadata workspace estimates,
  budget-bounded run growth, reduction sub-panelling, reusable run workspaces,
  and caller-owned truncation scratch where supported.
- [ ] **R5 — integration.** Restore TLR-result `gemm!` using only the ARA path;
  add boundary regions, remaining layout pairs, transpose handling, and
  `beta != 0`.

## Deferred

Ragged per-row/per-column ranks; non-regular boundary tiles; transposed
operands with `beta != 0`; two-sided/BLR² output; alternative batched QR
implementations; and a measured run-level fold cost model.
