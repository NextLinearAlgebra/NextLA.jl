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

The canonical API stores `C`, `A`, and `B` physically tile-row-major. Sampling
side follows the logical layouts:

- `NN` has only the natural right sample and uses fixed output rows;
- `TT` has only the natural left sample and uses fixed output rows;
- `NT` admits both. A rank-derived retained-workspace estimate chooses a
  fixed-column right run or a fixed-row left run for the whole operation;
- `TN` admits neither natural contraction stack and is deferred.

The side is never selected per tile: doing so would fragment the batch and
require a second active-packing layer. Full-capacity panels are zero-copy.
When rank metadata trims a panel below its physical capacity, its active
prefix is packed once into compact run-local storage and reused for every ARA
pass; reshaping a strided rank slice is not a valid GPU GEMM operand.

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

## Cleanup before R3/R4 — 2026-07-27

The unsupported shared-basis TLR-result implementation and dense-slab
recompression fallback were removed together with their tests and obsolete
design document.

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

## R3 canonical TLR-result integration — 2026-07-27

Added

```julia
gemm!(C::TLRMatrix, A::TLRMatrix, B::TLRMatrix;
      alpha=true, beta=false, transA='N', transB='N',
      tol=0, rel=false, eps_rel=nothing,
      r_required=10, block=32, compute=nothing)
```

for regular tile grids with physical `TileRowMajor` storage. The implementation
supports right and left sampling, swap-compacts converged members, performs one
co-range apply and batched truncation per run, and scatters directly into the
canonical row-major output. `beta*C` participates as an additional factor pair.

For `NT`, both natural sampling sides exist. The chooser uses the active rank
caps from the operand metadata and compares the retained right/left run
workspace. This is an operation-level decision, so every run has one uniform
batch shape. The more complicated `TN` case, arbitrary physical orders,
boundary tiles, and reduction sub-panelling remain explicitly outside this
contract.

Focused verification:

- CPU: canonical default order, `NN`, `NN` with nonzero beta, both `NT` rank
  choices, `TT` with and without beta, chooser behavior, and rejection
  boundaries: 19/19.
- CUDA through `../gpuenv`: `NN`, right-selected `NT`, and `TT`
  reconstruction and rank-cap checks: 6/6.
- The GPU test also verifies compact active-rank panel packing; full-rank
  canonical panels remain views.

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
- [x] **R3 — canonical TLR-result GEMM.** Fixed-column and fixed-row
  packed-active runs, right and left sampling, operation-level rank-based side
  selection, output scatter, and the regular-grid row-major `gemm!` contract.
- [ ] **R4 — scheduler and arena.** Rank-metadata workspace estimates,
  budget-bounded run growth, reduction sub-panelling, reusable run workspaces,
  and caller-owned truncation scratch where supported.
- [ ] **R5 — general-storage integration.** Boundary regions, arbitrary
  physical layout descriptors, the packed/reduced `TN` path, and measured
  scheduling across those cases.

## Deferred

Ragged per-row/per-column ranks; non-regular boundary tiles; arbitrary-storage
and `TN` packing/reduction; two-sided/BLR² output; alternative batched QR
implementations; and a measured run-level fold cost model.
