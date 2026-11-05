# Ordering experiments in Float64

For the broader AMD/RCM/SCOTCH comparison with before/after pictures and an HTML
report, see [ordering/README.md](ordering/README.md). This file describes the
smaller sTiles correctness check.

Run from the repository root in an environment containing sTiles and MatrixMarket
(the existing `test/catalog/setup.jl` installs both):

```bash
julia --project=test/catalog test/stiles_ordering_test.jl --example-only
MATRIX_DIR=/path/to/matrices ORDERING_MATRIX=bodyy5 STILES_CORES=8 ORDERING_REPEATS=5 julia --project=test/catalog test/stiles_ordering_test.jl
```

The example checks correctness; use real SPD matrices to judge performance.
Missing matrix files are reported as skipped. Indefinite inputs fail Cholesky;
the script does not shift the diagonal to make them pass.

The script compares natural order, CHOLMOD's default permutation, and sTiles'
chosen permutation. All three are applied to the same sparse Float64 matrix and
factored by CPU CHOLMOD with its internal reordering disabled. This measures how
the permutations affect that solver, rather than comparing solver implementations.
No NextLA/CUDA import or mixed precision is involved.

The current upstream Julia API runs an ordering bake-off during
`F = sTiles.analyze(A; cores=cores, inverse=false)` and returns the permutation
through `sTiles.permutation(F)`. Close `F` afterward, even if extraction fails.
The script also supports the older `sTiles.ordering(A; cores=cores)` API and its
`iperm` field. The current wrapper exposes no public keyword to force AMD, RCM,
or SCOTCH, and its `mode` keyword controls tile regime, not ordering strategy.
The native selection objective is not specified by this wrapper; do not assume
its chosen order minimizes CHOLMOD time or `nnz(L)`.

For each candidate:

```julia
Ap = A[p, p]
F = cholesky(Symmetric(Ap); perm=collect(1:size(A, 1)))
y = F \ b[p]
x = similar(y)
x[p] = y
```

`nnz(L)` measures factor storage; `fill` divides it by the stored lower-triangle
nonzeros of the input. Lower is better for storage. `factor_s` is the minimum
of repeated complete symbolic + numeric factorizations after one warm-up.
It excludes permutation application, factor extraction, and correctness checks.
`order_s` measures a single setup call and includes compilation overhead; its
scope differs between APIs/candidates, so it is diagnostic, not a fair ordering
speed comparison. CHOLMOD's permutation is obtained with a full factorization;
current sTiles setup includes tile layout. Repeat runs on representative matrices
before drawing timing conclusions. The script reports separate winners for
storage and factor time, and checks factor/solve residuals and forward error.

For dense Cholesky, all orderings still have a dense triangular factor and roughly
the same cubic work. Sparse fill reductions therefore do not automatically imply
speedups for NextLA's dense Cholesky; that requires a separate experiment.

Upstream source: https://github.com/esmail-abdulfattah/sTiles/blob/main/julia/src/factor.jl
