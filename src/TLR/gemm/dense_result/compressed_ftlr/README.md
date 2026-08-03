# CompressedFTLR dense-output scheduling

How the output grid is partitioned into work units, why the current partition is
computed the way it is, and what implementing the 2-D generalisation would take.

Files: `rank_metadata.jl` (rank facts) → `fold_cost.jl` (what each fold costs) →
`schedule.jl` (greedy partition) / `schedule_dp.jl` (optimal partition) →
`execute.jl` (GEMM task construction).

---

## 1. The scheduling problem

For each output tile,

```
C_ij = Σ_k U_ik (V_ik' W_kj) Z_kj'
```

Stage 1 computes the shared `S_ikj = V_ik' W_kj` — identical regardless of what
happens downstream. The remaining product can then be bracketed two ways:

| | Stage 2 | Stage 3 | intermediate |
|---|---|---|---|
| **FoldRight** | `T_ikj = S_ikj Z_kj'` | `C_ij = [U_i1 … U_iq]·[T_i1j; …; T_iqj]` | `n_j · ρ_i` |
| **FoldLeft** | `T_ikj = U_ik S_ikj` | `C_ij = [T_i1j … T_iqj]·[Z_1j'; …; Z_qj']` | `m_i · γ_j` |

with `ρ_i = Σ_k r^A_ik` (A's row rank) and `γ_j = Σ_k r^B_kj` (B's column rank).

When tiles are square (`m_i = n_j = b`, the common case) the two Stage-2 costs are
**identical** — both reduce to `b·Σ_k r^A_ik r^B_kj` — so only Stage 3 differs, and
the per-tile rule is simply:

```
FoldRight if ρ_i ≤ γ_j, else FoldLeft
```

The same comparison also minimises intermediate storage, so cost and memory agree
and there is no trade-off to arbitrate.

**Why not just apply that rule per tile?** Because a fold is chosen per *work unit*,
not per tile: one fold determines the whole intermediate `T` layout and the Stage-3
GEMM shape for that unit. The scheduling question is therefore what the work units
should be, and the answer is constrained by the workspace budget — `S` and `T` for a
unit must fit simultaneously.

---

## 2. Row-run scheduling

A **run** is a contiguous range of output tile-rows, assigned one fold. Runs execute
sequentially and reuse the same arena (`_arena_reset!` between them), so peak
workspace is the *max* over runs, not their sum.

### 2.1 Why greedy is not enough

`_compressed_ftlr_row_runs` (schedule.jl) extends a run until the budget stops it,
then commits. That is provably optimal for minimising the **number** of runs — the
standard interval-covering greedy, since feasibility degrades monotonically as a
range grows — but the objective here is **cost**, and greedy never revisits a
boundary. Merging rows whose cheaper fold differs forces one global choice on all of
them.

Minimal example — three rows with `(F_right, F_left)` costs `(10,1)`, `(1,10)`,
`(10,1)` and an unconstrained budget. Greedy merges all three and picks the cheaper
global fold: `min(Σright, Σleft) = min(21,12) = 12`. Split into three single-row
runs, each takes its own minimum: `1+1+1 = 3`. A 4× gap, from the merge rule alone.

This is not hypothetical: with the generous budget production actually uses
(`gemm_maximum_workspace_bytes`), greedy collapses to a **single run spanning the
whole matrix** and picks one fold for every row.

### 2.2 The DP

`_compressed_ftlr_row_runs_dp` (schedule_dp.jl) solves the partition exactly:

```
DP[0] = 0
DP[j] = min over i=1..j of { DP[i-1] + cost(i:j) }
cost(i:j) = FLOPs of the cheaper fold that fits the budget, or ∞ if neither fits
```

Backpointers recover the partition. Fold selection reuses
`_compressed_ftlr_select_fold` verbatim, so the DP and greedy differ *only* in where
they place boundaries — never in how a run is costed or executed.

**Optimality.** Standard optimal-substructure exchange argument: if `P*` is an
optimal partition and its last run starts at `i`, then `P*` restricted to rows
`1..i-1` must itself be optimal — otherwise substituting a cheaper prefix (keeping
the last run fixed) would beat `P*`, contradiction. Since the recurrence tries every
last-run start against the true optimal prefix cost `DP[i-1]`, it attains the global
minimum.

Scope of that claim, stated precisely for the paper: optimal **over contiguous
row-range partitions with one fold per range** — the class `execute.jl` realises
without extra data movement. It is *not* a claim about unrestricted schedules;
`tile_optimal_flops` in `experiments/fold_schedule_tradeoff.jl` bounds that larger
space.

**Complexity.** O(q²) range queries, each O(1) off the profile's prefix sums; O(q)
memory. Measured ~6 µs at q=32 versus ~0.4 µs for greedy — both negligible against
the GEMMs they schedule. Widening a range only adds non-negative bytes, so once
`i:j` is infeasible no smaller `i` can be, and the inner scan breaks early.

The DP can never lose: greedy's own partition is one of the candidates it evaluates.
Verified over 600 randomised rank grids — never worse, strictly better in 267.

### 2.3 The call-cost caveat

The cost model is pure arithmetic and cannot see per-grouped-call overhead. At
`call_cost = 0` the DP splits freely to chase small FLOP wins: on a uniform
distribution it takes **16 runs to save 1.59%**, i.e. 16× the grouped-call
submissions and arena resets — plausibly a net loss in wall clock.

`COMPRESSED_FTLR_DP_CALL_COST` prices one extra run in the same MAC units. Raising it
monotonically consolidates runs (at `1e8` the uniform schedule collapses from 16 runs
to 2). **Calibrate it against measured launch overhead before trusting the FLOP
ranking.** This is why the default policy remains `:greedy`.

### 2.4 Measured behaviour

`experiments/fold_schedule_tradeoff.jl`, q=32, `:q8`, generous budget. "sorted" =
generator order (rank magnitude coincides with spatial position); "shuffled" = A's
rows and B's columns permuted, which preserves the marginal multisets and the whole
Stage-2 total *exactly* and changes only the arrangement.

| distribution | layout | call cost | greedy → DP | runs |
|---|---|---|---|---|
| uniform | sorted | 0 | −1.59% | 1 → 16 |
| uniform | shuffled | 0 | −1.59% | 1 → 15 |
| uniform | shuffled | 1e8 | −0.08% | 1 → 3 |
| decay | sorted | 0 | −3.03% | 1 → 3 |
| decay | shuffled | 1e8 | −1.08% | 1 → 8 |
| corner (adversarial) | sorted | 0 | −29.36% | 1 → 2 |
| corner | shuffled | 1e8 | −25.62% | 1 → 19 |

Two properties worth knowing:

- **At `call_cost = 0` the DP optimum is permutation-invariant.** It equals
  `Σ_i min(right_row[i], left_row[i])`, and each term depends only on row `i`'s ranks
  and the *global* γ distribution — not on ordering. Confirmed to the last digit
  across both layouts. What shuffling changes is the **run count** needed to reach
  it (2–3 → 15–21), so once calls are priced the achievable gain does erode.
- **The gain is highly distribution-dependent** (−0.08% to −29.4%). Quote it per
  distribution, never as a single headline number, and be explicit that these are
  synthetic rank maps.

---

## 3. If we had to implement peeling

**Peeling** generalises runs from row ranges to rectangles. From remaining region
`(i0..q, j0..q)` you peel either a row-range `i0..b` spanning the remaining columns,
or a column-range `j0..d` spanning the remaining rows. The remainder stays a
rectangle, which keeps the state space O(q²).

### 3.1 Current status and why it is not built

Prototyped and validated in `experiments/fold_schedule_tradeoff.jl`
(`RectCost`, `rect_cost`, `peeling_range_dp`), including the strong cross-check that
**row-only peeling reproduces the row-run DP exactly** — confirming the rectangle
cost formulas agree with the production profile rather than merely ordering the same
way. Peeling is a strict superset of row-runs, so the comparison measures the value
of relaxing contiguity from 1-D to 2-D, nothing else.

Measured gain of peeling **over the row-run DP**:

| distribution | sorted, cc=0 | shuffled, cc=0 | sorted, cc=1e8 | shuffled, cc=1e8 |
|---|---|---|---|---|
| uniform | −0.62% | −0.50% | +0.00% | −0.90% |
| decay | −0.49% | −0.14% | −0.03% | −0.51% |
| corner | −0.28% | −0.10% | −0.02% | −1.12% |

**Maximum observed: 1.12%**, on the adversarial distribution. Against 2–3 days of
work concentrated in offset arithmetic, that does not pay. Note the direction flips
with call cost: sorted layouts favour peeling when calls are free (column peels
isolate contiguous γ blocks), scattered layouts favour it once calls are priced
(a column peel captures scattered-row structure in one region).

### 3.2 What implementation would require

**Scheduling (~half a day, low risk — already prototyped).**

The row-run profile hoists `ω_k = Σ_j n_j·r^B_kj`, which makes range queries O(1) —
but that hoist is **only valid while the column extent is pinned to all columns**.
Once both extents vary you need the full 3-way coupling `Q[i,j] = Σ_k r^A_ik r^B_kj`
(a plain matrix product on the rank grids, O(q³) once), then a 2-D prefix sum gives
O(1) rectangle queries. This complexity increase is structural, not an artefact of
how it is written.

Port `RectCost` / `rect_cost` / `peeling_range_dp` into `src/`; add `Q` and its
prefix sum to the rank plan.

**Execution (2–3 days, moderate-to-high risk — the real cost).**

Both builders in `execute.jl` hardcode "all columns". Every quantity derived from
that needs a range query and a rebase:

| site | change |
|---|---|
| `_compressed_ftlr_stage1_layout` | `koff` uses `plan.b_row_ranks[k]` (σ_k over all j) → jrange-restricted σ_k |
| `_compressed_ftlr_stage1_tasks` | restricted σ_k; `_compressed_ftlr_row_w_stack` stacks W over all j → needs a jrange variant |
| `_compressed_ftlr_sblock` | `b_row_ranks[k]` (block width) rebased; callers' `b_row_k_prefix[k,j]` column offsets likewise |
| FoldRight | `output_col_prefix[end]` → restricted width; `output_col_prefix[j]` → rebased; `for j in 1:qn` → jrange; `view(C, rows, :)` → `view(C, rows, cols)` |
| FoldLeft | `b_total_rank` → restricted; `tbase_cols` and `b_col_prefix` → rebased; `for j in 1:qn` → jrange |
| both | `scale_targets` entries restricted to the rectangle |
| `RaggedRowRun` | gains a column range (or becomes `RaggedRegion`), rippling into `analysis.jl` and the one test calling a builder directly |

~20 edit sites, nearly all offset rebasing. Conceptually it is parameterisation, not
new architecture — but that category produces silent wrong answers rather than
crashes. Calibration: the Stage-1 fusion change was a strictly smaller edit and still
needed 8 targeted differential cases plus a baseline comparison.

**What does NOT change — and must not be claimed otherwise.** Peeling keeps every
factor stack zero-copy. Because A's inner factor is `TileColMajor` and B's outer is
`TileRowMajor`, sub-ranges along *both* axes remain contiguous; `U` and `Z` are only
ever stacked over the full `k` range, which restriction never touches. Alignment also
survives: a rectangle write starts at element offset `(i0-1)·bm + (j0-1)·bn·ldC`, and
the constructor assertion forces both `bm` and `bn` to be multiples of the alignment
quantum. Arguing that peeling would break contiguity or alignment would be wrong and
easy to falsify.

**Test plan.** Differential against a dense reference across: uniform ranks, ragged
ranks with explicit zeros, entire zero rows/columns, tail tiles, both transpose
combinations, and budgets from minimum to generous. Plus the degenerate identity —
a peeling schedule restricted to row peels must produce byte-identical results to
the row-run path.

### 3.3 The open question that would justify revisiting

Peeling's remaining case is **feasibility, not cost**: row-only scheduling fails
outright when `budget < max_i(row_bytes[i])`, since a single unschedulable row kills
the GEMM. A column-capable scheme has a different floor (`max_j` of a column's
requirement) and taking the min of both orientations is strictly lower than either.
This is untested — every experiment above used budgets ≥ `profile.minimum`.

Caveat: peeling's peels always span the full remaining extent in one dimension, so
its *first* peel is never small and it may not move the floor much either. Lowering
the floor meaningfully likely needs arbitrary rectangles, a larger scheme than
peeling. Measure the floor (binary-search minimum feasible budget, row-only vs
column-capable — cheap, CPU-only) before building anything.
