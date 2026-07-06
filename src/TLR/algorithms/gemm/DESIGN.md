# TLR × TLR → dense GEMM — implementation guide

This document explains the code under `src/TLR/algorithms/gemm/`. It describes
what each layer does and how a call flows through them, so the individual files
can be read without reverse-engineering the whole pipeline.

Entry point:

```julia
gemm!(C::AbstractMatrix, A::TLRMatrix, B::TLRMatrix; alpha=true, beta=false, max_workspace)
```

computes `C := alpha·(A·B) + beta·C`, where `A`, `B` are tile-low-rank and `C` is
a dense column-major matrix.

## 1. What is computed

Split each operand into its dense diagonal `D` and low-rank off-diagonal `O`:

```text
A·B = (D_A + O_A)(D_B + O_B)
    = D_A D_B  +  O_A D_B  +  D_A O_B  +  O_A O_B
```

All four terms accumulate into the same dense `C` (which is first scaled by
`beta`). The first three are "easy" — single-stage batched products handled in
`diagonal.jl`; they already support boundary tile categories and the corner
diagonal tile.

The **hard term `O_A O_B`** is the focus of the rest of the pipeline. With
`A_ik = U_ik V_ikᵀ` and `B_kj = W_kj Z_kjᵀ` (constant `maxrank`), it is computed
in three stages, summed over the contraction tile index `k`:

```text
Stage 1   S_ikj = V_ikᵀ W_kj      (r_A × r_B)   contract the block dim b
Stage 2   T_ikj = S_ikj Z_kjᵀ     (r_A × b)     contract r_B
Stage 3   C_ij += U_ik T_ikj      (b   × b)     contract r_A, reduce over k
```

Only Stage 3 writes `C`, and its left operand `U` comes from `A` — this single
fact drives the whole traversal choice (§4).

## 2. Scope and invariants

The hard term operates on the **uniform core** only (interior `b×b` off-diagonal
tiles, `int_U`/`int_V`), where `b` is the shared nominal tile size; constant
`maxrank` with zero-padded
inactive rank columns. It does not pack factors or materialise padded tensors —
every stage operand is a zero-copy strided view, and both CPU and CUDA execute
through `gemm_batched!`. Boundary panels (`right`/`bottom`) and the corner are
handled only for the easy terms so far.

## 3. Layering (the morphism chain)

```text
physical TLR storage
  → zero-copy factor-panel views        panel.jl
  → logical operands                     panel.jl
  → budgeted run descriptors             schedule.jl
  → S/T scratch + batch view buffers     schedule.jl
  → stage descriptors                    stage.jl
  → batched GEMM execution               stage.jl → gemm_batched!
```

Files:

| file | responsibility |
| --- | --- |
| `layout.jl` | pure traits: `Stride1Axis`, `KAxisSchedule`, `FreeAxisSchedule`, and the functions deriving them from `TLRMatrix` types |
| `panel.jl` | physical `PanelView` over factor storage + tile-coordinate utilities; logical operands (`LogicalTLROperands`), `ScratchS`/`ScratchT` |
| `schedule.jl` | budgeted `RowRun`/`ColumnRun` iterators; `allocate_workspace` and the reusable batch-view buffers |
| `stage.jl` | `StageDescriptor` and the `execute_stage!` methods that lower each stage straight to `gemm_batched!` |
| `diagonal.jl` | the three easy terms |
| `gemm.jl` | `gemm!` entry + validation + `_offdiag_gemm!` orchestration |

The key design rule: no single function knows about storage layout, stage
algebra, diagonal skipping, workspace budgeting, and execution dispatch at once.
Control flow is selected by trait dispatch, not by branches inside a big driver.

## 4. Layout traits and the four combos (`layout.jl`)

Storage order fixes which tile axis is contiguous:

```text
stride1_axis_left(A)  = Stride1Axis{:i}  (A col-major)  |  Stride1Axis{:k}  (A row-major)
stride1_axis_right(B) = Stride1Axis{:k}  (B col-major)  |  Stride1Axis{:j}  (B row-major)
```

Two derived traits then pick the algorithm:

- **`KAxisSchedule`** — where the `k`-reduction goes. Determined by A only:
  - `A :k` → `KAsGemmK{BAx}` — `k` fuses into Stage 3's contraction dim ⇒
    each `C` tile is **written once** (block-ROW sweep).
  - `A :i` → `KAsSerialLoop{BAx}` — `k` becomes an outer loop, Stage 3
    **accumulates** with `β=1` (block-COLUMN sweep).
  - `BAx` is B's contiguous panel axis (`:k` or `:j`), carried as a type
    parameter so Stage 1 specialisation is selected without runtime checks.

- **`FreeAxisSchedule`** — how Stage 1 fuses the free tile axes:
  - `free_axis_schedule(::KAsGemmK{:j}) = JAsGemmN`   (j-fused Stage 1)
  - `free_axis_schedule(::KAsGemmK{:k}) = FreeAsBatch` (per-tile Stage 1)
  - `free_axis_schedule(::KAsSerialLoop{:k}) = IAsGemmM`   (i-fused Stage 1)
  - `free_axis_schedule(::KAsSerialLoop{:j}) = IJAsGemmMN`  (i- and j-fused)

Resulting dispatch for the four `(A order, B order)` combinations:

| combo | `stride1_axis_left/right` | `KAxisSchedule` | `FreeAxisSchedule` | traversal / Stage 1 |
| --- | --- | --- | --- | --- |
| `(k,j)` Row/Row | `:k`,`:j` | `KAsGemmK{:j}` | `JAsGemmN` | write-once ROW, j-fused S1 |
| `(k,k)` Row/Col | `:k`,`:k` | `KAsGemmK{:k}` | `FreeAsBatch` | write-once ROW, per-tile S1 |
| `(i,k)` Col/Col | `:i`,`:k` | `KAsSerialLoop{:k}` | `IAsGemmM` | accumulate COL, i-fused S1 |
| `(i,j)` Col/Row | `:i`,`:j` | `KAsSerialLoop{:j}` | `IJAsGemmMN` | accumulate COL, i+j-fused S1 |

`(k,j)` is the recommended layout: write-once traversal *and* a fused Stage 1.
Fusing `j` (B row-major makes `W_k,:` contiguous) turns Stage 1 from many tiny
`r_A×r_B` GEMMs into one wide GEMM per `(i,k)` — measured ~1.2–2× on the whole
`gemm!` at large `nt`, since tiny-GEMM Stage 1 otherwise dominates the row family.
`(k,k)` cannot fuse (both operands stride-1 in `k`) and stays tilewise.

## 5. Physical access (`panel.jl`)

`PanelView{Side,Factor,Ax,M,A3}` wraps one flat factor array `[b, maxrank, n_off]`
plus its `TLRMatrix` and `noff = nt-1`. `Side`/`Factor` (`LeftOperand`/`VFactor`,
…) document which stage quantity a view is, so stage code never spells raw
`view(A.int_V, …)`.

Accessors, all zero-copy:

- `tilefactor(p, i, j)` — factor of off-diagonal tile `(i,j)` (via `_offdiag_index`);
- `rowpanel(p, r)` — a contiguous `[b, maxrank, noff]` row panel;
- `local_to_col(r, pos)` — map a row panel's local position to the actual tile
  column, skipping the diagonal.

`logical_operands(A, B)` bundles the stage-panel operands as
`LogicalTLROperands(av=V, bw=W, bz=Z)`. Stage 3 uses the `U` views already
stacked in workspace. The dense output is passed as `C` directly;
`dense_tile(C, ...)` / `dense_rowblock(C, ...)` cut zero-copy `b×b` or row-block
views of it.

## 6. Scheduling and workspace (`schedule.jl`)

The workspace budget (bytes) is the only runtime knob. It caps the run width; the
T-workspace dominates (`≈ r_A · noff · b · run_width`).

- **Row family** (`KAsGemmK`): `RowRun(i0, i1, j0, j1)` — a rectangular block of
  output tiles (rows `i0:i1` × columns `j0:j1`). Rows are independent (no
  cross-`i` dependence), so every stage batches over the whole block — `i` is a
  batch axis, not a serial loop. `_row_block` sets `maxI × maxJ` from the budget
  (columns filled first), so at full budget the entire grid is one run of 3
  batched GEMMs.
- **Column family** (`KAsSerialLoop`): `ColumnRun(k0, k1, jpos0, jpos1)` — a
  block of contraction tiles `k0:k1` × B row-`k` **local** panel positions
  `jpos0:jpos1` (`jpos`; actual columns via `local_to_col(k, jpos)`, which avoids
  allocating a `[j for j≠k]` list per `k`). Stages 1/2 are independent over `k`
  so they batch the whole `k`-block; Stage 3 loops `k` (the reduction).
  `_column_block` sets `maxK × maxJ` from the budget (positions filled first), so
  at full budget the hard term is `2 + nt` launches instead of `3·nt`.

`runs(placement, A, B, budget)` returns the matching iterator.

`allocate_workspace(placement, …)` allocates, once per hard-term call:
`ScratchS`/`ScratchT` sized to the run width, the reshaped `Ustacked`
(row) or `Vstacked`/`Ufactored` (column) operands, and a named tuple of
**batch-view buffers**. The buffers are concrete `Vector`s of a concrete view
type, `sizehint!`-ed once and refilled per run with `empty!`/`push!` — capacity
is reused, so hot loops allocate nothing.

## 7. Stages and execution (`stage.jl`)

A `StageDescriptor` bundles `(stage, placement, run, ops, workspace, C, alpha,
blocking)`. `stage1/2/3(placement, run, ops, ws[, C, alpha])` build them;
`execute_stage!` is dispatched on `(Stage, KAxisSchedule[, FreeAxisSchedule])`
and refills the batch buffers, then calls `gemm_batched!` once.

`prepare_run!` runs before the stages: for row runs it zeroes the used T slice
(so the dead `k=j` slots contribute nothing to the fused-K Stage 3); for column
runs it is a no-op (Stage 2 writes exactly the slots Stage 3 reads).

### Row / write-once family (`KAsGemmK`)

Per `RowRun(i0:i1, j0:j1)`. Scratch `T[:,kk,:,jl,il]` is indexed by absolute
column `jl` (so the fused-K Stage 3 sees a clean `(k,j)` grid); scratch
`S[:,:,p,kk,il]` is indexed by the *off-diagonal position* `p` within the block
(the diagonal `j=k` is skipped, so columns past `k` shift down) — this lets a
fused Stage 1 write a contiguous `[r_A, len·r_B]` slice. Stage 2 maps `p→jl`.

- **Stage 1** — `FreeAsBatch` (`(k,k)`): for each `i`, `k≠i`, `j∈j0:j1` with
  `j≠k`, push `V_ikᵀ`, `W_kj`, the `S` slot; one batched `'T','N'` over `(i,k,j)`.
  `JAsGemmN` (`(k,j)`): B row-major makes the block's off-diagonal columns a
  contiguous slice of `rowpanel(k)`, so `j` fuses into N — `V_ikᵀ · W_k[block]`,
  one wide GEMM per `(i,k)`, batched over `(i,k)`.
- **Stage 2**: same iteration; `S_ikj · Z_kjᵀ` scattered into the clean `(k,j)`
  grid `T[:, kk, :, jl, il]`; batched `'N','T'`.
- **Stage 3**: `k` is fused into `K = noff·r_A`. For each `i ∈ i0:i1`,
  `Ustack = Ustacked[:,:,i]`, `Tstack = reshape(T[…,il], noff·r_A, Jw·b)`; a
  single `'N','N'` batched GEMM over `i` (β=1) writes every `C[i, j0:j1]`
  row-block in place. Rows being independent, the whole block is one launch.

### Column / accumulate family (`KAsSerialLoop`)

Per `ColumnRun(k0:k1, jpos0:jpos1)` (scratch `S`/`T` carry a `k`-axis:
`S[:,:,jx,kx]`, `T[:,:,:,jx,kx]`):

- **Stage 1** — `IAsGemmM` (`(i,k)`): `Vpanel_kᵀ · W_kj` fuses over `i`
  (`M = |I|·r_A`), batched over `(k, jpos)`, `'T','N'`. `IJAsGemmMN` (`(i,j)`):
  the column block is a contiguous slice of B's row-`k` panel, so `j` also fuses
  into one GEMM (`Vpanel_kᵀ · Wsub`) per `k`, batched over `k`.
- **Stage 2**: `S · Z_kjᵀ` into `T`, batched over `(k, jpos)`, `'N','T'`.
- **Stage 3**: the reduction axis `k` is looped (one accumulate GEMM per `k`,
  batched over `(i, jpos)`, `β=1`); different `k` write the same `dense_tile(i,j)`
  so they cannot share a batch, but successive launches accumulate.

## 8. Orchestration (`gemm.jl`)

`gemm!` validates shapes and the shared nominal tile size, scales `C` by `beta`, adds the
three easy terms, then calls `_offdiag_gemm!`, which is small:

```julia
placement = k_axis_schedule(stride1_axis_left(A), stride1_axis_right(B))
ws  = allocate_workspace(placement, A, B, C, budget)
for run in runs(placement, A, B, budget)
    prepare_run!(placement, run, ws)
    execute_stage!(stage1(placement, run, ops, ws))
    execute_stage!(stage2(placement, run, ops, ws))
    execute_stage!(stage3(placement, run, ops, ws, C, alpha))
end
```

The control-flow shape is fixed by `placement`; the loop body only touches
concrete run, workspace, operand, and dense output objects.

## 9. Status and extension points

Working and validated (CPU + CUDA, all four combos, `alpha`/`beta`,
`budget = 1` and large): the full `A·B` on the uniform core. Launch counts at
full budget: **3** (row family, all of `(i,k,j)` batched) and **2 + nt** (column
family, `k` looped in the reduction); `(k,j)` Stage 1 is `j`-fused (`JAsGemmN`).

Not yet done, in rough priority:

- **Occupancy floor / k-panel fallback** — batch Stage 3 over several rows for
  small `b`; sub-panel `k` with `β=1` when one tile's full-`k` T exceeds the
  budget (the only correctness gap at extreme budgets).
- **Boundary tiles in the hard term** — extend beyond the uniform core.
- **`canonicalize`** — fold `(i,k)→(k,j)` (transpose) and `op(A)·op(B)` /
  conjugation into trait selection.
- **TLR×TLR → TLR** — introduce an output abstraction only when a low-rank
  accumulator/recompression target exists; the panel/schedule/stage machinery
  can then be reused intentionally.
