# TLR GEMM architecture

The destination representation is the top-level implementation boundary.

| Destination | Supported product | Implementation |
|---|---|---|
| `AbstractMatrix` | compressed × compressed | `dense_result/` three-stage lowering |
| `AbstractMatrix` | compressed × dense, dense × compressed | `dense_result/` two-stage specializations |
| `AbstractMatrix` | dense-diagonal TLR × dense, dense × dense-diagonal TLR | compressed two-stage specialization plus block-diagonal updates |
| `AbstractMatrix` | dense-diagonal TLR × dense-diagonal TLR | compressed product plus diagonal updates |
| `PaddedFTLRMatrix` | padded × padded | `padded_result/` ARA recompression |

No dense-result code depends on the padded-result scheduler, and no
padded-result code participates in dense accumulation.

## Source layout

```text
gemm/
├── common/
│   ├── operands.jl       logical N/T view of a TLR matrix
│   ├── precision.jl      compute policy and output scaling
│   └── arena.jl          result-independent bump arena
├── dense_result/
│   ├── dense_operand.jl  standalone dense operand and dense-diagonal-tile views
│   ├── workspace.jl      DenseGemmWorkspace
│   ├── runs.jl           shared schedule/task/prepared-run execution
│   ├── compressed_ftlr/  CompressedFTLRMatrix logical-N/T accessors, metadata, and lowering
│   ├── dense_diagonal.jl dense-diagonal cross terms
│   └── driver.jl         dense-destination public dispatch
└── padded_result/
    ├── operands.jl       padded factor-panel views
    ├── workspace.jl      TLRGemmWorkspace and ARA arenas
    ├── *coupling.jl      implicit product operators
    ├── *schedule.jl      rolling ARA scheduler
    └── driver.jl         padded-destination public dispatch
```

`dense_result.jl` and `padded_result.jl` are the only subsystem include files.
`TLRmodule.jl` includes the small shared layer first and then these two
subsystems.

## Dense accumulation

For compressed operands

```text
A[i,k] = U[i,k] V[i,k]'
B[k,j] = W[k,j] Z[k,j]'
```

the full product has a shared contraction and two possible folds:

```text
S[i,k,j] = V[i,k]' W[k,j]

FoldRight: T = S Z'   then C = U T
FoldLeft:  T = U S    then C = T Z'
```

`compressed_ftlr/three_stage.jl` emits those three stages. Rank metadata and
workspace costs select rectangular output runs and one fold per run.

The products with one dense operand are fixed-fold specializations of the same
lowering:

```text
compressed × dense  = FoldRight with W=B and the S Z' stage elided
dense × compressed  = FoldLeft  with V'=A and the U S stage elided
```

`compressed_ftlr/two_stage.jl` therefore emits `DenseResultRunTasks` with two
stages and the same terminal-stage semantics used by the three-stage product.
Both paths share:

- `DenseResultRun` schedule regions and fold names;
- grouped-task preparation and cleanup;
- fallback detection and pointer-mode management;
- beta-only output regions;
- terminal `alpha`/`beta` substitution;
- one-shot execution through reusable symbolic analysis.

A workspace too small for one fused two-stage rank stack uses the compressed-only
tilewise fallback. This fallback exists solely to preserve the low-workspace API;
there is no generic PaddedFTLR × dense implementation.

## PaddedFTLR accumulation

`gemm!(C::PaddedFTLRMatrix, A::PaddedFTLRMatrix, B::PaddedFTLRMatrix)` never
materializes a dense output tile. It exposes the product as implicit factor-list
operators and runs a blocked adaptive randomized approximation into `C`'s fixed
rank capacity. See [`padded_result/README.md`](padded_result/README.md) and
[`padded_result/algorithm.tex`](padded_result/algorithm.tex).
