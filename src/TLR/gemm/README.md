# TLR GEMM architecture

The destination representation is the implementation boundary.

| Destination | Product | Implementation |
|---|---|---|
| `AbstractMatrix` | compressed × compressed | `dense_accumulation/` three-stage lowering |
| `AbstractMatrix` | compressed × dense, dense × compressed | `dense_accumulation/` two-stage specializations |
| `AbstractMatrix` | dense-diagonal TLR products | compressed lowering plus block-diagonal updates |
| newly allocated `CompressedFTLRMatrix` | compressed × compressed | `compressed_accumulation/` ARA and final packing |

Dense accumulation never depends on output-rank discovery. Compressed output
uses private fixed-width numerical staging because its packed offsets cannot be
known until ARA converges.

## Source layout

```text
gemm/
├── compute_policy.jl        GEMM tensor-core compute-mode policy
├── arena.jl                 result-independent bump allocator
├── dense_accumulation/       dense accumulation
│   ├── schedule.jl           rank metadata, cost formulas, and run scheduling
│   ├── three_stage.jl         compressed × compressed lowering + its analysis
│   ├── two_stage.jl            one-dense-operand specializations + their analysis
│   ├── dense_diagonal.jl        dense diagonal cross terms
│   └── driver.jl                 public gemm! dispatch and shared validation
└── compressed_accumulation/  ARA compressed-output construction
    ├── workspace.jl          ARA arenas
    ├── run_coupling.jl       factor panels and implicit product operator
    ├── rolling_schedule.jl   runtime admission and retirement
    └── driver.jl             allocation-returning gemm
```

## Dense accumulation

For `A[i,k] = U[i,k]V[i,k]'` and `B[k,j] = W[k,j]Z[k,j]'`, the shared
contraction is `S[i,k,j] = V[i,k]'W[k,j]`. The three-stage lowering chooses
between:

```text
FoldRight: T = S Z'   then C = U T
FoldLeft:  T = U S    then C = T Z'
```

The mixed products are exact two-stage special cases:

```text
compressed × dense  = FoldRight with the compressed-right stage elided
dense × compressed  = FoldLeft  with the compressed-left stage elided
```

They share run scheduling, grouped-task preparation, beta handling, symbolic
analysis, and low-workspace fallback with compressed × compressed.

## Compressed output

`gemm(A::CompressedFTLRMatrix, B::CompressedFTLRMatrix; maxrank, ...)` exposes
the product as an implicit factor-list operator and discovers output tile ranks
with blocked ARA. The current implementation accepts regular grids only. It
uses fixed-width storage privately, then allocates final complementary packed
factors and copies only active columns. There is no in-place compressed-output
`gemm!`, reserved-capacity public container, or `PaddedFTLRMatrix`.

See [`compressed_accumulation/README.md`](compressed_accumulation/README.md) and
[`compressed_accumulation/algorithm.tex`](compressed_accumulation/algorithm.tex).
