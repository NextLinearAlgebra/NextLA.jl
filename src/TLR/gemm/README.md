# TLR GEMM architecture

The destination representation is the implementation boundary.

| Destination | Product | Implementation |
|---|---|---|
| `AbstractMatrix` | compressed × compressed | `dense_result/` three-stage lowering |
| `AbstractMatrix` | compressed × dense, dense × compressed | `dense_result/` two-stage specializations |
| `AbstractMatrix` | dense-diagonal TLR products | compressed lowering plus block-diagonal updates |
| newly allocated `CompressedFTLRMatrix` | compressed × compressed | `compressed_result/` ARA and final packing |

Dense accumulation never depends on output-rank discovery. Compressed output
uses private fixed-width numerical staging because its packed offsets cannot be
known until ARA converges.

## Source layout

```text
gemm/
├── common/               precision policy and bump arena
├── dense_result/         dense accumulation
│   ├── compressed_ftlr/  rank metadata and two/three-stage lowering
│   ├── dense_diagonal.jl dense diagonal cross terms
│   └── driver.jl         public gemm! dispatch
└── compressed_result/    ARA compressed-output construction
    ├── workspace.jl      ARA arenas
    ├── run_coupling.jl   factor panels and implicit product operator
    ├── rolling_schedule.jl runtime admission and retirement
    └── driver.jl         allocation-returning gemm
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

See [`compressed_result/README.md`](compressed_result/README.md) and
[`compressed_result/algorithm.tex`](compressed_result/algorithm.tex).
