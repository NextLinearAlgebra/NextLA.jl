"""
    GemmStage

Singleton supertype for the three stages of the off-diagonal product
`O_A O_B = U (V' W) Z'` summed over the contraction tile index `k`.
"""
abstract type GemmStage end

"""Stage 1: `S_ikj = V_ik' W_kj`."""
struct Stage1 <: GemmStage end

"""Stage 2: `T_ikj = S_ikj Z_kj'`."""
struct Stage2 <: GemmStage end

"""Stage 3: `C_ij += U_ik T_ikj`, reducing over `k`."""
struct Stage3 <: GemmStage end

"""
    Stride1Axis{Ax}

Layout trait naming the logical tile axis stored contiguously.
For the left operand `A`, `Ax` is `:i` or `:k`; for the right operand `B`,
`Ax` is `:k` or `:j`.
"""
struct Stride1Axis{Ax} end

# Order is the 3rd parameter of `AbstractTLRMatrix`, so these cover both container types.
@inline stride1_axis_left(::AbstractTLRMatrix{<:Any,<:Any,TileColMajor}) = Stride1Axis{:i}()
@inline stride1_axis_left(::AbstractTLRMatrix{<:Any,<:Any,TileRowMajor}) = Stride1Axis{:k}()

@inline stride1_axis_right(::AbstractTLRMatrix{<:Any,<:Any,TileColMajor}) = Stride1Axis{:k}()
@inline stride1_axis_right(::AbstractTLRMatrix{<:Any,<:Any,TileRowMajor}) = Stride1Axis{:j}()

"""
    KAxisSchedule

Trait binding the reduction tile index `k` onto a hardware iteration axis.
`BPanelAxis` carries the right operand's stride-1 axis (`:k` or `:j`).
"""
abstract type KAxisSchedule end

"""`k` is bound onto the GEMM's shared **K** dimension: one accumulating GEMM
sums over `k`, giving write-once output rows."""
struct KAsGemmK{BPanelAxis} <: KAxisSchedule end

"""`k` is bound onto an outer serial loop that accumulates into `C`."""
struct KAsSerialLoop{BPanelAxis} <: KAxisSchedule end

@inline k_axis_schedule(::Stride1Axis{:k}, ::Stride1Axis{BPanelAxis}) where {BPanelAxis} =
    KAsGemmK{BPanelAxis}()

@inline k_axis_schedule(::Stride1Axis{:i}, ::Stride1Axis{BPanelAxis}) where {BPanelAxis} =
    KAsSerialLoop{BPanelAxis}()

"""
    FreeAxisSchedule

Trait binding the spatial tile indices `i`, `j` onto Stage 1's GEMM operand
dimensions (M, N) or leaving them as batch axes.
"""
abstract type FreeAxisSchedule end

"""`i` and `j` stay batch axes: Stage 1 is a pointer batch of tile GEMMs."""
struct FreeAsBatch <: FreeAxisSchedule end

"""`i` is bound onto the Stage 1 GEMM **M** dimension."""
struct IAsGemmM <: FreeAxisSchedule end

"""`i` and `j` are bound onto the Stage 1 GEMM **M** and **N** dimensions."""
struct IJAsGemmMN <: FreeAxisSchedule end

"""`j` is bound onto the Stage 1 GEMM **N** dimension (`i` stays a batch axis)."""
struct JAsGemmN <: FreeAxisSchedule end

# Row family: B stride-1 `:j` makes `W_k,:` contiguous over `j`, so Stage 1 fuses
# `j` into N (`JAsGemmN`); B stride-1 `:k` cannot, so it stays tilewise.
@inline free_axis_schedule(::KAsGemmK{:j}) = JAsGemmN()
@inline free_axis_schedule(::KAsGemmK{:k}) = FreeAsBatch()
@inline free_axis_schedule(::KAsSerialLoop{:k}) = IAsGemmM()
@inline free_axis_schedule(::KAsSerialLoop{:j}) = IJAsGemmMN()

"""
    GridKind

Interior tile-grid enumeration policy of an operand. `SkipDiag` is the dense-diagonal
interior (the diagonal tile is stored separately, so it is excluded from the low-rank
grid); `FullGrid` is the fully low-rank interior (every tile present). The staged core
consults this to decide how tile rows enumerate their contraction tiles, so one set of
`execute_stage!` methods serves both container types; for `FullGrid` the skip helpers
fold to identities. See `panel.jl` for the policy bodies.
"""
abstract type GridKind end
struct SkipDiag <: GridKind end   # dense-diag interior: diagonal excluded
struct FullGrid <: GridKind end   # fully low-rank: every tile present

"""
    FoldSide

Which outer low-rank factor is folded *last* (stacked in Stage 3) when lowering a
contraction term `A_ik B_kj = U_ik (V_ik' W_kj) Z_kj'` to the three staged GEMMs.

`FoldRight` (the default) forms `T = S Z'` in Stage 2 and stacks A's `U` in Stage 3
(`C += Σ_k U_ik T_ikj`, one write-once GEMM when A's k-axis is stride-1, i.e. A is
`TileRowMajor`). `FoldLeft` forms `T' = U S` in Stage 2 and stacks B's `Z` in Stage 3
(`C += Σ_k T'_ikj Z_kj'`, write-once when B's k-axis is stride-1, i.e. B is
`TileColMajor`).

The choice is dictated by storage layout — it is made so the reduction becomes a
write-once fused GEMM without transposing either operand; the resulting change in
intermediate size (`T` is `r_A×b_n`, `T'` is `b_m×r_B`) is a documented consequence,
used only to break ties when both or neither layout yields write-once.
"""
abstract type FoldSide end
struct FoldRight <: FoldSide end   # stack A's U (write-once ⟺ A TileRowMajor)
struct FoldLeft  <: FoldSide end   # stack B's Z (write-once ⟺ B TileColMajor)
