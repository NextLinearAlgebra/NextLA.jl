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

@inline stride1_axis_left(::TLRDenseDiagMatrix{<:Any,<:Any,<:Any,<:Any,TileColMajor}) = Stride1Axis{:i}()
@inline stride1_axis_left(::TLRDenseDiagMatrix{<:Any,<:Any,<:Any,<:Any,TileRowMajor}) = Stride1Axis{:k}()

@inline stride1_axis_right(::TLRDenseDiagMatrix{<:Any,<:Any,<:Any,<:Any,TileColMajor}) = Stride1Axis{:k}()
@inline stride1_axis_right(::TLRDenseDiagMatrix{<:Any,<:Any,<:Any,<:Any,TileRowMajor}) = Stride1Axis{:j}()

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
