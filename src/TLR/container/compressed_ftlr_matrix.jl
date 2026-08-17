"""
    CompressedFTLRMatrix

Exact-rank, block-compressed low-rank storage for a regular tile grid.  Every
factor is a single packed allocation; `offsets` gives the scalar span of each
tile in that factor's own tile order.  The outer and inner factors may use
different orders, which is what permits row-packed `W` and column-packed `Z`.
"""
struct CompressedFTLRPackedFactors{AT<:AbstractVector,O<:TileOrderStyle}
    data::AT
    offsets::Vector{Int}                 # scalar offsets, 1-based, length ntile + 1
    order::O
    leading_dimensions::Vector{Int}
    logical_dimensions::Vector{Int}
    dimension_axis::Symbol
    qm::Int
    qn::Int
end

struct CompressedFTLRMatrix{BackendT<:Backend,T,AT<:AbstractVector{T},RankT<:Integer,
                  OuterT<:TileOrderStyle,InnerT<:TileOrderStyle,
                  OrderT<:TileOrderStyle} <: AbstractTLRMatrix{BackendT,T,OrderT}
    backend::BackendT
    order::OrderT                        # diagnostic rank-vector order
    m::Int
    n::Int
    nominal_tile_size::NTuple{2,Int}
    tail_tile_size::NTuple{2,Int}
    outer::CompressedFTLRPackedFactors{AT,OuterT}
    inner::CompressedFTLRPackedFactors{AT,InnerT}
    ranks::Vector{RankT}
    execution_ranks::Vector{RankT}
    execution_rank_policy::Symbol
    resid::Vector{Float64}
    maxrank::Int
    execution_maxrank::Int
end

@inline function _compressed_ftlr_rank(A::CompressedFTLRMatrix, i::Int, j::Int)
    return Int(A.ranks[_rank_index(A, i, j)])
end
@inline function _compressed_ftlr_execution_rank(A::CompressedFTLRMatrix, i::Int, j::Int)
    return Int(A.execution_ranks[_rank_index(A, i, j)])
end
@inline _compressed_ftlr_logical_coords(::CompressedFTLRMatrix, i::Int, j::Int) = (i, j)

@inline function _compressed_ftlr_slot(f::CompressedFTLRPackedFactors, i::Int, j::Int)
    return tile_linear_index(f.order, f.qm, f.qn, i, j)
end

@inline function _compressed_ftlr_factor_view(f::CompressedFTLRPackedFactors, stored_r::Int,
                                               visible_r::Int, i::Int, j::Int)
    axis_index = f.dimension_axis === :row ? i : j
    logical_rows = f.logical_dimensions[axis_index]
    stored_r == 0 && return reshape(view(f.data, 1:0), logical_rows, 0)
    slot = _compressed_ftlr_slot(f, i, j)
    first = f.offsets[slot]
    last = f.offsets[slot + 1] - 1
    ld = f.leading_dimensions[axis_index]
    packed = reshape(view(f.data, first:last), ld, stored_r)
    return view(packed, 1:logical_rows, 1:visible_r)
end

@inline compressed_ftlr_outer(A::CompressedFTLRMatrix, i::Int, j::Int) =
    _compressed_ftlr_factor_view(A.outer, _compressed_ftlr_execution_rank(A, i, j),
                                 _compressed_ftlr_rank(A, i, j), i, j)
@inline compressed_ftlr_inner(A::CompressedFTLRMatrix, i::Int, j::Int) =
    _compressed_ftlr_factor_view(A.inner, _compressed_ftlr_execution_rank(A, i, j),
                                 _compressed_ftlr_rank(A, i, j), i, j)
@inline compressed_ftlr_execution_outer(A::CompressedFTLRMatrix, i::Int, j::Int) =
    _compressed_ftlr_factor_view(A.outer, _compressed_ftlr_execution_rank(A, i, j),
                                 _compressed_ftlr_execution_rank(A, i, j), i, j)
@inline compressed_ftlr_execution_inner(A::CompressedFTLRMatrix, i::Int, j::Int) =
    _compressed_ftlr_factor_view(A.inner, _compressed_ftlr_execution_rank(A, i, j),
                                 _compressed_ftlr_execution_rank(A, i, j), i, j)
@inline compressed_ftlr_outer_order(A::CompressedFTLRMatrix) = A.outer.order
@inline compressed_ftlr_inner_order(A::CompressedFTLRMatrix) = A.inner.order
@inline execution_ranks(A::CompressedFTLRMatrix) = A.execution_ranks
@inline execution_maxrank(A::CompressedFTLRMatrix) = A.execution_maxrank
@inline execution_rank_policy(A::CompressedFTLRMatrix) = A.execution_rank_policy

"""
    _compressed_ftlr_uniform_view(f::CompressedFTLRPackedFactors)

Reinterpret `f`'s packed storage as a dense `[extent, capacity, qm*qn]` array,
where the third index is `f`'s own tile order
(`tile_linear_index(f.order, qm, qn, i, j)`). Valid only when every tile shares
the same stored capacity and logical extent along `f`'s axis — i.e. `f`
belongs to a reserved-capacity, regular-grid `CompressedFTLRMatrix` (as
`padded_result/` constructs for its canonical TLR-output GEMM). Throws if that
uniformity does not hold, rather than silently returning a misinterpreted view.
"""
function _compressed_ftlr_uniform_view(f::CompressedFTLRPackedFactors)
    ntiles = f.qm * f.qn
    ntiles == 0 && return reshape(view(f.data, 1:0), 0, 0, 0)
    ld = f.leading_dimensions[1]
    extent = f.logical_dimensions[1]
    (all(==(ld), f.leading_dimensions) && all(==(extent), f.logical_dimensions)) ||
        throw(ArgumentError("packed factors are not on a regular grid (nonuniform tile extent)"))
    stride = f.offsets[2] - f.offsets[1]
    stride % ld == 0 || throw(ArgumentError(
        "packed factor stride is not a multiple of its leading dimension"))
    capacity = stride ÷ ld
    all(k -> f.offsets[k + 1] - f.offsets[k] == stride, 1:ntiles) || throw(ArgumentError(
        "packed factors do not have a uniform per-tile capacity; not a reserved-capacity container"))
    length(f.data) == stride * ntiles || throw(ArgumentError(
        "packed factor storage size is inconsistent with a uniform-capacity layout"))
    return view(reshape(f.data, ld, capacity, ntiles), 1:extent, :, :)
end

@inline function _compressed_ftlr_execution_rank(r::Integer, policy::Symbol=:q8)
    r = Int(r)
    r >= 0 || throw(ArgumentError("CompressedFTLR ranks must be nonnegative"))
    iszero(r) && return 0
    policy === :exact && return r
    policy === :q8 && return cld(r, 8) * 8
    policy === :q16 && return cld(r, 16) * 16
    policy === :pow2 && return max(8, nextpow(2, r))
    throw(ArgumentError("unknown CompressedFTLR execution-rank policy $policy; " *
                        "expected :exact, :q8, :q16, or :pow2"))
end

"""
    _validate_compressed_ftlr_tile_alignment(backend, T, bm, bn)

Reject nominal tile sizes that would place a dense-output tile boundary on a
misaligned address for `backend`'s grouped-GEMM primitive, if it has such a
requirement. Dispatches on `backend`; the fallback here is a no-op. A backend
whose grouped-GEMM kernels share cuBLAS's tensor-core alignment fault (see
[`_validate_compressed_ftlr_tile_alignment_cuda`](@ref) for why CUDA needs
this) must add its own `_validate_compressed_ftlr_tile_alignment(::TheBackend,
::Type, bm, bn)` method — there is no capability query that forces one
automatically, so a new grouped-GEMM backend silently gets the permissive
default until someone opts it in.
"""
@inline _validate_compressed_ftlr_tile_alignment(backend, ::Type, bm::Int, bn::Int) = nothing

"""
    _validate_compressed_ftlr_tile_alignment_cuda(T, bm, bn)

Stage 3 writes `view(C, (i-1)*bm+1 : ..., :)`, whose byte offset is
`(i-1)*bm*sizeof(T)` and is therefore independent of the leading dimension. When
cuBLAS selects a tensor-core kernel for `cublasGemmGroupedBatchedEx` it issues
vectorized loads that require 16-byte alignment and hard-faults
(`CUDA_ERROR_MISALIGNED_ADDRESS`) otherwise. The grouped entry point takes its
pointer table in *device* memory, so it cannot inspect the pointers host-side and
has no opportunity to dispatch an alignment-safe kernel the way ordinary
`cublasGemmEx` does. Measured on H100: FP16, BF16 and TF32 all fault; FP32 under
`CUBLAS_COMPUTE_32F` and FP64 do not, because those never reach a tensor-core
kernel.

Requiring both extents to be multiples of [`gemm_alignment_quantum`](@ref) makes
every tile boundary aligned by construction. The trailing tile may still be
short: nothing starts after it, so its extent cannot misalign anything.
"""
function _validate_compressed_ftlr_tile_alignment_cuda(::Type{T}, bm::Int, bn::Int) where {T}
    q = gemm_alignment_quantum(T)
    (bm % q == 0 && bn % q == 0) || throw(ArgumentError(
        "CompressedFTLR nominal tile size ($bm, $bn) is not 16-byte aligned for $T: " *
        "both extents must be multiples of $q. Grouped-GEMM tensor-core kernels " *
        "fault on misaligned tile boundaries; use " *
        "($(cld(bm, q) * q), $(cld(bn, q) * q)) instead."))
    return nothing
end

@inline function _compressed_ftlr_offsets(order::TileOrderStyle, qm::Int, qn::Int,
                                leading_dimension_at, rank_at)
    offsets = Vector{Int}(undef, qm * qn + 1)
    offsets[1] = 1
    @inbounds for slot in 1:(qm * qn)
        i, j = inverse_tile_index(order, qm, qn, slot)
        offsets[slot + 1] = offsets[slot] + leading_dimension_at(i, j) * rank_at(i, j)
    end
    return offsets
end

function _compressed_ftlr_rank_vector(order::TileOrderStyle, ranks_in::AbstractMatrix,
                           ::Type{RankT}) where {RankT<:Integer}
    qm, qn = size(ranks_in)
    ranks_out = Vector{RankT}(undef, qm * qn)
    @inbounds for j in 1:qn, i in 1:qm
        ranks_out[tile_linear_index(order, qm, qn, i, j)] = RankT(ranks_in[i, j])
    end
    return ranks_out
end

"""
    CompressedFTLRMatrix(backend, T, m, n, tile_size, ranks;
               outer_order=TileRowMajor, inner_order=TileColMajor,
               execution_rank_policy=:q8,
               execution_ranks=nothing,
               rank_type=Int32)

Allocate an exact-rank CompressedFTLR matrix. The public ranks remain exact,
while physical factor widths follow `execution_rank_policy` and are padded with
zeros as necessary. `:q8` is the default aligned execution policy; `:exact`,
`:q16`, and `:pow2` are useful for measurement. A regular nominal grid may have
one trailing row and/or column tile. Passing an explicit `execution_ranks`
matrix bypasses the policy calculation. This is intended for containers that
need reserved factor capacity while their logical ranks are still zero.
"""

function CompressedFTLRMatrix(backend::Backend, ::Type{T}, m::Int, n::Int,
                    tile_size::NTuple{2,Int}, ranks_in::AbstractMatrix{<:Integer};
                    outer_order=TileRowMajor, inner_order=TileColMajor,
                    execution_rank_policy::Symbol=:q8,
                    execution_ranks=nothing,
                    rank_type::Type{<:Integer}=Int32) where {T}
    bm, bn = tile_size
    m > 0 && n > 0 && bm > 0 && bn > 0 ||
        throw(ArgumentError("m, n, and tile dimensions must be positive"))
    _validate_compressed_ftlr_tile_alignment(backend, T, bm, bn)
    qm, qn = cld(m, bm), cld(n, bn)
    size(ranks_in) == (qm, qn) ||
        throw(DimensionMismatch("ranks must be a $qm × $qn matrix"))
    all(>=(0), ranks_in) || throw(ArgumentError("CompressedFTLR ranks must be nonnegative"))
    rowdims = [min(bm, m - (i - 1) * bm) for i in 1:qm]
    coldims = [min(bn, n - (j - 1) * bn) for j in 1:qn]
    @inbounds for j in 1:qn, i in 1:qm
        ranks_in[i, j] <= min(rowdims[i], coldims[j]) ||
            throw(ArgumentError("CompressedFTLR rank at ($i, $j) exceeds its logical tile extent"))
    end

    outer_style = _order_instance(outer_order)
    inner_style = _order_instance(inner_order)
    rank_style = outer_style
    execution_rank_grid, stored_policy = if execution_ranks === nothing
        # Validate even for an all-zero rank grid, then capture the selected policy.
        _compressed_ftlr_execution_rank(0, execution_rank_policy)
        (map(r -> _compressed_ftlr_execution_rank(r, execution_rank_policy), ranks_in),
         execution_rank_policy)
    else
        size(execution_ranks) == (qm, qn) ||
            throw(DimensionMismatch("execution_ranks must be a $qm × $qn matrix"))
        all(>=(0), execution_ranks) ||
            throw(ArgumentError("CompressedFTLR execution ranks must be nonnegative"))
        @inbounds for j in 1:qn, i in 1:qm
            execution_ranks[i, j] >= ranks_in[i, j] || throw(ArgumentError(
                "CompressedFTLR execution rank at ($i, $j) is smaller than its logical rank"))
        end
        (Matrix{Int}(execution_ranks), :explicit)
    end
    execution_rank_at = (i, j) -> execution_rank_grid[i, j]
    # A multiple-of-eight leading dimension plus a multiple-of-eight execution
    # rank keeps every packed factor base 16-byte aligned for the default and
    # bucketed policies, including ragged boundary tiles.
    uld = [cld(x, 8) * 8 for x in rowdims]
    vld = [cld(x, 8) * 8 for x in coldims]
    uoffsets = _compressed_ftlr_offsets(
        outer_style, qm, qn, (i, j) -> uld[i], execution_rank_at)
    voffsets = _compressed_ftlr_offsets(
        inner_style, qm, qn, (i, j) -> vld[j], execution_rank_at)
    udata = zeros(backend, T, uoffsets[end] - 1)
    vdata = zeros(backend, T, voffsets[end] - 1)
    outer = CompressedFTLRPackedFactors(udata, uoffsets, outer_style, uld, rowdims, :row, qm, qn)
    inner = CompressedFTLRPackedFactors(vdata, voffsets, inner_style, vld, coldims, :col, qm, qn)
    rankvec = _compressed_ftlr_rank_vector(rank_style, ranks_in, rank_type)
    execution_rankvec = _compressed_ftlr_rank_vector(
        rank_style, execution_rank_grid, rank_type)
    resid = Base.zeros(Float64, qm * qn)
    return CompressedFTLRMatrix{typeof(backend),T,typeof(udata),rank_type,typeof(outer_style),
                      typeof(inner_style),typeof(rank_style)}(
        backend, rank_style, m, n, tile_size, (m % bm, n % bn), outer, inner,
        rankvec, execution_rankvec, stored_policy, resid,
        isempty(rankvec) ? 0 : maximum(rankvec),
        isempty(execution_rankvec) ? 0 : maximum(execution_rankvec))
end

function CompressedFTLRMatrix(backend::Backend, ::Type{T}, m::Int, n::Int, b::Int,
                    ranks_in::AbstractMatrix{<:Integer}; kwargs...) where {T}
    return CompressedFTLRMatrix(backend, T, m, n, (b, b), ranks_in; kwargs...)
end

"""Pack a padded `PaddedFTLRMatrix` into exact-rank CompressedFTLR storage."""
function pack_compressed_ftlr(A::PaddedFTLRMatrix{<:Any,T}; outer_order=TileRowMajor,
                   inner_order=TileColMajor, execution_rank_policy::Symbol=:q8,
                   rank_type::Type{<:Integer}=Int32) where {T}
    qm, qn = grid_size(A)
    rank_grid = Matrix{Int}(undef, qm, qn)
    @inbounds for j in 1:qn, i in 1:qm
        rank_grid[i, j] = Int(ranks(A)[_rank_index(A, i, j)])
    end
    B = CompressedFTLRMatrix(get_backend(A), T, A.m, A.n, nominal_tile_size(A), rank_grid;
                   outer_order, inner_order, execution_rank_policy, rank_type)
    @inbounds for j in 1:qn, i in 1:qm
        r = rank_grid[i, j]
        r == 0 && continue
        U, V = get_factors(A, i, j)
        copyto!(compressed_ftlr_outer(B, i, j), U)
        copyto!(compressed_ftlr_inner(B, i, j), V)
    end
    B.resid .= residuals(A)
    return B
end

@inline function get_factors(A::CompressedFTLRMatrix, i::Int, j::Int)
    return compressed_ftlr_outer(A, i, j), compressed_ftlr_inner(A, i, j)
end

@inline lowrank_regions(::CompressedFTLRMatrix) = (_INTERIOR,)
@inline outer_factors(A::CompressedFTLRMatrix, ::InteriorRegion) = A.outer
@inline inner_factors(A::CompressedFTLRMatrix, ::InteriorRegion) = A.inner
@inline region_tile_count(A::CompressedFTLRMatrix, ::InteriorRegion) = length(A.ranks)
@inline region_tile_coords(A::CompressedFTLRMatrix, ::InteriorRegion, k::Int) =
    inverse_tile_index(A.order, grid_size(A)..., k)
