export compress!

include("workspace.jl")

# Dense-diagonal copy ----------------------------------------------------------

# Kernels specific to dense-to-TLR compression. Shared orthogonalization, norm,
# and factor-pruning kernels live in `src/TLR/numerics/`.
@kernel function _copy_diag_kernel!(D::AbstractArray{T,3},
                                    A::AbstractMatrix{T},
                                    tile_m::Int,
                                    tile_n::Int,
) where {T}
    row, col, batch = @index(Global, NTuple)
    p0 = (batch - 1) * tile_m + 1
    q0 = (batch - 1) * tile_n + 1
    @inbounds D[row, col, batch] = A[p0+row-1, q0+col-1]
end

"""
    _copy_diagonal_from_dense!(A_tlr, A) -> A_tlr

Populate `A_tlr`'s dense diagonal storage from the corresponding tiles of the
dense matrix `A`.
"""
function _copy_diagonal_from_dense!(A_tlr::TLRDenseDiagMatrix{<:Any,T},
                                    A::AbstractMatrix{T},
) where {T}
    n_full_diag = _nfull_diag_tiles(A_tlr)
    bm, bn = nominal_tile_size(A_tlr)
    _copy_diag_kernel!(A_tlr.backend)(
        A_tlr.D, A, bm, bn;
        ndrange=(bm, bn, n_full_diag),
    )
    if size(A_tlr.D_corner, 3) != 0
        tile_k = ndiag_tiles(A_tlr)
        tm, tn = tile_size(A_tlr, tile_k, tile_k)
        copyto!(view(A_tlr.D_corner, 1:tm, 1:tn, 1),
                _dense_tile_view(A, A_tlr, tile_k, tile_k),
        )
    end
    return _set_dense_diagonal_diagnostics!(A_tlr)
end

# Tile-batch compression core --------------------------------------------------

abstract type TileSource{T} end

@inline _ntiles(src::TileSource) = length(src.tiles)

# Off-diagonal tiles carved from a dense matrix (today's `compress!` path).
struct DenseTiles{T,AT<:AbstractMatrix{T},TV<:AbstractVector,CV} <: TileSource{T}
    A::AT           # dense source matrix
    tiles::TV       # per-tile views into A (gemm operands)
    p0s::CV         # per-tile row origin (device Int32) — for the norm kernel
    q0s::CV         # per-tile col origin (device Int32)
    tm::Int         # tile rows
    tn::Int         # tile cols
end

# out[k] = ‖A_tile_k‖²_F.
function _tile_norms_sq!(out, src::DenseTiles)
    n = _ntiles(src)
    n == 0 && return out
    backend = get_backend(src.A)
    W, _, NT = _norm_launch(backend, src.tn)
    _tile_norm_sq_kernel!(backend, NT)(out, src.A, src.p0s, src.q0s, src.tm, src.tn,
        Val{W}(), Val{NT}(); ndrange=(NT * n,), workgroupsize=NT)
    return out
end

# A packed [tm, tn, ntiles] batch of dense tiles (e.g. gemm intermediates).
struct PackedTiles{T,PT<:AbstractArray{T,3},TV<:AbstractVector} <: TileSource{T}
    data::PT        # [tm, tn, ntiles]
    tiles::TV       # per-slab views (gemm operands)
end
PackedTiles(data::AbstractArray{<:Any,3}) =
    PackedTiles(data, [view(data,:,:,k) for k in axes(data, 3)])

function _tile_norms_sq!(out, src::PackedTiles)
    return batch_frobenius_norms_sq!(out, src.data)
end

# Q = A·Ω and V = Aᴴ·Q, batched over tiles
@inline _sketch!(Q_tiles, src::TileSource{T}, Ω_tiles) where {T} =
    gemm_batched!('N', 'N', one(T), src.tiles, Ω_tiles, zero(T), Q_tiles)
@inline _cosketch!(V_tiles, src::TileSource{T}, Q_tiles) where {T} =
    gemm_batched!(_adjoint_blas_char(T), 'N', one(T), src.tiles, Q_tiles, zero(T), V_tiles)

"""
    compress_tiles!(src, cat; eps_sq, rel) -> cat

Randomized-sketch compression (randQB_EI) of the tile batch described by `src`
into the workspace `cat`: writes the retained factors into `cat.U`/`cat.V` and the
per-tile rank / squared error into `cat.ranks_local` / `cat.norm_err_sq`. Input-
agnostic — see [`TileSource`](@ref). Degenerates to rank 0 when `cat.R_keep == 0`.
"""
function compress_tiles!(src::TileSource{T}, cat::CompressCategoryWorkspace; eps_sq::Float64, rel::Bool) where {T}
    _ntiles(src) == 0 && return cat

    if cat.R_keep == 0   # maxrank == 0: every tile degenerates to rank 0
        _tile_norms_sq!(cat.norm_err_sq, src)
        fill!(cat.ranks_local, zero(eltype(cat.ranks_local)))
        return cat
    end

    # Step 1: range sampling  Q = A·Ω  (Ω drawn into V_T; step 3 overwrites it)
    Random.randn!(cat.V_T)
    _sketch!(cat.Q_tiles, src, cat.V_tiles)

    # Step 2: form/factor each Gram matrix in high precision, then apply its
    # triangular factor to Q with a work-precision TRSM. Ω is dead after the
    # sketch, so its leading S×S rows hold the triangular factors.
    R_work = view(cat.V_T, 1:cat.S, :, :)
    mixed_cholqr2_basis!(cat.Q_T, cat.Y_hi, cat.G_hi, R_work, cat.R_tiles,
        cat.Q_tiles, cat.shift_mult)

    # Step 3: co-range  V = Aᴴ·Q  (overwrites the Ω we no longer need)
    _cosketch!(cat.V_tiles, src, cat.Q_tiles)

    # Step 4: the final Gram matrix is dead, so its first element in each slab
    # stores ‖A‖² and is then overwritten by the achieved squared error.
    _tile_norms_sq!(cat.norm_err_sq, src)

    # Step 5: rank detection + truncation (fused SMEM kernel, EI-corrected budget)
    delta_floor = Float64(_cholqr_shift_coeff(eltype(cat.G_hi), size(cat.Y_hi, 1), cat.S))
    prune_randqb_columns!(cat.U, cat.V, cat.ranks_local, cat.norm_err_sq,
        cat.S, cat.R_keep, eps_sq, rel, delta_floor)
    return cat
end

# Compress one off-diagonal tile category from the dense matrix `A`: wrap its tiles
# as a `DenseTiles` source and run the input-agnostic core.
function _compress_category!(
    A_tlr::AbstractTLRMatrix,
    A::AbstractMatrix,
    cat::CompressCategoryWorkspace,
    eps_sq::Float64,
    rel::Bool,
)
    n = size(cat.U, 3)
    n == 0 && return cat
    tiles = [_dense_tile_view(A, A_tlr, region_tile_coords(A_tlr, cat.region, k)...) for k in 1:n]
    src = DenseTiles(A, tiles, cat.p0s, cat.q0s, size(cat.Q_T, 1), size(cat.V_T, 1))
    return compress_tiles!(src, cat; eps_sq, rel)
end

# ─── Storage helpers ──────────────────────────────────────────────────────────

# Scatter one category's local ranks / squared errors back into the global
# A_tlr.ranks / A_tlr.resid (converting squared error to a Frobenius residual).
function _store_category_results!(A_tlr::AbstractTLRMatrix, cat::CompressCategoryWorkspace)
    n = size(cat.U, 3)
    n == 0 && return
    rk_host = cat.ranks_local isa Vector ? cat.ranks_local : Array(cat.ranks_local)
    err_host = cat.norm_err_sq isa Vector ? cat.norm_err_sq : Array(cat.norm_err_sq)
    @inbounds for (k, rank_idx) in enumerate(cat.rank_indices)
        A_tlr.ranks[rank_idx] = rk_host[k]
        A_tlr.resid[rank_idx] = sqrt(max(Float64(real(err_host[k])), 0.0))
    end
end

# ─── Orchestration ────────────────────────────────────────────────────────────

# Compress all tile categories and scatter their results into A_tlr. On GPU
# each category runs on its own stream (overlap) and is synced before storing; on
# CPU they run sequentially.
function _compress_all_categories!(
    A_tlr::AbstractTLRMatrix{<:Any,T},
    A::AbstractMatrix{T},
    ws::CompressWorkspace,
    eps_sq::Float64,
    rel::Bool,
) where {T}
    cats = ws.cats
    backend = get_backend(A_tlr)
    if backend isa KernelAbstractions.CPU
        for cat in cats
            _compress_category!(A_tlr, A, cat, eps_sq, rel)
        end
    else
        for (cat, stream) in zip(cats, ws.streams)
            with_stream(backend, stream) do
                _compress_category!(A_tlr, A, cat, eps_sq, rel)
            end
        end
        for stream in ws.streams
            sync_stream(backend, stream)
        end
    end
    for cat in cats
        _store_category_results!(A_tlr, cat)
    end
    A_tlr
end

"""
    compress!(A_tlr, A [, ws]; tol=0.0, rel=false)

Compress dense matrix `A` into the TLR container `A_tlr` in-place.

Per-tile effective ranks are detected via greedy V-column-norm thresholding
against an error-indicator-corrected budget and stored in `ranks(A_tlr)`; the
estimated per-tile Frobenius error lands in `residuals(A_tlr)`.  The indicator
(`‖A_tile‖²_F − ‖V‖²_F`, à la randQB_EI) accounts for the range-capture error
of the sketch, so a tile whose spectrum does not fit within `maxrank` keeps
full rank and reports a residual above `tol` instead of silently claiming
convergence — check `residuals` to route such tiles to dense storage or a
higher-rank second pass.

Factor arrays are updated in-place; call `alloc_workspace` once to amortise
device allocations across repeated calls:

    ws = alloc_workspace(A_tlr)
    for A in matrices
        compress!(A_tlr, A, ws; tol=1f-3)
    end

## Keywords

`tol` — per-tile Frobenius error budget (default `0.0`). The squared budget is
floored at the orthogonality floor of the shifted high-precision CholQR basis.

`rel` — when `true`, the budget for each tile is `tol * ‖A_tile‖_F` instead of
the absolute `tol`.

`maxrank` is both the output capacity and the sketch capacity. Reserve any
desired randomized-range buffer in `maxrank` itself.

The sketch basis is orthogonalised with two shifted Cholesky-QR passes in
higher precision.
"""
compress!(A_tlr::AbstractTLRMatrix{<:Any,T}, A::AbstractMatrix{T}; kwargs...) where {T} =
    compress!(A_tlr, A, alloc_workspace(A_tlr); kwargs...)

function compress!(A_tlr::TLRDenseDiagMatrix{<:Any,T}, A::AbstractMatrix{T},
    ws::CompressWorkspace;
    tol::Real=0.0, rel::Bool=false) where {T}

    size(A) == (A_tlr.m, A_tlr.n) ||
        throw(DimensionMismatch("A dimensions must match A_tlr"))
    A_tlr.m == A_tlr.n ||
        throw(ArgumentError("compress! currently requires square matrices"))
    tol >= 0 || throw(ArgumentError("tol must be >= 0"))

    _copy_diagonal_from_dense!(A_tlr, A)

    eps_sq = Float64(tol)^2
    _compress_all_categories!(A_tlr, A, ws, eps_sq, rel)

    A_tlr
end

function compress!(A_tlr::TLRMatrix{<:Any,T}, A::AbstractMatrix{T},
    ws::CompressWorkspace;
    tol::Real=0.0, rel::Bool=false) where {T}

    size(A) == (A_tlr.m, A_tlr.n) ||
        throw(DimensionMismatch("A dimensions must match A_tlr"))
    tol >= 0 || throw(ArgumentError("tol must be >= 0"))

    eps_sq = Float64(tol)^2
    _compress_all_categories!(A_tlr, A, ws, eps_sq, rel)

    A_tlr
end
