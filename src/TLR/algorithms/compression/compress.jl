export compress!

include("kernels.jl")
include("workspace.jl")
include("algorithm.jl")

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
