export compress!

# ─── Kernels ──────────────────────────────────────────────────────────────────

@kernel function _copy_diag_kernel!(D::AbstractArray{T,3},
    A::AbstractMatrix{T}, m::Int, n::Int, tile_m::Int, tile_n::Int) where {T}
    row, col, b = @index(Global, NTuple)
    p0 = (b - 1) * tile_m + 1
    q0 = (b - 1) * tile_n + 1
    tm = min(tile_m, m - p0 + 1)
    tn = min(tile_n, n - q0 + 1)
    if row <= tm && col <= tn
        @inbounds D[row, col, b] = A[p0+row-1, q0+col-1]
    end
end

@kernel function _add_reg_diag_kernel!(G::AbstractArray{T,3}, reg::T) where {T}
    j, b = @index(Global, NTuple)
    @inbounds G[j, j, b] += reg
end

# Fused norm + descending sort + greedy threshold + column gather.
# One workgroup per off-diagonal tile; R threads per workgroup (one per column).
@kernel function _fused_truncate_kernel!(
    U_out::AbstractArray{T,3}, V_out::AbstractArray{T,3},
    U::AbstractArray{T,3}, V::AbstractArray{T,3},
    rk::AbstractVector, eps_sq::RT, ::Val{R}) where {T,RT,R}
    ob = @index(Group, Linear)
    j = @index(Local, Linear)
    norms_s = @localmem RT (R,)
    perm_s = @localmem Int32 (R,)
    k_buf = @localmem Int32 (1,)
    acc = zero(RT)
    @inbounds for row in 1:size(V, 1)
        x = V[row, j, ob]
        acc += real(conj(x) * x)
    end
    @inbounds norms_s[j] = acc
    @inbounds perm_s[j] = Int32(j)
    @synchronize
    if j == 1
        @inbounds for _ in 1:(R-1), i in 1:(R-1)
            ni = norms_s[i];
            ni1 = norms_s[i+1]
            if ni < ni1
                norms_s[i] = ni1;
                norms_s[i+1] = ni
                pi = perm_s[i];
                perm_s[i] = perm_s[i+1];
                perm_s[i+1] = pi
            end
        end
        budget = eps_sq;
        k_val = R
        @inbounds for idx in R:-1:1
            cost = norms_s[idx]
            cost <= budget || break
            budget -= cost;
            k_val -= 1
        end
        rk[ob] = eltype(rk)(k_val)
        k_buf[1] = Int32(k_val)
    end
    @synchronize
    k = Int(@inbounds k_buf[1])
    src = Int(@inbounds perm_s[j])
    bU = size(U, 1);
    bV = size(V, 1)
    if j <= k
        @inbounds for row in 1:bU
            ;
            U_out[row, j, ob] = U[row, src, ob];
        end
        @inbounds for row in 1:bV
            ;
            V_out[row, j, ob] = V[row, src, ob];
        end
    else
        @inbounds for row in 1:bU
            ;
            U_out[row, j, ob] = zero(T);
        end
        @inbounds for row in 1:bV
            ;
            V_out[row, j, ob] = zero(T);
        end
    end
end

# ─── Tile-view helpers ────────────────────────────────────────────────────────

function _tile_views(A::AbstractMatrix, A_tlr::TLRMatrix, obs::Vector{Int})
    return map(obs) do ob
        lin = _linear_from_offdiag(A_tlr, ob)
        ti, tj = _inverse_tile_index(A_tlr, lin)
        p0, q0 = tile_origin_coords(A_tlr, ti, tj)
        tm, tn = tile_size(A_tlr, ti, tj)
        view(A, p0:(p0+tm-1), q0:(q0+tn-1))
    end
end

@inline _batch_views(A::AbstractArray{T,3}) where {T} =
    [view(A,:,:,k) for k in axes(A, 3)]

# ─── Precision helpers ────────────────────────────────────────────────────────

@inline _compress_accum_type(::Type{Float16}) = Float32
@inline _compress_accum_type(::Type{Float32}) = Float64
@inline _compress_accum_type(::Type{Float64}) = Float64
@inline _compress_accum_type(::Type{ComplexF32}) = ComplexF64
@inline _compress_accum_type(::Type{ComplexF64}) = ComplexF64
@inline _compress_accum_type(::Type{T}) where {T} = T

@inline _adjoint_blas_char(::Type{<:Complex}) = 'C'
@inline _adjoint_blas_char(::Type) = 'T'

# ─── Workspace structs ────────────────────────────────────────────────────────

struct CompressCategoryWorkspace{
    ObsT<:Vector{Int},
    OmegaT<:AbstractArray,
    UT<:AbstractArray,
    VT<:AbstractArray,
    YHiT<:AbstractArray,
    GHiT<:AbstractArray,
    GTT<:AbstractArray,
    UGT<:AbstractArray,
    UTmpT<:AbstractArray,
    VTmpT<:AbstractArray,
    RanksT,
}
    obs::ObsT
    Omega::OmegaT
    U::UT           # aliases A_tlr panel array directly
    V::VT           # aliases A_tlr panel array directly
    Y_hi::YHiT
    G_hi::GHiT
    G_T::GTT
    UG::UGT
    U_tmp::UTmpT
    V_tmp::VTmpT
    ranks_local::RanksT
end

struct CompressWorkspace{IntWS,RightWS,BottomWS,StreamV}
    interior::IntWS
    right::RightWS
    bottom::BottomWS
    streams::StreamV
end

# ─── Workspace allocation ─────────────────────────────────────────────────────

function _allocate_category_workspace(
    prototype,
    obs::Vector{Int},
    tile_m::Int,
    tile_n::Int,
    max_rank::Int,
    ::Type{T},
    ::Type{Thi},
    rank_type::Type{<:Integer};
    U_store=nothing,
    V_store=nothing,
) where {T,Thi}
    count = length(obs)
    Omega = similar(prototype, T, tile_n, max_rank, count)
    U = U_store === nothing ? similar(prototype, T, tile_m, max_rank, count) : U_store
    V = V_store === nothing ? similar(prototype, T, tile_n, max_rank, count) : V_store
    Y_hi = similar(prototype, Thi, tile_m, max_rank, count)
    G_hi = similar(prototype, Thi, max_rank, max_rank, count)
    G_T = similar(prototype, T, max_rank, max_rank, count)
    UG = similar(prototype, T, tile_m, max_rank, count)
    U_tmp = similar(prototype, T, tile_m, max_rank, count)
    V_tmp = similar(prototype, T, tile_n, max_rank, count)
    ranks_local = similar(prototype, rank_type, count)
    return CompressCategoryWorkspace(
        obs, Omega, U, V, Y_hi, G_hi, G_T, UG, U_tmp, V_tmp, ranks_local,
    )
end

"""
    alloc_workspace(A_tlr) → CompressWorkspace

Pre-allocate all scratch buffers for `compress!`.  Reuse across calls on
matrices with the same layout to avoid repeated device allocations:

    ws = alloc_workspace(A_tlr)
    for A in matrices
        compress!(A_tlr, A, ws; tol=1f-3)
    end
"""
function alloc_workspace(A_tlr::TLRMatrix{<:Any,T}) where {T}
    prototype = A_tlr.D          # same backend/element type as all factor arrays
    rank_type = eltype(A_tlr.ranks)
    r = A_tlr.maxrank
    Thi = _compress_accum_type(T)
    b = A_tlr.tile_m
    mt, nt = tilegrid_size(A_tlr)
    tail_m = max(A_tlr.m - (mt - 1) * b, 1)
    tail_n = max(A_tlr.n - (nt - 1) * b, 1)

    interior = _allocate_category_workspace(
        prototype, A_tlr.obs_int, b, b, r, T, Thi, rank_type;
        U_store=A_tlr.int_U, V_store=A_tlr.int_V,
    )
    right = _allocate_category_workspace(
        prototype, A_tlr.obs_right, b, tail_n, r, T, Thi, rank_type;
        U_store=A_tlr.right_U, V_store=A_tlr.right_V,
    )
    bottom = _allocate_category_workspace(
        prototype, A_tlr.obs_bottom, tail_m, b, r, T, Thi, rank_type;
        U_store=A_tlr.bottom_U, V_store=A_tlr.bottom_V,
    )
    return CompressWorkspace(interior, right, bottom, create_streams(A_tlr.backend, 3))
end

# ─── Orthogonalisation helpers ────────────────────────────────────────────────

function _cholqr_pass!(Y_hi::AbstractArray{Thi,3}, G_hi, backend) where {Thi}
    count = size(Y_hi, 3)
    count == 0 && return Y_hi
    r = size(Y_hi, 2)
    Treal = typeof(real(zero(Thi)))
    reg = Thi(sqrt(eps(Treal)) * size(Y_hi, 1))
    gemm_batched!(_adjoint_blas_char(Thi), 'N', one(Thi), Y_hi, Y_hi, zero(Thi), G_hi)
    _add_reg_diag_kernel!(backend)(G_hi, reg; ndrange=(r, count))
    potrf_batched!('U', G_hi)
    trsm_batched!('R', 'U', 'N', 'N', G_hi, Y_hi, one(Thi))
    return Y_hi
end

function _newton_schulz!(U::AbstractArray{T,3}, G_T, UG; niters::Int=1) where {T}
    size(U, 3) == 0 && return U
    adj = _adjoint_blas_char(T)
    for _ in 1:niters
        gemm_batched!(adj, 'N', one(T), U, U, zero(T), G_T)
        gemm_batched!('N', 'N', one(T), U, G_T, zero(T), UG)
        @. U = T(3) / 2 * U - T(1) / 2 * UG
    end
    return U
end

function _truncate!(U::AbstractArray{T,3}, V, rk, U_tmp, V_tmp, eps_sq, backend) where {T}
    noff = size(U, 3)
    noff == 0 && return U
    R = size(U, 2)
    kernel! = _fused_truncate_kernel!(backend, R)
    kernel!(U_tmp, V_tmp, U, V, rk, eps_sq, Val{R}();
        ndrange=(R * noff,), workgroupsize=R)
    copyto!(U, U_tmp)
    copyto!(V, V_tmp)
    return U
end

# ─── Per-category compression pipeline ───────────────────────────────────────

function _compress_category!(
    backend,
    A_tlr::TLRMatrix,
    A::AbstractMatrix{T},
    cat::CompressCategoryWorkspace,
    eps_sq,
    alg::Symbol,
    ns_iters::Int,
) where {T}
    isempty(cat.obs) && return cat
    Random.randn!(cat.Omega)
    A_tiles = _tile_views(A, A_tlr, cat.obs)
    Omega_tiles = _batch_views(cat.Omega)
    U_tiles = _batch_views(cat.U)
    V_tiles = _batch_views(cat.V)

    # Step 1: range sampling  Y = A·Ω → U
    gemm_batched!('N', 'N', one(T), A_tiles, Omega_tiles, zero(T), U_tiles)

    # Step 2: orthogonalise U (in higher precision)
    copyto!(cat.Y_hi, cat.U)
    if alg === :cholqr2
        _cholqr_pass!(cat.Y_hi, cat.G_hi, backend)
        _cholqr_pass!(cat.Y_hi, cat.G_hi, backend)
        copyto!(cat.U, cat.Y_hi)
    else   # :cholqr_ns
        _cholqr_pass!(cat.Y_hi, cat.G_hi, backend)
        copyto!(cat.U, cat.Y_hi)
        _newton_schulz!(cat.U, cat.G_T, cat.UG; niters=ns_iters)
    end

    # Step 3: co-range  V = Aᵀ·U
    gemm_batched!(_adjoint_blas_char(T), 'N', one(T), A_tiles, U_tiles, zero(T), V_tiles)

    # Step 4: rank detection + truncation (fused SMEM kernel)
    _truncate!(cat.U, cat.V, cat.ranks_local, cat.U_tmp, cat.V_tmp, eps_sq, backend)
    return cat
end

# ─── Storage helpers ──────────────────────────────────────────────────────────

function _zero_offdiag!(A_tlr::TLRMatrix{<:Any,T}) where {T}
    fill!(A_tlr.int_U, zero(T))
    fill!(A_tlr.int_V, zero(T))
    fill!(A_tlr.right_U, zero(T))
    fill!(A_tlr.right_V, zero(T))
    fill!(A_tlr.bottom_U, zero(T))
    fill!(A_tlr.bottom_V, zero(T))
    fill!(A_tlr.ranks, zero(eltype(A_tlr.ranks)))
end

# Copy per-category local ranks back into the global A_tlr.ranks vector.
function _store_ranks!(A_tlr::TLRMatrix, cat::CompressCategoryWorkspace)
    isempty(cat.obs) && return
    rk_host = cat.ranks_local isa Vector ? cat.ranks_local : Array(cat.ranks_local)
    @inbounds for (k, ob) in enumerate(cat.obs)
        A_tlr.ranks[ob] = rk_host[k]
    end
end

# ─── Orchestration ────────────────────────────────────────────────────────────

function _compress_categories!(
    A_tlr::TLRMatrix{<:Any,T},
    A::AbstractMatrix{T},
    ws::CompressWorkspace,
    eps_sq,
    alg::Symbol,
    ns_iters::Int,
) where {T}
    _zero_offdiag!(A_tlr)
    cats = (ws.interior, ws.right, ws.bottom)
    if A_tlr.backend isa KernelAbstractions.CPU
        for cat in cats
            _compress_category!(A_tlr.backend, A_tlr, A, cat, eps_sq, alg, ns_iters)
        end
    else
        for (cat, stream) in zip(cats, ws.streams)
            with_stream(A_tlr.backend, stream) do
                _compress_category!(A_tlr.backend, A_tlr, A, cat, eps_sq, alg, ns_iters)
            end
        end
        for stream in ws.streams
            sync_stream(A_tlr.backend, stream)
        end
    end
    for cat in cats
        _store_ranks!(A_tlr, cat)
    end
    return A_tlr
end

# ─── Public API ───────────────────────────────────────────────────────────────

"""
    compress!(A_tlr, A [, ws]; tol=0.0, alg=:cholqr2, ns_iters=1)

Compress dense matrix `A` into the TLR container `A_tlr` in-place.

Per-tile effective ranks are detected via greedy V-column-norm thresholding
and stored in `ranks(A_tlr)`.  Factor arrays are updated in-place; call
`alloc_workspace` once to amortise device allocations across repeated calls:

    ws = alloc_workspace(A_tlr)
    for A in matrices
        compress!(A_tlr, A, ws; tol=1f-3)
    end

## Keywords

`tol` — Frobenius budget for dropped columns (default `0.0`).

`alg` — Orthogonalisation algorithm:
- `:cholqr2` (default) — two-pass Cholesky QR; more stable.
- `:cholqr_ns`         — one-pass cholqr + Newton-Schulz; faster on GPU.

`ns_iters` — Newton-Schulz iterations (only for `alg = :cholqr_ns`).
"""
function compress!(A_tlr::TLRMatrix{<:Any,T}, A::AbstractMatrix{T};
    tol::Real=0.0, alg::Symbol=:cholqr2, ns_iters::Int=1) where {T}
    ws = alloc_workspace(A_tlr)
    compress!(A_tlr, A, ws; tol, alg, ns_iters)
end

function compress!(A_tlr::TLRMatrix{<:Any,T}, A::AbstractMatrix{T},
    ws::CompressWorkspace;
    tol::Real=0.0, alg::Symbol=:cholqr2, ns_iters::Int=1) where {T}

    size(A, 1) == A_tlr.m && size(A, 2) == A_tlr.n ||
        throw(DimensionMismatch("A dimensions must match A_tlr"))
    A_tlr.m == A_tlr.n && A_tlr.tile_m == A_tlr.tile_n ||
        throw(ArgumentError("compress! currently requires square matrices with square tiles"))
    alg ∈ (:cholqr2, :cholqr_ns) ||
        throw(ArgumentError("alg must be :cholqr2 or :cholqr_ns"))
    tol >= 0 || throw(ArgumentError("tol must be >= 0"))

    n_diag = ndiag_tiles(A_tlr)

    _copy_diag_kernel!(A_tlr.backend)(
        A_tlr.D, A, A_tlr.m, A_tlr.n, A_tlr.tile_m, A_tlr.tile_n;
        ndrange=(A_tlr.tile_m, A_tlr.tile_n, n_diag),
    )

    RT = real(T); eps_sq = RT(tol)^2
    _compress_categories!(A_tlr, A, ws, eps_sq, alg, ns_iters)
    
    return A_tlr
end