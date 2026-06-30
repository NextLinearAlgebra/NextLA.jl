export compress!, workspace_info

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
            U_out[row, j, ob] = U[row, src, ob];
        end
        @inbounds for row in 1:bV
            V_out[row, j, ob] = V[row, src, ob];
        end
    else
        @inbounds for row in 1:bU
            U_out[row, j, ob] = zero(T);
        end
        @inbounds for row in 1:bV
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

# Scratch buffers for one tile category during compress!.
#
# Temporal reuse:
#   V doubles as Omega: filled with randn! for the range sketch (step 1),
#   then overwritten by the co-range A^T·U (step 3). No separate Omega needed.
#
#   U_tmp doubles as UG: Newton-Schulz writes U·G_T here (step 2),
#   then truncation uses it as the sorted-U output (step 4). No separate UG.
struct CompressCategoryWorkspace{
    ObsT  <: Vector{Int},
    UT    <: AbstractArray,   # aliases A_tlr panel — U factors + sketch target
    VT    <: AbstractArray,   # aliases A_tlr panel — Omega initially, then V = A^T U
    YHiT  <: AbstractArray,   # high-precision U copy for cholqr
    GHiT  <: AbstractArray,   # high-precision Gram matrix for cholqr
    GTT   <: AbstractArray,   # working-precision Gram matrix (Newton-Schulz)
    UTmpT <: AbstractArray,   # UG buffer (NS) reused as sorted-U (truncation)
    VTmpT <: AbstractArray,   # sorted-V output (truncation)
    RanksT,
}
    obs::ObsT
    U::UT
    V::VT
    Y_hi::YHiT
    G_hi::GHiT
    G_T::GTT
    U_tmp::UTmpT
    V_tmp::VTmpT
    ranks_local::RanksT
end

struct CompressWorkspace{IntWS, RightWS, BottomWS, BufT, BufHiT, StreamV}
    interior::IntWS
    right::RightWS
    bottom::BottomWS
    buf_T::BufT      # flat T-precision backing buffer (G_T + U_tmp + V_tmp, all categories)
    buf_hi::BufHiT   # flat Thi-precision backing buffer (Y_hi + G_hi, all categories)
    streams::StreamV
end

# ─── Workspace allocation ─────────────────────────────────────────────────────

# Carve a reshaped view from a flat buffer starting at offset `off` (1-based).
@inline function _buf_view(buf, off::Int, dims::Vararg{Int})
    len = prod(dims)
    reshape(view(buf, off:off + len - 1), dims...)
end

@inline _category_specs(A_tlr, b, tail_m, tail_n) = (
    (; name="interior", obs=A_tlr.obs_int, U=A_tlr.int_U, V=A_tlr.int_V, tm=b, tn=b),
    (; name="right",    obs=A_tlr.obs_right, U=A_tlr.right_U, V=A_tlr.right_V, tm=b, tn=tail_n),
    (; name="bottom",   obs=A_tlr.obs_bottom, U=A_tlr.bottom_U, V=A_tlr.bottom_V, tm=tail_m, tn=b),
)

@inline _scratch_sizes(tm, tn, r, n) = ((r*r + tm*r + tn*r) * n, (tm*r + r*r) * n)

function _carve_category_workspace(buf_T, pT::Int, buf_hi, pH::Int, spec, r::Int, rank_type, proto)
    n = length(spec.obs)
    G_T   = _buf_view(buf_T,  pT, r,       r, n); pT += r*r*n
    U_tmp = _buf_view(buf_T,  pT, spec.tm, r, n); pT += spec.tm*r*n
    V_tmp = _buf_view(buf_T,  pT, spec.tn, r, n); pT += spec.tn*r*n
    Y_hi  = _buf_view(buf_hi, pH, spec.tm, r, n); pH += spec.tm*r*n
    G_hi  = _buf_view(buf_hi, pH, r,       r, n); pH += r*r*n
    cat = CompressCategoryWorkspace(
        spec.obs, spec.U, spec.V, Y_hi, G_hi, G_T, U_tmp, V_tmp,
        similar(proto, rank_type, n),
    )
    cat, pT, pH
end

"""
    alloc_workspace(A_tlr) → CompressWorkspace

Pre-allocate all scratch buffers for `compress!`.  Two flat device allocations
(one in working precision T, one in accumulation precision Thi) serve all three
tile categories; U and V alias A_tlr storage directly.  Reuse across repeated
calls on the same matrix layout:

    ws = alloc_workspace(A_tlr)
    for A in matrices
        compress!(A_tlr, A, ws; tol=1f-3)
    end
"""
function alloc_workspace(A_tlr::TLRMatrix{<:Any,T}) where {T}
    Thi       = _compress_accum_type(T)
    proto     = A_tlr.D
    rank_type = eltype(A_tlr.ranks)
    r  = A_tlr.maxrank
    b  = A_tlr.tile_m
    mt, nt = tilegrid_size(A_tlr)
    tail_m = max(A_tlr.m - (mt - 1) * b, 1)
    tail_n = max(A_tlr.n - (nt - 1) * b, 1)
    specs = _category_specs(A_tlr, b, tail_m, tail_n)

    sT = sH = 0
    for spec in specs
        dT, dH = _scratch_sizes(spec.tm, spec.tn, r, length(spec.obs))
        sT += dT
        sH += dH
    end

    buf_T  = similar(proto, T, max(sT, 1))
    buf_hi = similar(proto, Thi, max(sH, 1))
    fill!(buf_T,  zero(T))
    fill!(buf_hi, zero(Thi))

    pT = Ref(1)
    pH = Ref(1)
    cats = map(specs) do spec
        cat, pT[], pH[] = _carve_category_workspace(
            buf_T, pT[], buf_hi, pH[], spec, r, rank_type, proto)
        cat
    end

    CompressWorkspace(cats..., buf_T, buf_hi, create_streams(A_tlr.backend, 3))
end

"""
    workspace_info([io,] A_tlr)

Print the scratch memory required by `alloc_workspace(A_tlr)` broken down by
category and array.  U and V are aliased to `A_tlr` storage (no extra allocation);
everything else is fresh scratch.
"""
function workspace_info(io::IO, A_tlr::TLRMatrix{<:Any,T}) where {T}
    Thi = _compress_accum_type(T)
    sT  = sizeof(T)
    sThi = sizeof(Thi)
    r   = A_tlr.maxrank
    b   = A_tlr.tile_m
    mt, nt = tilegrid_size(A_tlr)
    tail_m = max(A_tlr.m - (mt - 1) * b, 1)
    tail_n = max(A_tlr.n - (nt - 1) * b, 1)
    specs = _category_specs(A_tlr, b, tail_m, tail_n)

    _mb(bytes) = bytes / 1_048_576
    _fmt_mb(x) = lpad(_fixed_3(_mb(x)), 10)

    println(io, "compress! workspace for $(A_tlr.m)×$(A_tlr.n)  b=$(b)  r=$(r)  T=$(T)  Thi=$(Thi)")
    println(io, "  U/V aliased to A_tlr storage — not counted below.")
    println(io, "  ┌──────────────┬───────┬────────────┬────────────┬────────────┐")
    println(io, "  │ category     │ tiles │  work (MB) │  hi   (MB) │  total(MB) │")
    println(io, "  │              │       │  T=$(T) │ Thi=$(Thi) │            │")
    println(io, "  ├──────────────┼───────┼────────────┼────────────┼────────────┤")

    total_T = total_Thi = 0
    for spec in specs
        n = length(spec.obs)
        scratch_T, scratch_Thi = _scratch_sizes(spec.tm, spec.tn, r, n)
        sT_cat = scratch_T * sT
        sThi_cat = scratch_Thi * sThi
        total_T += sT_cat
        total_Thi += sThi_cat
        println(io,
            "  │ ", rpad(spec.name, 12),
            " │ ", lpad(string(n), 5),
            " │ ", _fmt_mb(sT_cat),
            " │ ", _fmt_mb(sThi_cat),
            " │ ", _fmt_mb(sT_cat + sThi_cat),
            " │")
    end

    println(io, "  ├──────────────┼───────┼────────────┼────────────┼────────────┤")
    println(io,
        "  │ ", rpad("TOTAL", 12),
        " │ ", lpad("", 5),
        " │ ", _fmt_mb(total_T),
        " │ ", _fmt_mb(total_Thi),
        " │ ", _fmt_mb(total_T + total_Thi),
        " │")
    println(io, "  └──────────────┴───────┴────────────┴────────────┴────────────┘")
end
workspace_info(A_tlr::TLRMatrix) = workspace_info(stdout, A_tlr)

function _fixed_3(x::Real)
    y = round(Float64(x); digits=3)
    s = string(y)
    dot = findfirst(==('.'), s)
    if isnothing(dot)
        return s * ".000"
    end
    decimals = lastindex(s) - dot
    decimals >= 3 && return s
    return s * repeat("0", 3 - decimals)
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
    Y_hi
end

function _newton_schulz!(U::AbstractArray{T,3}, G_T, UG; niters::Int=1) where {T}
    size(U, 3) == 0 && return U
    adj = _adjoint_blas_char(T)
    for _ in 1:niters
        gemm_batched!(adj, 'N', one(T), U, U, zero(T), G_T)
        gemm_batched!('N', 'N', one(T), U, G_T, zero(T), UG)
        @. U = T(3) / 2 * U - T(1) / 2 * UG
    end
    U
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
    U
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

    A_tiles = _tile_views(A, A_tlr, cat.obs)
    U_tiles = _batch_views(cat.U)
    V_tiles = _batch_views(cat.V)

    # Step 1: range sampling  Y = A·Ω → U
    # V holds Ω (filled with randn!) for now; step 3 will overwrite it with A^T·U.
    Random.randn!(cat.V)
    gemm_batched!('N', 'N', one(T), A_tiles, V_tiles, zero(T), U_tiles)

    # Step 2: orthogonalise U (in higher precision)
    copyto!(cat.Y_hi, cat.U)
    if alg === :cholqr2
        _cholqr_pass!(cat.Y_hi, cat.G_hi, backend)
        _cholqr_pass!(cat.Y_hi, cat.G_hi, backend)
        copyto!(cat.U, cat.Y_hi)
    else   # :cholqr_ns
        _cholqr_pass!(cat.Y_hi, cat.G_hi, backend)
        copyto!(cat.U, cat.Y_hi)
        # U_tmp doubles as the UG buffer here; it is free until truncation (step 4).
        _newton_schulz!(cat.U, cat.G_T, cat.U_tmp; niters=ns_iters)
    end

    # Step 3: co-range  V = Aᵀ·U  (overwrites the Ω we no longer need)
    gemm_batched!(_adjoint_blas_char(T), 'N', one(T), A_tiles, U_tiles, zero(T), V_tiles)

    # Step 4: rank detection + truncation (fused SMEM kernel)
    _truncate!(cat.U, cat.V, cat.ranks_local, cat.U_tmp, cat.V_tmp, eps_sq, backend)
    cat
end

# ─── Storage helpers ──────────────────────────────────────────────────────────

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
    A_tlr
end

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
compress!(A_tlr::TLRMatrix{<:Any,T}, A::AbstractMatrix{T}; kwargs...) where {T} =
    compress!(A_tlr, A, alloc_workspace(A_tlr); kwargs...)

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

    A_tlr
end
