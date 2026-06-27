export compress!, alloc_workspace

# ─── Kernels ──────────────────────────────────────────────────────────────────

@kernel function _copy_diag_kernel!(D::AbstractArray{T,3},
    A::AbstractMatrix{T}, layout::TileMap) where {T}
    row, col, b = @index(Global, NTuple)
    tm, tn = tile_sizes(layout, b, b)
    p0, q0 = tile_origin_coords(layout, b, b)
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
# Shared memory: norms[R], perm[R], k_buf[1].
@kernel function _fused_truncate_kernel!(
    U_out::AbstractArray{T,3}, V_out::AbstractArray{T,3},
    U::AbstractArray{T,3}, V::AbstractArray{T,3},
    rk::AbstractVector, eps_sq::T, ::Val{R}) where {T,R}
    ob = @index(Group, Linear)
    j = @index(Local, Linear)
    norms_s = @localmem T (R,)
    perm_s = @localmem Int32 (R,)
    k_buf = @localmem Int32 (1,)
    acc = zero(T)
    @inbounds for row in 1:size(V, 1)
        x = V[row, j, ob]
        acc += x * x
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
        rk[ob] = eltype(rk)(k_val);
        k_buf[1] = Int32(k_val)
    end
    @synchronize
    k = Int(@inbounds k_buf[1]);
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

# ─── Geometry helpers (also used by compact!) ─────────────────────────────────

function _classify_offdiag_tiles(layout::TileMap)
    mt, nt = size(layout)
    has_tr = layout.m % layout.tile_m != 0
    has_tc = layout.n % layout.tile_n != 0
    int_obs = Int[];
    rb_obs = Int[];
    bb_obs = Int[]
    for j in 1:nt, i in 1:mt
        i == j && continue
        ob = offdiag_batch_index(layout, i, j)
        if has_tc && j == nt
            push!(rb_obs, ob)
        elseif has_tr && i == mt
            push!(bb_obs, ob)
        else
            push!(int_obs, ob)
        end
    end
    return int_obs, rb_obs, bb_obs
end

function _tile_views(A::AbstractMatrix, layout::TileMap,
    obs::Vector{Int}, m_cat::Int, n_cat::Int)
    map(obs) do ob
        lin = offdiag_linear_index(layout, ob)
        ti, tj = inverse_tile_index(layout, lin)
        p0, q0 = tile_origin_coords(layout, ti, tj)
        view(A, p0:(p0+m_cat-1), q0:(q0+n_cat-1))
    end
end

# ─── Workspace structs ────────────────────────────────────────────────────────

"""
    CPUCompressWorkspace

Lightweight workspace for CPU compression.  Holds tile-category indices;
all per-tile scratch is handled by LAPACK and Julia's allocator.
"""
struct CPUCompressWorkspace
    int_obs::Vector{Int}
    rb_obs::Vector{Int}
    bb_obs::Vector{Int}
end

"""
    GPUCompressWorkspace{T, Arr3T, Arr3F64, StreamV}

Pre-allocated device buffers for GPU `compress!`.  Every temporary array lives
here; the compress path itself performs zero device allocations.

- `Omega_int/rb/bb` — sketch matrices per tile category (refilled with randn! each call)
- `Y64`             — Float64 U for Gram-matrix cholqr
- `G_f64`           — Float64 Gram matrix (cholqr)
- `G_T`             — working-precision Gram matrix (Newton-Schulz)
- `UG`              — working-precision U·G (Newton-Schulz)
- `U_tmp / V_tmp`   — truncation output before in-place copy-back
- `streams`         — 3 independent GPU streams for concurrent category GEMMs
"""
struct GPUCompressWorkspace{T,Arr3T,Arr3F64,StreamV}
    int_obs::Vector{Int}
    rb_obs::Vector{Int}
    bb_obs::Vector{Int}
    Omega_int::Arr3T      # [b,    r, n_int]
    Omega_rb::Arr3T       # [tail, r, n_rb]  — 3rd dim = 0 when n % b == 0
    Omega_bb::Arr3T       # [b,    r, n_bb]
    Y64::Arr3F64          # [b, r, noff]
    G_f64::Arr3F64        # [r, r, noff]
    G_T::Arr3T            # [r, r, noff]
    UG::Arr3T             # [b, r, noff]
    U_tmp::Arr3T          # [b, r, noff]
    V_tmp::Arr3T          # [b, r, noff]
    streams::StreamV
end

"""
    alloc_workspace(A_tlr) → CPUCompressWorkspace | GPUCompressWorkspace

Pre-allocate all scratch buffers for `compress!`.  Reuse across calls on
matrices with the same layout to avoid repeated device allocations.
"""
function alloc_workspace(A_tlr::TLRMatrix{<:KernelAbstractions.CPU,T}) where {T}
    int_obs, rb_obs, bb_obs = _classify_offdiag_tiles(A_tlr.layout)
    return CPUCompressWorkspace(int_obs, rb_obs, bb_obs)
end

function alloc_workspace(A_tlr::TLRMatrix{<:Any,T}) where {T}
    layout = A_tlr.layout
    backend = A_tlr.backend
    r = maxrank(A_tlr)
    b = layout.tile_m
    tail = A_tlr.n % b

    int_obs, rb_obs, bb_obs = _classify_offdiag_tiles(layout)
    noff = noffdiag_tiles(layout)
    U = left_factors(A_tlr);
    V = right_factors(A_tlr)

    return GPUCompressWorkspace(
        int_obs, rb_obs, bb_obs,
        similar(U, T, b, r, length(int_obs)),
        similar(U, T, max(tail, 1), r, length(rb_obs)),
        similar(U, T, b, r, length(bb_obs)),
        similar(U, Float64, b, r, noff),
        similar(U, Float64, r, r, noff),
        similar(U, T, r, r, noff),
        similar(U, T, b, r, noff),
        similar(U, T, b, r, noff),
        similar(V, T, layout.tile_n, r, noff),
        create_streams(backend, 3),
    )
end

# ─── GPU algorithm steps (pre-allocated buffers, zero internal allocations) ───

# One Gram-matrix Cholesky QR pass: promote to Float64, factor, solve, demote.
function _cholqr_pass!(U::AbstractArray{T,3}, Y64, G_f64, backend) where {T}
    copyto!(Y64, U)
    r, count = size(Y64, 2), size(Y64, 3)
    gemm_batched!('T', 'N', 1.0, Y64, Y64, 0.0, G_f64)
    _add_reg_diag_kernel!(backend)(G_f64, sqrt(eps(Float64)) * size(Y64, 1);
        ndrange=(r, count))
    potrf_batched!('U', G_f64)
    trsm_batched!('R', 'U', 'N', 'N', G_f64, Y64, 1.0)
    copyto!(U, Y64)
end

# Newton-Schulz: U ← (3/2)U − (1/2)U(UᵀU).
function _newton_schulz!(U::AbstractArray{T,3}, G_T, UG; niters::Int=1) where {T}
    for _ in 1:niters
        gemm_batched!('T', 'N', one(T), U, U, zero(T), G_T)
        gemm_batched!('N', 'N', one(T), U, G_T, zero(T), UG)
        @. U = T(3)/2 * U - T(1)/2 * UG
    end
end

# Rank detection + truncation via fused SMEM kernel.
function _truncate!(U::AbstractArray{T,3}, V, rk, U_tmp, V_tmp, eps_sq::T, backend) where {T}
    R = size(U, 2);
    noff = size(U, 3)
    kernel! = _fused_truncate_kernel!(backend, R)
    kernel!(U_tmp, V_tmp, U, V, rk, eps_sq, Val{R}();
        ndrange=(R * noff,), workgroupsize=R)
    copyto!(U, U_tmp);
    copyto!(V, V_tmp)
end

# ─── CPU compression path ─────────────────────────────────────────────────────
# Per-tile sequential loop; LAPACK Householder QR; standard Julia sort.
# No KernelAbstractions, streams, or device memory.

function _compress_cpu!(A_tlr::TLRMatrix{<:Any,T}, A::AbstractMatrix{T},
    ::CPUCompressWorkspace, eps_sq::T,
    alg::Symbol, ns_iters::Int) where {T}
    layout = A_tlr.layout
    U, V, rk = left_factors(A_tlr), right_factors(A_tlr), ranks(A_tlr)
    r = maxrank(A_tlr)
    fill!(U, zero(T));
    fill!(V, zero(T))

    for ob in 1:noffdiag_tiles(layout)
        lin = offdiag_linear_index(layout, ob)
        tile_i, tile_j = inverse_tile_index(layout, lin)
        p0, q0 = tile_origin_coords(layout, tile_i, tile_j)
        tm, tn = tile_sizes(layout, tile_i, tile_j)
        r_eff = min(r, tm, tn)
        A_tile = view(A, p0:(p0+tm-1), q0:(q0+tn-1))

        # Step 1: range sampling + Householder QR
        Y = A_tile * randn(T, tn, r_eff)
        _, tau = LAPACK.geqrf!(Y);
        LAPACK.orgqr!(Y, tau, r_eff)

        # Step 2: optional Newton-Schulz refinement
        if alg === :cholqr_ns
            for _ in 1:ns_iters
                G = Y' * Y
                UG_cpu = Y * G
                @. Y = T(3)/2 * Y - T(1)/2 * UG_cpu
            end
        end

        # Step 3: co-range
        W = A_tile' * Y

        # Step 4: rank detection (V column norms → greedy threshold)
        v_sq = vec(sum(abs2, W; dims=1))
        perm = sortperm(v_sq; rev=true)
        budget = eps_sq;
        k_val = r_eff
        for idx in r_eff:-1:1
            cost = v_sq[perm[idx]]
            cost <= budget || break
            budget -= cost;
            k_val -= 1
        end

        U[1:tm, 1:k_val, ob] .= Y[:, perm[1:k_val]]
        V[1:tn, 1:k_val, ob] .= W[:, perm[1:k_val]]
        U[1:tm, (k_val+1):r_eff, ob] .= 0
        V[1:tn, (k_val+1):r_eff, ob] .= 0
        rk[ob] = eltype(rk)(k_val)
    end
end

# ─── GPU compression path ─────────────────────────────────────────────────────
# Four explicit steps; all temporaries come from ws — zero on-the-fly allocations.

function _compress_gpu!(A_tlr::TLRMatrix{<:Any,T}, A::AbstractMatrix{T},
    ws::GPUCompressWorkspace, eps_sq::T,
    alg::Symbol, ns_iters::Int) where {T}
    layout = A_tlr.layout
    backend = A_tlr.backend
    b = layout.tile_m
    tail = A_tlr.n % b

    U, V, rk = left_factors(A_tlr), right_factors(A_tlr), ranks(A_tlr)
    fill!(U, zero(T));
    fill!(V, zero(T))

    s1, s2, s3 = ws.streams[1], ws.streams[2], ws.streams[3]

    # Non-allocating tile views into A (lda = size(A,1), uniform within each category).
    # On CUDA/AMDGPU dispatches to cublasGemmBatchedEx via Vector{<:StridedCuMatrix}.
    A_int_v = _tile_views(A, layout, ws.int_obs, b, b)
    A_rb_v = _tile_views(A, layout, ws.rb_obs, b, tail)
    A_bb_v = _tile_views(A, layout, ws.bb_obs, tail, b)

    U_int_v = [view(U, 1:b, :, ob) for ob in ws.int_obs]
    U_rb_v = [view(U, 1:b, :, ob) for ob in ws.rb_obs]
    U_bb_v = [view(U, 1:tail, :, ob) for ob in ws.bb_obs]
    V_int_v = [view(V, 1:b, :, ob) for ob in ws.int_obs]
    V_rb_v = [view(V, 1:tail, :, ob) for ob in ws.rb_obs]
    V_bb_v = [view(V, 1:b, :, ob) for ob in ws.bb_obs]

    Omega_int_v = [view(ws.Omega_int,:,:,k) for k in eachindex(ws.int_obs)]
    Omega_rb_v = [view(ws.Omega_rb,:,:,k) for k in eachindex(ws.rb_obs)]
    Omega_bb_v = [view(ws.Omega_bb,:,:,k) for k in eachindex(ws.bb_obs)]

    # ── Step 1: Range sampling  Y = A·Ω → U  (3 concurrent streams) ──────────
    Random.randn!(ws.Omega_int)
    Random.randn!(ws.Omega_rb)
    Random.randn!(ws.Omega_bb)

    with_stream(backend, s1) do
        gemm_batched!('N', 'N', one(T), A_int_v, Omega_int_v, zero(T), U_int_v)
    end
    with_stream(backend, s2) do
        !isempty(ws.rb_obs) &&
            gemm_batched!('N', 'N', one(T), A_rb_v, Omega_rb_v, zero(T), U_rb_v)
    end
    with_stream(backend, s3) do
        !isempty(ws.bb_obs) &&
            gemm_batched!('N', 'N', one(T), A_bb_v, Omega_bb_v, zero(T), U_bb_v)
    end
    sync_stream(backend, s1);
    sync_stream(backend, s2);
    sync_stream(backend, s3)

    # ── Step 2: Orthogonalise U  (single batched call over all tiles) ─────────
    if alg === :cholqr2
        _cholqr_pass!(U, ws.Y64, ws.G_f64, backend)
        _cholqr_pass!(U, ws.Y64, ws.G_f64, backend)
    else   # :cholqr_ns
        _cholqr_pass!(U, ws.Y64, ws.G_f64, backend)
        _newton_schulz!(U, ws.G_T, ws.UG; niters=ns_iters)
    end

    # ── Step 3: Co-range  V = A^T·U  (3 concurrent streams) ──────────────────
    sync_streams_with_default(backend, ws.streams)   # GPU-side barrier post-cholqr

    with_stream(backend, s1) do
        gemm_batched!('T', 'N', one(T), A_int_v, U_int_v, zero(T), V_int_v)
    end
    with_stream(backend, s2) do
        !isempty(ws.rb_obs) &&
            gemm_batched!('T', 'N', one(T), A_rb_v, U_rb_v, zero(T), V_rb_v)
    end
    with_stream(backend, s3) do
        !isempty(ws.bb_obs) &&
            gemm_batched!('T', 'N', one(T), A_bb_v, U_bb_v, zero(T), V_bb_v)
    end
    sync_stream(backend, s1);
    sync_stream(backend, s2);
    sync_stream(backend, s3)

    # ── Step 4: Rank detection + truncation (fused SMEM kernel) ───────────────
    _truncate!(U, V, rk, ws.U_tmp, ws.V_tmp, eps_sq, backend)
end

# ─── Public API ───────────────────────────────────────────────────────────────

"""
    compress!(A_tlr, A [, ws]; tol=0.0, alg=:cholqr2, ns_iters=1)

Compress dense `A` into the TLR container `A_tlr` in-place.

Per-tile effective ranks are detected and stored in `ranks(A_tlr)`.  U and V
factors are kept padded to `blocksize × maxrank`; call [`compact!`](@ref) to
trim each factor to its actual rank.

Pass a pre-allocated [`alloc_workspace`](@ref) to amortise device allocations
across multiple compressions of matrices with the same layout:

    ws = alloc_workspace(A_tlr)
    for A in training_set
        compress!(A_tlr, A, ws; tol = 1f-3)
    end

## Keywords

`tol` — Frobenius budget for dropped columns (default `0.0`).  Columns whose
cumulative `‖V[:,j]‖²` fits within `tol²` are removed.  Use `0.0` to remove
only exact-zero columns (true null space on CPU via Householder QR); use a
small positive value on GPU to also discard near-zero residuals.

`alg` — Orthogonalisation algorithm:
- `:cholqr2` (default) — two-pass Gram-matrix Cholesky QR; stable.
- `:cholqr_ns`         — one-pass cholqr + Newton-Schulz; faster on GPU.

`ns_iters` — Newton-Schulz iterations (only for `alg = :cholqr_ns`).
"""
function compress!(A_tlr::TLRMatrix{<:Any,T}, A::AbstractMatrix{T};
    tol::Real=0.0, alg::Symbol=:cholqr2,
    ns_iters::Int=1) where {T}
    ws = alloc_workspace(A_tlr)
    compress!(A_tlr, A, ws; tol, alg, ns_iters)
end

function compress!(A_tlr::TLRMatrix{<:Any,T}, A::AbstractMatrix{T},
    ws::Union{CPUCompressWorkspace,GPUCompressWorkspace};
    tol::Real=0.0, alg::Symbol=:cholqr2,
    ns_iters::Int=1) where {T}
    A_tlr.storage isa UniformTileStorage ||
        throw(ArgumentError("compress! requires padded UniformTileStorage; " *
                            "call it before compact! or allocate a fresh TLRMatrix"))
    compress_diag(A_tlr) &&
        throw(ArgumentError("compress! requires compress_diag=false"))
    size(A, 1) == A_tlr.m && size(A, 2) == A_tlr.n ||
        throw(DimensionMismatch("A dimensions must match A_tlr"))
    A_tlr.m == A_tlr.n && A_tlr.layout.tile_m == A_tlr.layout.tile_n ||
        throw(ArgumentError("compress! currently requires square matrices with square tiles"))
    alg ∈ (:cholqr2, :cholqr_ns) ||
        throw(ArgumentError("alg must be :cholqr2 or :cholqr_ns"))
    tol >= 0 || throw(ArgumentError("tol must be ≥ 0"))

    # Diagonal tiles
    ndiag = ndiag_tiles(A_tlr.layout)
    if ndiag > 0
        _copy_diag_kernel!(A_tlr.backend)(
            dense_diag(A_tlr), A, A_tlr.layout;
            ndrange=(A_tlr.layout.tile_m, A_tlr.layout.tile_n, ndiag))
    end
    noffdiag_tiles(A_tlr.layout) == 0 && return A_tlr

    eps_sq = T(tol)^2
    if A_tlr.backend isa KernelAbstractions.CPU
        _compress_cpu!(A_tlr, A, ws, eps_sq, alg, ns_iters)
    else
        _compress_gpu!(A_tlr, A, ws, eps_sq, alg, ns_iters)
    end

    return A_tlr
end
