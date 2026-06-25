export compress!

# ─── Kernel: copy diagonal tiles A → D ───────────────────────────────────────

@kernel function _copy_diag_kernel!(D::AbstractArray{T,3},
                                    A::AbstractMatrix{T},
                                    layout::TileMap) where {T}
    row, col, b_idx = @index(Global, NTuple)
    tile_m, tile_n = tile_sizes(layout, b_idx, b_idx)
    p0, q0 = tile_origin_coords(layout, b_idx, b_idx)
    if row <= tile_m && col <= tile_n
        @inbounds D[row, col, b_idx] = A[p0 + row - 1, q0 + col - 1]
    end
end

function _copy_diag!(D::AbstractArray{T,3}, A::AbstractMatrix{T},
                     layout::TileMap, backend) where {T}
    ndiag = ndiag_tiles(layout)
    ndiag == 0 && return D
    _copy_diag_kernel!(backend)(D, A, layout; ndrange=(layout.tile_m, layout.tile_n, ndiag))
    return D
end

# ─── Kernel: scatter contiguous cat buffer → TileOrder storage ───────────────
# ndrange must be (nrows, ncols, count) so only the target region is written.

@kernel function _scatter_offdiag_kernel!(dst::AbstractArray{T,3},
                                          src::AbstractArray{T,3},
                                          offdiag_indices::AbstractVector{<:Integer}) where {T}
    row, col, b_idx = @index(Global, NTuple)
    ob = @inbounds offdiag_indices[b_idx]
    @inbounds dst[row, col, ob] = src[row, col, b_idx]
end

# ─── Kernel: regularise Gram matrix diagonal ─────────────────────────────────
# Adds a caller-supplied `reg` value to every G[j,j,b] to ensure positive-
# definiteness.  The caller computes reg = sqrt(eps(T)) outside the kernel 
@kernel function _add_reg_diag_kernel!(G::AbstractArray{T,3}, reg::T) where {T}
    j, b = @index(Global, NTuple)
    @inbounds G[j, j, b] += reg
end

# ─── Kernel: column squared-norms of a 3-D cat buffer ────────────────────────

@kernel function _col_norms_sq_kernel!(norms::AbstractMatrix{T},
                                       src::AbstractArray{T,3}) where {T}
    j, b = @index(Global, NTuple)
    n_rows = size(src, 1)
    acc = zero(T)
    @inbounds for row in 1:n_rows
        x = src[row, j, b]
        acc += x * x
    end
    @inbounds norms[j, b] = acc
end

# ─── Kernel: apply column permutation to a cat buffer ────────────────────────
# perm[k, b] is the source column index for destination column k of tile b.
# Columns beyond ks[b] are zeroed (compaction + truncation in one pass).

@kernel function _compact_col_kernel!(dst::AbstractArray{T,3},
                                      src::AbstractArray{T,3},
                                      perm::AbstractMatrix{Int32},
                                      ks::AbstractVector{Int32}) where {T}
    row, k, b = @index(Global, NTuple)
    k_b = Int(@inbounds ks[b])
    if k <= k_b
        j = Int(@inbounds perm[k, b])
        @inbounds dst[row, k, b] = src[row, j, b]
    else
        @inbounds dst[row, k, b] = zero(T)
    end
end

# ─── Classify off-diagonal tiles by tile-dimension category ──────────────────

function _classify_offdiag_tiles(layout::TileMap)
    mt, nt   = size(layout)
    has_tail_row = layout.m % layout.tile_m != 0
    has_tail_col = layout.n % layout.tile_n != 0

    int_obs = Int[]
    rb_obs  = Int[]
    bb_obs  = Int[]

    for j in 1:nt, i in 1:mt
        i == j && continue
        ob = offdiag_batch_index(layout, i, j)
        if has_tail_col && j == nt
            push!(rb_obs, ob)
        elseif has_tail_row && i == mt
            push!(bb_obs, ob)
        else
            push!(int_obs, ob)
        end
    end

    return int_obs, rb_obs, bb_obs
end

# ─── Build non-allocating tile views into A ──────────────────────────────────
# Each view is a StridedMatrix window into A with lda = size(A,1).
# On CUDA/AMDGPU this dispatches to cublasGemmBatchedEx / rocblas_gemm_batched_ex
# via the existing Vector{<:StridedCuMatrix} / Vector{<:StridedROCMatrix} path.

function _tile_views(A::AbstractMatrix, layout::TileMap,
                     obs::Vector{Int}, m_cat::Int, n_cat::Int)
    return map(obs) do ob
        lin = offdiag_linear_index(layout, ob)
        tile_i, tile_j = inverse_tile_index(layout, lin)
        p0, q0 = tile_origin_coords(layout, tile_i, tile_j)
        view(A, p0:p0+m_cat-1, q0:q0+n_cat-1)
    end
end

# ─── QR orthogonalization ─────────────────────────────────────────────────────
#
# CPU path: Householder QR via LAPACK geqrf! + orgqr!
# ─────────────────────────────────────────────────────────────────────────────
# All columns of Y = A·Ω lie in range(A) (a k-dimensional subspace).  This
# means G = Y^T Y has rank k < r and r-k zero eigenvalues.  Cholqr cannot
# resolve the null space: it amplifies float-point noise in those directions
# rather than zeroing them.
#
# Householder QR correctly handles rank-deficient Y: the first k columns of Q
# span range(Y)=range(A) and the remaining r-k columns span range(A)^⊥.
# Consequently V[:,j] = A^T Q[:,j] = 0 exactly for j > k, which is the
# property required for V-norm truncation to detect the true tile rank.
#
# One pass suffices — Q^T Q = I to machine precision.

function _cholqr!(Y_cat::AbstractArray{T,3},
                  ::KernelAbstractions.CPU) where {T<:AbstractFloat}
    r = size(Y_cat, 2)
    for b in axes(Y_cat, 3)
        Y_b = view(Y_cat, :, :, b)   # contiguous [m, r] column-major slice
        _, tau = LAPACK.geqrf!(Y_b)  # in-place QR; Y_b overwritten with H-vecs + R
        LAPACK.orgqr!(Y_b, tau, r)   # form explicit Q in Y_b (first r columns)
    end
    return Y_cat
end

# GPU path: promote to Float64, run Gram-matrix cholqr, cast back.
# The F64 promotion suppresses catastrophic cancellation in G = Y^T Y.
# A second pass (cholqr2) is recommended on GPU to recover orthogonality
# degraded by casting back to the storage precision.

function _cholqr!(Y_cat::AbstractArray{T,3}, backend) where {T<:AbstractFloat}
    Y64 = similar(Y_cat, Float64)
    copyto!(Y64, Y_cat)
    _cholqr!(Y64, backend)
    copyto!(Y_cat, Y64)
    return Y_cat
end

function _cholqr!(Y_cat::AbstractArray{Float64,3}, backend)
    r, count = size(Y_cat, 2), size(Y_cat, 3)
    G   = similar(Y_cat, r, r, count)
    gemm_batched!('T', 'N', 1.0, Y_cat, Y_cat, 0.0, G)
    reg = sqrt(eps(Float64)) * size(Y_cat, 1)
    _add_reg_diag_kernel!(backend)(G, reg; ndrange=(r, count))
    potrf_batched!('U', G)
    trsm_batched!('R', 'U', 'N', 'N', G, Y_cat, 1.0)
    return Y_cat
end

# ─── Cholesky QR2 (two passes, GPU) ──────────────────────────────────────────
# Second pass: G₂ = U₁^T U₁ ≈ I is well-conditioned regardless of κ(Y),
# recovering orthogonality lost when casting back from F64 to the storage type.
# On CPU the Householder path is exact in one pass; the second call is a no-op.

function _cholqr2!(Y_cat::AbstractArray{T,3}, backend) where {T}
    _cholqr!(Y_cat, backend)
    _cholqr!(Y_cat, backend)
    return Y_cat
end

# ─── Newton-Schulz refinement ────────────────────────────────────────────────
# One iteration: U ← (3/2)U - (1/2) U (U^T U).
# Each iteration is one syrk + one gemm (≈ 2 GEMMs).
# Converges quadratically near σ=1; safe to apply after cholqr since
# real columns have σ ≈ 1.  Null-space columns (σ ≈ 0) grow by ≤ (3/2)^niters
# per iteration but remain near-zero for typical niters=1.

function _newton_schulz!(U_cat::AbstractArray{T,3}; niters::Int=1) where {T}
    r, count = size(U_cat, 2), size(U_cat, 3)
    G = similar(U_cat, r, r, count)
    for _ in 1:niters
        gemm_batched!('T', 'N', one(T),    U_cat, U_cat, zero(T), G)   # G = U^T U
        gemm_batched!('N', 'N', -one(T)/2, U_cat, G,    T(3)/2,  U_cat) # U ← (3/2)U - (1/2)UG
    end
    return U_cat
end

# ─── Truncate one category (Algorithm 2 only) ────────────────────────────────
# Sorts columns of U_cat / V_cat by descending ‖V[:,j]‖ and removes trailing
# columns until the total dropped squared norm ≤ eps_sq.
# Compaction is applied in-place via a temporary buffer.

function _truncate_category!(U_cat::AbstractArray{T,3},
                              V_cat::AbstractArray{T,3},
                              rk::AbstractVector,
                              obs::Vector{Int},
                              eps_sq::Real,
                              backend) where {T}
    r, count = size(V_cat, 2), size(V_cat, 3)
    count == 0 && return

    # Compute column norms² of V_cat on device
    norms_d = similar(V_cat, real(T), r, count)
    _col_norms_sq_kernel!(backend)(norms_d, V_cat; ndrange=(r, count))

    norms_cpu = Array(norms_d)  # [r, count] — small transfer

    # Per-tile: descending sort → truncation rank k
    perms_cpu = ones(Int32, r, count)
    ks_cpu    = ones(Int32, count)
    for b in 1:count
        s    = @view norms_cpu[:, b]
        perm = sortperm(s; rev=true)          # descending norm order
        perms_cpu[:, b] .= perm

        # Greedily remove cheapest columns while budget allows
        budget = real(T)(eps_sq)
        k = r
        for idx in r:-1:1
            cost = s[perm[idx]]
            cost <= budget || break
            budget -= cost
            k -= 1
        end
        ks_cpu[b] = Int32(k)
        rk[obs[b]] = eltype(rk)(k)
    end

    # Apply compaction kernel with an out-of-place temp buffer
    perms_d = similar(U_cat, Int32, r, count)
    ks_d    = similar(U_cat, Int32, count)
    copyto!(perms_d, perms_cpu)
    copyto!(ks_d,    ks_cpu)

    U_tmp = similar(U_cat)
    V_tmp = similar(V_cat)
    compact_k! = _compact_col_kernel!(backend)
    compact_k!(U_tmp, U_cat, perms_d, ks_d; ndrange=(size(U_cat, 1), r, count))
    compact_k!(V_tmp, V_cat, perms_d, ks_d; ndrange=(size(V_cat, 1), r, count))

    copyto!(U_cat, U_tmp)
    copyto!(V_cat, V_tmp)
end

# ─── Core: compress one off-diagonal tile category ───────────────────────────
#
# alg = :cholqr2  → Algorithm 1 (fixed rank):   Y → cholqr2 → U
# alg = :cholqr   → Algorithm 2 (adaptive rank): Y → cholqr → NS → U
#                    followed by truncation on V column norms
#
# The A tiles are accessed via non-allocating views (no packing).
# On CUDA/AMDGPU these dispatch to pointer-batched cuBLAS/rocBLAS via the
# existing Vector{<:StridedCuMatrix} / Vector{<:StridedROCMatrix} paths.

function _compress_offdiag_category!(U::AbstractArray{T,3},
                                     V::AbstractArray{T,3},
                                     A::AbstractMatrix{T},
                                     layout::TileMap,
                                     obs::Vector{Int},
                                     obs_d::AbstractVector{<:Integer},
                                     m_cat::Int,
                                     n_cat::Int,
                                     r_eff::Int,
                                     backend;
                                     alg::Symbol   = :cholqr2,
                                     ns_iters::Int = 1,
                                     eps_sq        = nothing,
                                     rk            = nothing) where {T}
    count = length(obs)
    count == 0 && return

    # Non-allocating views into A — no tile data is copied here.
    # lda = size(A,1) is uniform across all tiles in this category.
    A_views = _tile_views(A, layout, obs, m_cat, n_cat)

    # Allocate sketch and sketch output (small: proportional to r, not to tile area)
    Ω_cat = similar(A, T, n_cat, r_eff, count)
    Y_cat = similar(A, T, m_cat, r_eff, count)
    Random.randn!(Ω_cat)

    Ω_views = [view(Ω_cat, :, :, k) for k in 1:count]
    Y_views = [view(Y_cat, :, :, k) for k in 1:count]

    # Step 2.1: range sampling  Y = A_tile · Ω  (pointer-batched GEMM)
    gemm_batched!('N', 'N', one(T), A_views, Ω_views, zero(T), Y_views)

    # Step 2.2: orthogonalize Y → U_cat  (Y_cat overwritten in-place)
    if alg === :cholqr2
        _cholqr2!(Y_cat, backend)
    else
        _cholqr!(Y_cat, backend)
        _newton_schulz!(Y_cat; niters=ns_iters)
    end
    # Y_cat now holds U_cat

    if alg === :cholqr2
        # ── Algorithm 1: scatter U, compute V directly into storage ──────────

        scatter_k! = _scatter_offdiag_kernel!(backend)
        scatter_k!(U, Y_cat, obs_d; ndrange=(m_cat, r_eff, count))

        # U_views and V_views are also non-allocating (lda = size(U/V, 1) = b)
        U_views = [view(U, 1:m_cat, 1:r_eff, ob) for ob in obs]
        V_views = [view(V, 1:n_cat, 1:r_eff, ob) for ob in obs]

        # Step 2.3: co-range  V = A^T · U  (pointer-batched GEMM)
        gemm_batched!('T', 'N', one(T), A_views, U_views, zero(T), V_views)

    else
        # ── Algorithm 2: keep V in cat buffer for truncation ─────────────────

        V_cat = similar(V, T, n_cat, r_eff, count)
        U_views_cat = [view(Y_cat, :, :, k) for k in 1:count]
        V_views_cat = [view(V_cat, :, :, k) for k in 1:count]

        # Step 2.3: co-range  V = A^T · U_cat  (pointer-batched GEMM)
        gemm_batched!('T', 'N', one(T), A_views, U_views_cat, zero(T), V_views_cat)

        # Step 2.4: truncate U_cat and V_cat by V column norms
        _truncate_category!(Y_cat, V_cat, rk, obs, eps_sq, backend)

        # Scatter compacted U_cat and V_cat → storage
        scatter_k! = _scatter_offdiag_kernel!(backend)
        scatter_k!(U, Y_cat, obs_d; ndrange=(m_cat, r_eff, count))
        scatter_k!(V, V_cat, obs_d; ndrange=(n_cat, r_eff, count))
    end
end

# ─── compress! ───────────────────────────────────────────────────────────────

"""
    compress!(A_tlr, A; tol=nothing, ns_iters=1)

Compress dense matrix `A` into the TLR representation `A_tlr` in-place.

## Algorithm 1 — fixed rank (`tol === nothing`, default)

Each off-diagonal tile is approximated at rank `r = maxrank(A_tlr)` via a
randomized range finder followed by **two-pass Cholesky QR** (cholqr2):

    Ωᵢ = randn(nᵢ, r),   Yᵢ = Aᵢ Ωᵢ
    Uᵢ = cholqr2(Yᵢ),    Vᵢ = Aᵢᵀ Uᵢ

Cholqr2 drives the orthogonality error of `U` from `O(κ(Y)²ε)` to `O(κ(Y)²ε²)`,
making the column norms of `U` accurate in float32.

## Algorithm 2 — adaptive rank (`tol = ε`)

Same as Algorithm 1 but with a **single cholqr pass + Newton-Schulz refinement**,
followed by truncation of V columns whose total squared norm lies within the
error budget `ε²`:

    error(full)² + Σ_{j removed} ‖Vᵢ[:,j]‖² ≤ ε²

Columns are sorted by `‖Vᵢ[:,j]‖` descending before truncation, so the kept
columns span the most energetic part of the range. Per-tile ranks are stored
in `ranks(A_tlr)`.

## Notes

- Off-diagonal tiles are batched into interior / right-boundary / bottom-boundary
  groups.  Within each group the GEMM is launched as a single pointer-batched call
  (no tile packing): `view(A, ...)` produces `StridedMatrix` windows that are
  passed directly to `gemm_batched!`, which dispatches to `cublasGemmBatchedEx`
  (CUDA) or the rocBLAS equivalent (AMDGPU).
- Diagonal tiles are copied densely (`compress_diag = false` required).
- Currently requires square matrices with square tiles.
"""
function compress!(A_tlr::TLRMatrix{<:Any,T}, A::AbstractMatrix{T};
                   tol::Union{Nothing,Real} = nothing,
                   ns_iters::Int = 1) where {T}
    compress_diag(A_tlr) && throw(ArgumentError("compress! requires compress_diag=false"))
    size(A, 1) == A_tlr.m && size(A, 2) == A_tlr.n || throw(DimensionMismatch("A dimensions must match A_tlr"))
    A_tlr.m == A_tlr.n || throw(ArgumentError("compress! currently requires square matrices"))
    layout = A_tlr.layout
    layout.tile_m == layout.tile_n || throw(ArgumentError("compress! currently requires square tiles"))

    backend = A_tlr.backend
    alg     = tol === nothing ? :cholqr2 : :cholqr
    eps_sq  = tol === nothing ? nothing  : T(tol)^2

    # Step 1: dense diagonal tiles
    _copy_diag!(dense_diag(A_tlr), A, layout, backend)
    noffdiag_tiles(layout) == 0 && return A_tlr

    r  = maxrank(A_tlr)
    b  = layout.tile_m
    tail = A_tlr.n % b

    U  = left_factors(A_tlr)
    V  = right_factors(A_tlr)
    rk = ranks(A_tlr)
    fill!(U, zero(T))
    fill!(V, zero(T))

    int_obs, rb_obs, bb_obs = _classify_offdiag_tiles(layout)

    # Transfer index arrays to device
    function _to_device(obs::Vector{Int})
        d = similar(U, Int, length(obs))
        copyto!(d, obs)
        return d
    end
    int_obs_d = _to_device(int_obs)
    rb_obs_d  = _to_device(rb_obs)
    bb_obs_d  = _to_device(bb_obs)

    # Effective rank per category (limits Cholesky to full-rank sketches)
    r_int = min(r, b)
    r_rb  = tail > 0 ? min(r, tail) : 0
    r_bb  = tail > 0 ? min(r, tail) : 0

    # Steps 2.1–2.3 (concurrent stream launch is future work)
    common = (; alg, ns_iters, eps_sq, rk)
    _compress_offdiag_category!(U, V, A, layout, int_obs, int_obs_d, b,    b,    r_int, backend; common...)
    tail > 0 && _compress_offdiag_category!(U, V, A, layout, rb_obs,  rb_obs_d,  b,    tail, r_rb,  backend; common...)
    tail > 0 && _compress_offdiag_category!(U, V, A, layout, bb_obs,  bb_obs_d,  tail, b,    r_bb,  backend; common...)

    # For Algorithm 1 ranks are uniform; set them now
    if alg === :cholqr2
        T_rank = eltype(rk)
        for ob in int_obs; rk[ob] = T_rank(r_int); end
        if tail > 0
            for ob in rb_obs; rk[ob] = T_rank(r_rb); end
            for ob in bb_obs; rk[ob] = T_rank(r_bb); end
        end
    end

    return A_tlr
end
