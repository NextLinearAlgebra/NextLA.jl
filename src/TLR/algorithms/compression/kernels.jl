# ----------- helpers -------------------

# accumulation precision used for cholqr and norms accumulation
@inline _compress_accum_type(::Type{Float16}) = Float32
@inline _compress_accum_type(::Type{Float32}) = Float64
@inline _compress_accum_type(::Type{Float64}) = Float64
@inline _compress_accum_type(::Type{ComplexF32}) = ComplexF64
@inline _compress_accum_type(::Type{ComplexF64}) = ComplexF64
@inline _compress_accum_type(::Type{T}) where {T} = T

@inline _adjoint_blas_char(::Type{<:Complex}) = 'C'
@inline _adjoint_blas_char(::Type) = 'T'

@inline function _cholqr_shift_coeff(::Type{Tgram}, m::Int, s::Int) where {Tgram}
    RT = real(Tgram)
    return RT(11 * (m * s + s * (s + 1))) * (eps(RT) / RT(2))
end

"""
    _batch_views(A, r=size(A, 2)) -> Vector{<:AbstractMatrix}

Per-batch-entry views into a `[m, maxrank, n]` factor array, trimmed to the
first `r` columns. Used to build `gemm_batched!` operand vectors; shared with
`uncompress.jl`.
"""
@inline _batch_views(A::AbstractArray{T,3}, r::Int=size(A, 2)) where {T} =
    [view(A, :, 1:r, k) for k in axes(A, 3)]

# ------- kernels --------

@kernel function _copy_diag_kernel!(D::AbstractArray{T,3},
    A::AbstractMatrix{T}, tile_m::Int, tile_n::Int) where {T}
    row, col, batch = @index(Global, NTuple)
    p0 = (batch - 1) * tile_m + 1
    q0 = (batch - 1) * tile_n + 1
    @inbounds D[row, col, batch] = A[p0+row-1, q0+col-1]
end

"""
    _copy_diagonal_from_dense!(A_tlr, A) -> A_tlr

Populate `A_tlr`'s dense diagonal storage (`A_tlr.D` and, if present,
`A_tlr.D_corner`) from the corresponding tiles of dense matrix `A`.
"""
function _copy_diagonal_from_dense!(A_tlr::TLRDenseDiagMatrix{<:Any,T}, A::AbstractMatrix{T}) where {T}
    n_full_diag = _nfull_diag_tiles(A_tlr)
    bm, bn = nominal_tile_size(A_tlr)
    _copy_diag_kernel!(A_tlr.backend)(
        A_tlr.D, A, bm, bn;
        ndrange=(bm, bn, n_full_diag),
    )
    if size(A_tlr.D_corner, 3) != 0
        tile_k = ndiag_tiles(A_tlr)
        tm, tn = tile_size(A_tlr, tile_k, tile_k)
        copyto!(view(A_tlr.D_corner, 1:tm, 1:tn, 1), _dense_tile_view(A, A_tlr, tile_k, tile_k))
    end
    return _set_dense_diagonal_diagnostics!(A_tlr)
end

#= Squared magnitude accumulated in Float64: the error indicator is a difference
   of two nearly equal sums, so per-element squaring must not round in T. =#
@inline _abs2_f64(x::Real) = abs2(Float64(x))
@inline _abs2_f64(x::Complex) = abs2(Float64(real(x))) + abs2(Float64(imag(x)))

@inline function _sum_norm_partials(partial, ::Val{R}) where {R}
    total = 0.0
    @inbounds for i in 1:R
        total += partial[i]
    end
    return total
end

# Per-slab FKNYY shift: 11(mS + S(S+1)) u_s max(diag(G)).
@kernel function _cholqr_shift_kernel!(
    G::AbstractArray{Thi,3}, coeff::RT, multipliers) where {Thi,RT}
    b = @index(Group, Linear)
    j = @index(Local, Linear)
    shift = @localmem RT (1,)
    r = size(G, 1)
    if j == 1
        mx = zero(RT)
        @inbounds for i in 1:r
            d = RT(real(G[i, i, b]))
            mx = max(mx, d)
        end
        reg = coeff * mx * RT(@inbounds multipliers[b])
        # Any positive value works for a zero Gram because the solve leaves Y=0.
        @inbounds shift[1] = ifelse(reg > zero(RT), reg, eps(RT))
    end
    @synchronize
    @inbounds G[j, j, b] += Thi(shift[1])
end

# Per-tile squared Frobenius norm of the dense tiles, with hp accumulation
# TODO use warp intrinsics for the in-warp reduction
@kernel function _tile_norm_sq_kernel!(out::AbstractVector{Float64},
    A::AbstractMatrix{T}, p0s, q0s, tm::Int, tn::Int, ::Val{R}) where {T,R}
    ob = @index(Group, Linear)
    j = @index(Local, Linear)
    partial = @localmem Float64 (R,)
    p0 = Int(@inbounds p0s[ob]) - 1
    q0 = Int(@inbounds q0s[ob]) - 1
    acc = 0.0
    col = j
    while col <= tn
        @inbounds for row in 1:tm
            acc += _abs2_f64(A[p0+row, q0+col])
        end
        col += R
    end
    @inbounds partial[j] = acc
    @synchronize
    if j == 1
        @inbounds out[ob] = _sum_norm_partials(partial, Val{R}())
    end
end

# Per-slab squared Frobenius norm of a packed [tm, tn, ntiles] batch (hp accumulation).
@kernel function _tile_norm_sq_kernel!(out::AbstractVector{Float64},
    P::AbstractArray{T,3}, ::Val{R}) where {T,R}
    ob = @index(Group, Linear)
    j = @index(Local, Linear)
    partial = @localmem Float64 (R,)
    tm = size(P, 1)
    tn = size(P, 2)
    acc = 0.0
    col = j
    while col <= tn
        @inbounds for row in 1:tm
            acc += _abs2_f64(P[row, col, ob])
        end
        col += R
    end
    @inbounds partial[j] = acc
    @synchronize
    if j == 1
        @inbounds out[ob] = _sum_norm_partials(partial, Val{R}())
    end
end

"""
    _fused_truncate_kernel!(U_out, V_out, Q, V, rk, err_sq, normA_sq,
                            eps_sq, rel, R_keep, delta_floor,
                            ::Val{S}, ::Val{W})

Select the retained rank for each off-diagonal tile and gather the selected
columns from the `S`-wide sketch factors into the maxrank-wide output panels.
One workgroup handles one tile. Its local threads are split into subgroups of
width `W`; each subgroup cooperatively computes column energies for one sketch
column at a time, then lane 1 of each subgroup acts as that column's
representative for norm reduction.

The squared error of the truncated factorization decomposes, with orthonormal
`Q`, as

    ‖A - Q_k V_k'‖² = resid + Σ dropped ‖v_j‖²
    resid = ‖A‖² - Σ_j ‖v_j‖²

where `resid` is the randQB_EI range-capture error left by the sketch. The
kernel greedily drops the currently-smallest remaining `V`-column energy while
it fits in the remaining error budget, then drops extra smallest columns if
needed to satisfy `R_keep = min(maxrank, S)`. Surviving source columns are
compacted in original order.

The final gather runs threads along rows so consecutive threads touch
contiguous column-major memory.
"""
@kernel function _fused_truncate_kernel!(
    U_out::AbstractArray{T,3}, V_out::AbstractArray{T,3},
    Q::AbstractArray{T,3}, V::AbstractArray{T,3},
    rk::AbstractVector, err_sq::AbstractVector{Float64},
    normA_sq::AbstractVector{Float64},
    eps_sq::Float64, rel::Bool, R_keep::Int, delta_floor::Float64,
    ::Val{S}, ::Val{W}
) where {T,S,W}

    ob = @index(Group, Linear)
    tid = @index(Local, Linear)

    nthreads = @uniform @groupsize()[1]

    norms = @localmem Float64 (S,)          # unsorted column norms
    dropped_flag = @localmem Int32 (S,)     # 1 if column is dropped, else 0
    order = @localmem Int32 (S,)            # columns sorted by ascending energy
    kept_src = @localmem Int32 (S,)         # compacted surviving source columns
    k_buf = @localmem Int32 (1,)

    # Phase A — one thread per column. This removes the S×W shared reduction
    # buffer, which was the occupancy limiter for wide sketches.
    for col in tid:nthreads:S
        acc = 0.0
        for row in axes(V, 1)
            @inbounds acc += _abs2_f64(V[row, col, ob])
        end
        @inbounds norms[col] = acc
    end

    @synchronize

    # Phase B greedy tail removal.
    if tid == 1
        total = 0.0

        @inbounds for i in 1:S
            total += norms[i]
            dropped_flag[i] = Int32(0)
            order[i] = Int32(i)
        end

        # Stable insertion sort gives deterministic index tie-breaking and one
        # ordering shared by tolerance pruning and the hard rank cap.
        @inbounds for i in 2:S
            key = order[i]
            key_norm = norms[Int(key)]
            j = i - 1
            while j >= 1
                prev = order[j]
                prev_norm = norms[Int(prev)]
                (prev_norm < key_norm || (prev_norm == key_norm && prev < key)) && break
                order[j + 1] = prev
                j -= 1
            end
            order[j + 1] = key
        end

        nA_sq = @inbounds normA_sq[ob]

        resid = max(nA_sq - total, 0.0)

        # resid = ‖A‖² − ‖V‖² 
        epsT = Float64(eps(real(T)))
        resid_floor = Float64(size(Q, 1)) * epsT * nA_sq
        resid = ifelse(resid < resid_floor, 0.0, resid)

        # precision floor
        target = rel ? eps_sq * nA_sq : eps_sq
        # The Pythagorean energy identity is only accurate to the CholQR
        # orthogonality floor. Allow one additional floor unit for rounding in
        # the two independently accumulated energies; without it, exact-rank
        # Float64 tiles can randomly land a few ulps above the nominal floor.
        budget = max(target, 2.0 * delta_floor * nA_sq) - resid

        k_val = S
        dropped = 0.0

        next_drop = 1
        if budget >= 0.0
            while next_drop <= S
                col = @inbounds order[next_drop]
                nc = @inbounds norms[Int(col)]
                if nc <= budget
                    @inbounds dropped_flag[Int(col)] = Int32(1)
                    budget -= nc
                    dropped += nc
                    k_val -= 1
                    next_drop += 1
                else
                    break
                end
            end
        end

        # If tolerance keeps more than `maxrank` columns,
        # drop the smallest columns until the stored rank fits.
        while k_val > R_keep
            col = @inbounds order[next_drop]
            @inbounds dropped_flag[Int(col)] = Int32(1)
            dropped += @inbounds norms[Int(col)]
            next_drop += 1
            k_val -= 1
        end

        # Phase C - compact kept columns
        pos = 1

        @inbounds for col in 1:S
            if dropped_flag[col] == 0
                kept_src[pos] = Int32(col)
                pos += 1
            end
        end

        @inbounds rk[ob] = eltype(rk)(k_val)
        @inbounds err_sq[ob] = resid + dropped
        @inbounds k_buf[1] = Int32(k_val)
    end

    @synchronize

    k = Int(@inbounds k_buf[1])
    Rmax = size(U_out, 2)
    # U_out has size(Q,1)=tm rows, V_out has size(V,1)=tn rows; these differ on
    # boundary tiles, so each gather uses its own row bound.
    bU = size(U_out, 1)
    bV = size(V_out, 1)

    @inbounds for jj in 1:Rmax
        src = jj <= k ? Int(kept_src[jj]) : 0
        for row in tid:nthreads:bU
            U_out[row, jj, ob] = jj <= k ? Q[row, src, ob] : zero(T)
        end
        for row in tid:nthreads:bV
            V_out[row, jj, ob] = jj <= k ? V[row, src, ob] : zero(T)
        end
    end
end
