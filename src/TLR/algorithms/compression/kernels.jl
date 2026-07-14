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

@inline function _norm_launch(backend, tn::Int)
    W = unwrap(SUBGROUP_SIZE(typeof(backend)))
    nsub = tn >= 8 ? 8 : tn >= 4 ? 4 : tn >= 2 ? 2 : 1
    return W, nsub, W * nsub
end

# Smallest power of two >= n, clamped to `cap`; workgroup width for the
# shared-memory tree reductions below (they require a power-of-two thread count).
@inline _reduce_threads(n::Int, cap::Int=1024) = min(max(nextpow(2, max(n, 1)), 1), cap)

#= Workgroup tree reductions below are written as a *fully unrolled, straight-line*
   halving sequence rather than a `while`/helper/macro. This is deliberate and load-
   bearing: KernelAbstractions' CPU backend splits a kernel into segments at each
   lexical `@synchronize`, so a barrier is only captured when it appears literally in
   the kernel body — a `@synchronize` inside a loop, an inlined function, or a macro
   expansion silently falls through to the "used outside kernel" error. `NT` is a
   compile-time `Val`, so every `NT > h` guard folds away and the dead levels vanish
   on GPU. Steps run from h=512 down to cover any NT ≤ 1024. =#

# Per-slab FKNYY shift: 11(mS + S(S+1)) u_s max(diag(G)). One workgroup per slab,
# `NT` threads (power of two) cooperatively reduce the diagonal: each thread folds
# a strided slice into a register max, then a shared-memory tree finds the slab
# max. All threads then add the shift to their strided diagonal entries.
@kernel function _cholqr_shift_kernel!(
    G::AbstractArray{Thi,3}, coeff::RT, multipliers, ::Val{NT}) where {Thi,RT,NT}
    b = @index(Group, Linear)
    tid = @index(Local, Linear)
    smax = @localmem RT (NT,)
    r = size(G, 1)

    mx = zero(RT)
    i = tid
    while i <= r
        @inbounds mx = max(mx, RT(real(G[i, i, b])))
        i += NT
    end
    @inbounds smax[tid] = mx

    @synchronize
    if NT > 512 && tid <= 512; @inbounds smax[tid] = max(smax[tid], smax[tid+512]); end
    @synchronize
    if NT > 256 && tid <= 256; @inbounds smax[tid] = max(smax[tid], smax[tid+256]); end
    @synchronize
    if NT > 128 && tid <= 128; @inbounds smax[tid] = max(smax[tid], smax[tid+128]); end
    @synchronize
    if NT >  64 && tid <=  64; @inbounds smax[tid] = max(smax[tid], smax[tid+ 64]); end
    @synchronize
    if NT >  32 && tid <=  32; @inbounds smax[tid] = max(smax[tid], smax[tid+ 32]); end
    @synchronize
    if NT >  16 && tid <=  16; @inbounds smax[tid] = max(smax[tid], smax[tid+ 16]); end
    @synchronize
    if NT >   8 && tid <=   8; @inbounds smax[tid] = max(smax[tid], smax[tid+  8]); end
    @synchronize
    if NT >   4 && tid <=   4; @inbounds smax[tid] = max(smax[tid], smax[tid+  4]); end
    @synchronize
    if NT >   2 && tid <=   2; @inbounds smax[tid] = max(smax[tid], smax[tid+  2]); end
    @synchronize
    if NT >   1 && tid == 1;   @inbounds smax[1]   = max(smax[1],   smax[2]);       end
    @synchronize

    reg = coeff * (@inbounds smax[1]) * RT(real(@inbounds multipliers[b]))
    # Any positive value works for a zero Gram because the solve leaves Y=0.
    shift = Thi(ifelse(reg > zero(RT), reg, eps(RT)))

    # `r` is recomputed here: KA's CPU backend splits the kernel at each
    # `@synchronize`, so plain locals bound before the barrier are out of scope now.
    rr = size(G, 1)
    i = tid
    while i <= rr
        @inbounds G[i, i, b] += shift
        i += NT
    end
end

# Per-tile squared Frobenius norm of the dense tiles, with hp accumulation.
# `NT = W·nsub` threads per tile: `nsub` subgroups stride over columns while the
# `W` lanes of each subgroup stride over rows, so a subgroup's lanes read adjacent
# column-major words (coalesced). Register partials are summed by a shared tree.
@kernel function _tile_norm_sq_kernel!(out::AbstractVector{Tout},
    A::AbstractMatrix{T}, p0s, q0s, tm::Int, tn::Int,
    ::Val{W}, ::Val{NT}) where {Tout,T,W,NT}
    ob = @index(Group, Linear)
    tid = @index(Local, Linear)
    partial = @localmem Float64 (NT,)
    p0 = Int(@inbounds p0s[ob]) - 1
    q0 = Int(@inbounds q0s[ob]) - 1
    lane = (tid - 1) % W + 1
    sg = (tid - 1) ÷ W + 1
    nsub = NT ÷ W
    acc = 0.0
    col = sg
    while col <= tn
        row = lane
        while row <= tm
            @inbounds acc += _abs2_f64(A[p0+row, q0+col])
            row += W
        end
        col += nsub
    end
    @inbounds partial[tid] = acc

    @synchronize
    if NT > 512 && tid <= 512; @inbounds partial[tid] += partial[tid+512]; end
    @synchronize
    if NT > 256 && tid <= 256; @inbounds partial[tid] += partial[tid+256]; end
    @synchronize
    if NT > 128 && tid <= 128; @inbounds partial[tid] += partial[tid+128]; end
    @synchronize
    if NT >  64 && tid <=  64; @inbounds partial[tid] += partial[tid+ 64]; end
    @synchronize
    if NT >  32 && tid <=  32; @inbounds partial[tid] += partial[tid+ 32]; end
    @synchronize
    if NT >  16 && tid <=  16; @inbounds partial[tid] += partial[tid+ 16]; end
    @synchronize
    if NT >   8 && tid <=   8; @inbounds partial[tid] += partial[tid+  8]; end
    @synchronize
    if NT >   4 && tid <=   4; @inbounds partial[tid] += partial[tid+  4]; end
    @synchronize
    if NT >   2 && tid <=   2; @inbounds partial[tid] += partial[tid+  2]; end
    @synchronize
    if NT >   1 && tid == 1;   @inbounds partial[1]   += partial[2];       end
    @synchronize
    if tid == 1
        @inbounds out[ob] = Tout(partial[1])
    end
end

# Per-slab squared Frobenius norm of a packed [tm, tn, ntiles] batch (hp accumulation).
@kernel function _tile_norm_sq_kernel!(out::AbstractVector{Tout},
    P::AbstractArray{T,3}, ::Val{W}, ::Val{NT}) where {Tout,T,W,NT}
    ob = @index(Group, Linear)
    tid = @index(Local, Linear)
    partial = @localmem Float64 (NT,)
    tm = size(P, 1)
    tn = size(P, 2)
    lane = (tid - 1) % W + 1
    sg = (tid - 1) ÷ W + 1
    nsub = NT ÷ W
    acc = 0.0
    col = sg
    while col <= tn
        row = lane
        while row <= tm
            @inbounds acc += _abs2_f64(P[row, col, ob])
            row += W
        end
        col += nsub
    end
    @inbounds partial[tid] = acc

    @synchronize
    if NT > 512 && tid <= 512; @inbounds partial[tid] += partial[tid+512]; end
    @synchronize
    if NT > 256 && tid <= 256; @inbounds partial[tid] += partial[tid+256]; end
    @synchronize
    if NT > 128 && tid <= 128; @inbounds partial[tid] += partial[tid+128]; end
    @synchronize
    if NT >  64 && tid <=  64; @inbounds partial[tid] += partial[tid+ 64]; end
    @synchronize
    if NT >  32 && tid <=  32; @inbounds partial[tid] += partial[tid+ 32]; end
    @synchronize
    if NT >  16 && tid <=  16; @inbounds partial[tid] += partial[tid+ 16]; end
    @synchronize
    if NT >   8 && tid <=   8; @inbounds partial[tid] += partial[tid+  8]; end
    @synchronize
    if NT >   4 && tid <=   4; @inbounds partial[tid] += partial[tid+  4]; end
    @synchronize
    if NT >   2 && tid <=   2; @inbounds partial[tid] += partial[tid+  2]; end
    @synchronize
    if NT >   1 && tid == 1;   @inbounds partial[1]   += partial[2];       end
    @synchronize
    if tid == 1
        @inbounds out[ob] = Tout(partial[1])
    end
end

"""
    _prune_rank_kernel(U, V, rk, norm_err_sq, eps_sq, rel,
                            R_keep, delta_floor, ::Val{S}, ::Val{W})

Select the retained rank for each off-diagonal tile and compact the selected
columns in-place within the maxrank-wide output panels. Only the first `S`
columns contain sketch factors; columns after the retained rank are cleared.
One workgroup handles one tile. During energy accounting, one thread accumulates
each column norm directly, avoiding the former `S×W` shared partial array.
During compaction, the local threads are split into subgroups of width `W` and
each subgroup cooperatively moves a U/V column pair.

The squared error of the truncated factorization decomposes, with orthonormal
`Q`, as

    ‖A - Q_k V_k'‖² = resid + Σ dropped ‖v_j‖²
    resid = ‖A‖² - Σ_j ‖v_j‖²

where `resid` is the randQB_EI range-capture error left by the sketch. The
kernel greedily drops the currently-smallest remaining `V`-column energy while
it fits in the remaining error budget, then drops extra smallest columns if
needed to satisfy `R_keep = min(maxrank, S)`. Compaction pairs the leftmost
hole with the rightmost retained column, producing a deterministic minimum-move
map without a second factor buffer.

Each subgroup moves one U/V column pair and threads run along rows so adjacent
threads touch contiguous column-major memory.
"""
@kernel function _prune_rank_kernel(
    U::AbstractArray{T,3}, V::AbstractArray{T,3},
    rk::AbstractVector, norm_err_sq::AbstractVector{Terr},
    eps_sq::Float64, rel::Bool, R_keep::Int, delta_floor::Float64,
    ::Val{S}, ::Val{W}
) where {T,Terr,S,W}

    ob = @index(Group, Linear)
    tid = @index(Local, Linear)

    nthreads = @uniform @groupsize()[1]

    norms = @localmem Float64 (S,)          # unsorted column norms
    dropped_flag = @localmem Int32 (S,)     # 1 if column is dropped, else 0
    order = @localmem Int32 (S,)            # columns sorted by ascending energy
    move_dst = @localmem Int32 (S,)         # holes filled from retained tail columns
    k_buf = @localmem Int32 (2,)            # retained rank, number of moves

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

        nA_sq = Float64(real(@inbounds norm_err_sq[ob]))

        resid = max(nA_sq - total, 0.0)

        # resid = ‖A‖² − ‖V‖² 
        epsT = Float64(eps(real(T)))
        resid_floor = Float64(size(U, 1)) * epsT * nA_sq
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

        # Construct a deterministic minimum-move compaction map: pair the
        # leftmost hole with the rightmost retained column until the first K
        # positions are full. `order` is dead after pruning and stores sources.
        left = 1
        right = S
        nmoves = 0
        while true
            @inbounds while left <= S && dropped_flag[left] == 0
                left += 1
            end
            @inbounds while right >= 1 && dropped_flag[right] != 0
                right -= 1
            end
            left < right || break
            nmoves += 1
            @inbounds move_dst[nmoves] = Int32(left)
            @inbounds order[nmoves] = Int32(right)
            left += 1
            right -= 1
        end

        @inbounds rk[ob] = eltype(rk)(k_val)
        @inbounds norm_err_sq[ob] = Terr(resid + dropped)
        @inbounds k_buf[1] = Int32(k_val)
        @inbounds k_buf[2] = Int32(nmoves)
    end

    @synchronize

    # Each subgroup moves one paired U/V column at a time. Destinations are
    # dropped columns, hence no move overwrites another move's source.
    lane = (tid - 1) % W + 1
    subgroup = (tid - 1) ÷ W + 1
    nsubgroups = nthreads ÷ W
    nmoves = Int(@inbounds k_buf[2])
    move = subgroup
    while move <= nmoves
        dst = Int(@inbounds move_dst[move])
        src = Int(@inbounds order[move])
        row = lane
        while row <= size(U, 1)
            @inbounds U[row, dst, ob] = U[row, src, ob]
            row += W
        end
        row = lane
        while row <= size(V, 1)
            @inbounds V[row, dst, ob] = V[row, src, ob]
            row += W
        end
        move += nsubgroups
    end

    @synchronize

    # The compact representation is the first K columns. Clear the tail so
    # padded downstream GEMMs cannot observe stale factors.
    # Recompute subgroup coordinates after the barrier for the CPU backend,
    # whose KernelAbstractions lowering does not preserve private values there.
    lane = (tid - 1) % W + 1
    subgroup = (tid - 1) ÷ W + 1
    nsubgroups = nthreads ÷ W
    k = Int(@inbounds k_buf[1])
    col = k + 1 + (subgroup - 1)
    while col <= size(U, 2)
        row = lane
        while row <= size(U, 1)
            @inbounds U[row, col, ob] = zero(T)
            row += W
        end
        row = lane
        while row <= size(V, 1)
            @inbounds V[row, col, ob] = zero(T)
            row += W
        end
        col += nsubgroups
    end
end
