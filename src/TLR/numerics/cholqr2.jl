@inline function _cholqr_shift_coeff(::Type{Tgram}, m::Int, s::Int) where {Tgram}
    RT = real(Tgram)
    return RT(11 * (m * s + s * (s + 1))) * (eps(RT) / RT(2))
end

"""
    cholqr_rank_rtol_sq(Tstorage, Tgram, m, s)

Default relative squared-energy floor for removing numerically null CholQR2
coordinates. It accounts for both storage rounding in the returned coefficient
factor and the shifted high-precision Gram factorization.
"""
@inline function cholqr_rank_rtol_sq(::Type{Tstorage},
                                     ::Type{Tgram},
                                     m::Int,
                                     s::Int,
) where {Tstorage,Tgram}
    # Up to `s` null coordinates may each carry one floor unit. Summing those
    # units remains a numerical-rank threshold rather than an approximation
    # budget for the represented matrix.
    coordinate_count = Float64(max(s, 1))
    storage_floor = coordinate_count * Float64(eps(real(Tstorage)))^2
    shift_floor =
        coordinate_count * Float64(_cholqr_shift_coeff(Tgram, m, s))^2
    return max(storage_floor, shift_floor)
end

# Smallest power of two >= n, clamped to `cap`; workgroup width for the
# shared-memory reduction in `_cholqr_shift_kernel!`.
@inline _reduce_threads(n::Int, cap::Int=1024) =
    min(max(nextpow(2, max(n, 1)), 1), cap)

# KernelAbstractions' CPU backend requires every barrier to occur literally in
# the kernel body. Keep this reduction as a straight-line sequence.
@kernel function _cholqr_shift_kernel!(G::AbstractArray{Thi,3},
                                       coeff::RT,
                                       multipliers,
                                       ::Val{NT}) where {Thi,RT,NT}
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

    # `coeff == 0` requests a genuinely unshifted factorization: `potrf` is then
    # allowed to break down, and that breakdown is the caller's rank signal
    # (breakdown coincides with `κ(Y) > u^{-1/2}`, exactly where CholeskyQR2
    # loses its `O(u)` orthogonality guarantee). With a nonzero coefficient an
    # all-zero panel would regularize to nothing, so the eps floor is kept there.
    reg = coeff * (@inbounds smax[1]) * RT(real(@inbounds multipliers[b]))
    shift = Thi(ifelse(coeff > zero(RT),
                       ifelse(reg > zero(RT), reg, eps(RT)),
                       zero(RT)))

    rr = size(G, 1)
    i = tid
    while i <= rr
        @inbounds G[i, i, b] += shift
        i += NT
    end
end

@kernel function _copy_upper_triangle_kernel!(dest::AbstractArray{T,3},
                                              src::AbstractArray{Thi,3}) where {T,Thi}
    row, col, batch = @index(Global, NTuple)
    @inbounds dest[row, col, batch] =
        row <= col ? T(src[row, col, batch]) : zero(T)
end

@inline function _shifted_cholesky!(G_hi::AbstractArray{Thi,3},
                                    Y_hi,
                                    multipliers,
                                    coeff,
                                    nt_shift;
                                    escalate::Bool,
                                    status_out=nothing,
) where {Thi}
    backend = get_backend(G_hi)
    count = size(G_hi, 3)
    RT = real(Thi)

    gemm_batched!(_adjoint_blas_char(Thi),
                  'N', one(Thi), Y_hi, Y_hi, zero(Thi), G_hi,
    )
    _cholqr_shift_kernel!(backend, nt_shift)(
        G_hi, coeff, multipliers, Val{nt_shift}();
        ndrange=(nt_shift * count,), workgroupsize=nt_shift,
    )
    result = potrf_batched!('U', G_hi)

    if escalate && result isa Tuple
        status = Array(result[2])
        attempts = 0
        while any(!iszero, status) && attempts < 40
            host_mult = Array(multipliers)
            @inbounds for b in eachindex(status)
                status[b] == 0 || (host_mult[b] *= RT(2))
            end
            copyto!(multipliers, host_mult)
            gemm_batched!(_adjoint_blas_char(Thi),
                          'N', one(Thi), Y_hi, Y_hi, zero(Thi), G_hi,
            )
            _cholqr_shift_kernel!(backend, nt_shift)(
                G_hi, coeff, multipliers, Val{nt_shift}();
                ndrange=(nt_shift * count,), workgroupsize=nt_shift,
            )
            result = potrf_batched!('U', G_hi)
            status = Array(result[2])
            attempts += 1
        end
        any(!iszero, status) &&
            throw(LinearAlgebra.PosDefException(findfirst(!iszero, status)))
    end

    # Per-member `potrf` status, for callers that read breakdown as information
    # rather than as an error (see the `coeff == 0` note above).
    if status_out !== nothing && result isa Tuple
        copyto!(status_out, result[2])
    end
    return G_hi
end

"""
    mixed_cholqr2_basis!(
        Q, Y_hi, G_hi, R_work, R_tiles, Q_tiles, multipliers,
    ) -> Q

Two FKNYY-shifted Cholesky-QR passes. `Q` is overwritten by its approximately
orthonormal basis. Gram matrices use the higher precision of `Y_hi`; triangular
solves retain `eltype(Q)`.

This is the minimal-workspace path used by randomized compression. It
intentionally does not retain the two triangular factors.
"""
function mixed_cholqr2_basis!(Q::AbstractArray{T,3},
                              Y_hi::AbstractArray{Thi,3},
                              G_hi,
                              R_work,
                              R_tiles,
                              Q_tiles,
                              multipliers,
) where {T,Thi}
    count = size(Q, 3)
    count == 0 && return Q

    m, r = size(Y_hi, 1), size(Y_hi, 2)
    RT = real(Thi)
    coeff = _cholqr_shift_coeff(Thi, m, r)
    nt_shift = _reduce_threads(r)
    fill!(multipliers, one(RT))

    @unroll for pass in 1:2
        Y_hi .= Q
        _shifted_cholesky!(G_hi,
                           Y_hi, multipliers, coeff, nt_shift; escalate=pass == 1,
        )
        R_work .= G_hi
        trsm_batched!('R', 'U', 'N', 'N', R_tiles, Q_tiles, one(T))
        fill!(multipliers, one(RT))
    end

    return Q
end

# Existing internal callers and downstream experiments used this name.
const cholqr2! = mixed_cholqr2_basis!

"""
    CholQR2FactorWorkspace(Q, V, Y_hi, G_hi, R1, R2, multipliers)

Allocation-free repeated-call bundle for [`mixed_cholqr2_factor!`](@ref).
`Q` contains the input panel on entry and the basis on return. `V` receives the
coefficient factor satisfying `Q * V' ≈ Q_input`. Returning the coefficient in
TLR factor orientation makes it directly consumable by fused energy pruning.
"""
struct CholQR2FactorWorkspace{QT,VT,YT,GT,RT,MV,QViews,VViews,RViews}
    Q::QT
    V::VT
    Y_hi::YT
    G_hi::GT
    R1::RT
    R2::RT
    multipliers::MV
    Q_tiles::QViews
    V_tiles::VViews
    R1_tiles::RViews
    R2_tiles::RViews
end

function CholQR2FactorWorkspace(Q::AbstractArray{T,3},
                                V::AbstractArray{T,3},
                                Y_hi::AbstractArray{Thi,3},
                                G_hi::AbstractArray{Thi,3},
                                R1::AbstractArray{T,3},
                                R2::AbstractArray{T,3},
                                multipliers,
) where {T,Thi}
    m, r, count = size(Q)
    size(V) == (r, r, count) ||
        throw(DimensionMismatch("V must have size ($r, $r, $count)"))
    size(Y_hi) == size(Q) ||
        throw(DimensionMismatch("Y_hi must have the same dimensions as Q"))
    size(G_hi) == size(V) ||
        throw(DimensionMismatch("G_hi must have size ($r, $r, $count)"))
    size(R1) == size(V) == size(R2) ||
        throw(DimensionMismatch("R1 and R2 must have the same dimensions as V"))
    length(multipliers) == count ||
        throw(DimensionMismatch("multipliers must have length $count"))

    return CholQR2FactorWorkspace(
        Q, V, Y_hi, G_hi, R1, R2, multipliers,
        _batch_views(Q), _batch_views(V), _batch_views(R1), _batch_views(R2),
    )
end

"""
    mixed_cholqr_pass!(ws, Rdest, Rdest_tiles; shift_coeff, escalate, status) -> ws

**One** mixed-precision (optionally shifted) Cholesky-QR pass over `ws.Q`:
Gram in the promoted type, Cholesky, then a working-precision `trsm` that
overwrites `ws.Q` with the orthonormalized panel. The triangular factor is
written to `Rdest`.

Exposed separately because the two passes of a CholeskyQR2 are not always
adjacent. A blocked range finder must interleave them with the projection
against the existing basis --- `(P·CholQR)²`, not `P²·CholQR²` --- since the
first Cholesky amplifies a near-null column's `O(u)` overlap with that basis to
`O(1)`, and only a projection placed *after* that amplification can remove it.
See `ara_build_basis!` and `docs/TODO.md` worklog item 9.
"""
function mixed_cholqr_pass!(ws::CholQR2FactorWorkspace, Rdest, Rdest_tiles;
                            shift_coeff=nothing,
                            escalate::Bool=true,
                            status=nothing,
)
    Q = ws.Q
    count = size(Q, 3)
    count == 0 && return ws

    T = eltype(Q)
    Thi = eltype(ws.Y_hi)
    RT = real(Thi)
    m, r = size(Q, 1), size(Q, 2)
    coeff = shift_coeff === nothing ? _cholqr_shift_coeff(Thi, m, r) :
            RT(shift_coeff)
    nt_shift = _reduce_threads(r)
    backend = get_backend(Q)

    fill!(ws.multipliers, one(RT))
    ws.Y_hi .= Q
    _shifted_cholesky!(ws.G_hi,
                       ws.Y_hi, ws.multipliers, coeff, nt_shift;
                       escalate, status_out=status,
    )
    _copy_upper_triangle_kernel!(backend)(
        Rdest, ws.G_hi; ndrange=size(Rdest),
    )
    trsm_batched!('R', 'U', 'N', 'N', Rdest_tiles, ws.Q_tiles, one(T))
    fill!(ws.multipliers, one(RT))
    return ws
end

"""
    mixed_cholqr2_factor!(ws::CholQR2FactorWorkspace) -> ws

Factor-producing mixed-precision shifted CholQR2. For the panel `X` initially
stored in `ws.Q`, computes

    X ≈ ws.Q * ws.V'

where `ws.Q` is the final basis and `ws.V' = R₂R₁` is the composite triangular
factor from both CholQR passes. The hot path uses prebuilt concrete batch
vectors and does not allocate an intermediate energy array.

## Keywords

`shift_coeff` — relative Cholesky shift, as a multiple of the panel's largest
squared column norm. Defaults to the value of Fukaya, Nakatsukasa, Yanagisawa &
Yamamoto (SIAM J. Sci. Comput. 42(1), 2020), which provably prevents breakdown
for any conditioning. Pass `0` for an unshifted factorization, which is the
right choice when the caller reads breakdown as a rank signal rather than an
error: `potrf` fails exactly when `κ(Y) ≳ u^{-1/2}`, which is precisely where
CholeskyQR2 loses its `O(u)` orthogonality guarantee (Yamamoto, Nakatsukasa,
Yanagisawa & Fukaya, ETNA 44, 2015), so success certifies the result and failure
certifies rank deficiency. A nonzero shift instead places a floor of
`√shift_coeff` under every `diag(R)` entry relative to the panel scale, which
silently defeats any rank or convergence test finer than that.

`escalate` — when `true` (default), double the shift until every member's
`potrf` succeeds and throw `PosDefException` if it never does. Set `false` to
let breakdown stand.

`status` — optional vector receiving the per-member `potrf` info from the first
pass. Only meaningful with `escalate=false`.
"""
function mixed_cholqr2_factor!(ws::CholQR2FactorWorkspace;
                               shift_coeff=nothing,
                               escalate::Bool=true,
                               status=nothing,
)
    Q = ws.Q
    count = size(Q, 3)
    count == 0 && return ws

    T = eltype(Q)
    Thi = eltype(ws.Y_hi)
    RT = real(Thi)
    m, r = size(Q, 1), size(Q, 2)
    coeff = shift_coeff === nothing ? _cholqr_shift_coeff(Thi, m, r) :
            RT(shift_coeff)
    nt_shift = _reduce_threads(r)
    backend = get_backend(Q)

    mixed_cholqr_pass!(ws, ws.R1, ws.R1_tiles; shift_coeff=coeff, escalate, status)

    fill!(ws.multipliers, one(RT))
    ws.Y_hi .= Q
    _shifted_cholesky!(ws.G_hi,
                       ws.Y_hi, ws.multipliers, coeff, nt_shift; escalate=false,
    )
    _copy_upper_triangle_kernel!(backend)(
        ws.R2, ws.G_hi; ndrange=size(ws.R2),
    )
    trsm_batched!('R',
                  'U', 'N', 'N', ws.R2_tiles, ws.Q_tiles, one(T),
    )

    gemm_batched!(_adjoint_blas_char(T),
                  _adjoint_blas_char(T),
                  one(T), ws.R1_tiles, ws.R2_tiles, zero(T), ws.V_tiles,
    )
    fill!(ws.multipliers, one(RT))
    return ws
end
