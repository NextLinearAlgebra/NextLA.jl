# R1: the factor-list sampler for one output tile (i,j) of X = α·A·B + β·C.
#
# Implements the two implicit operators of algorithm.tex §2 without ever
# materializing the b_m×b_n tile:
#
#   ApplyRight(Ω) = X_ij·Ω           (eq:apply, via eq:HT/eq:Y)
#   ApplyLeft(Q)  = X_ijᵀ·Q          (the transpose analogue)
#
# where X_ij = α Σ_ℓ A_iℓB_ℓj + β C_ij. Both cost a constant number of kernel
# launches regardless of the contraction count q_k: one batched GEMM for
# H (or G), one pointer-batched GEMM for T (or W), and one plain GEMM for the
# fused reduction -- this is what "the long dimension is touched only once"
# (algorithm.tex §2.1) means in code.
#
# Requires A tile-row-major and B tile-col-major, so `rowpanel`/`colpanel` give
# zero-copy [b, maxrank, q_k] views over the whole contraction. Other layout
# pairs are R5's concern (packing or a different fusion), not this file's.

"""
    TileCoupling(ops, i, j; alpha, compute)

Prologue coupling matrices `S_{iℓj} = (V^A_{iℓ})ᵀ U^B_{ℓj} ∈ R^{r_A×r_B}` for one
output tile `(i,j)`, one per contraction index `ℓ = 1..q_k` (eq:coupling).
Computed with a single strided-batched GEMM over the zero-copy row/column
panels, and retained across every ARA pass: `S` does not depend on the sketch.

`alpha` is *not* folded in here -- it is applied once, in
[`apply_right!`](@ref)/[`apply_left!`](@ref), matching eq:HT
(`T_{iℓj} := α S_{iℓj} H_{ℓj}`) rather than baking a scalar into retained state.
"""
struct TileCoupling{ST,RVT,RUT,CWT,CZT,T}
    S::ST        # rA × rB × qk
    rowV::RVT    # rowpanel(av, i): bk × rA × qk  (V^A_{i·})
    rowU::RUT    # rowpanel(au, i): bm × rA × qk  (U^A_{i·})
    colW::CWT    # colpanel(bw, j): bk × rB × qk  (U^B_{·j})
    colZ::CZT    # colpanel(bz, j): bn × rB × qk  (V^B_{·j})
    alpha::T
end

function TileCoupling(ops::LogicalTLROperands, i::Integer, j::Integer;
                      alpha, compute=nothing)
    rowV = rowpanel(ops.av, i)
    rowU = rowpanel(ops.au, i)
    colW = colpanel(ops.bw, j)
    colZ = colpanel(ops.bz, j)
    size(rowV, 3) == size(colW, 3) ||
        throw(DimensionMismatch("A's row panel and B's column panel disagree on q_k"))
    size(rowV, 1) == size(colW, 1) ||
        throw(DimensionMismatch("A's and B's contraction tile size (b_k) disagree"))

    T = eltype(rowV)
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    rA, rB, qk = size(rowV, 2), size(colW, 2), size(rowV, 3)
    S = similar(rowV, T, rA, rB, qk)
    if qk > 0
        precision_gemm_batched!(_adjoint_blas_char(T), 'N', one(T), rowV, colW,
                                zero(T), S, mode)
    end
    return TileCoupling(S, rowV, rowU, colW, colZ, T(alpha))
end

"""
    TileBetaTerm(Uc, Vc, beta)

The existing output tile `C_ij = U^C_{ij} (V^C_{ij})ᵀ`, folded into the implicit
operators as one further factor pair `(U^C_{ij}, β V^C_{ij})`
(algorithm.tex §"Folding in the existing βC_ij tile"). Passing `nothing` in its
place (in [`apply_right!`](@ref)/[`apply_left!`](@ref)) means `β == 0`.
"""
struct TileBetaTerm{UT,VT,T}
    Uc::UT   # bm × rC
    Vc::VT   # bn × rC
    beta::T
end

"""
    tile_factor_list(ops, i, j; alpha, beta=false, C=nothing, compute)
        -> (; coupling, beta_term)

Build the prologue for one output tile `(i,j)` of `X = α·op(A)·op(B) + β·C`:
the coupling matrices (always) and, when `β ≠ 0`, `C`'s own tile factors folded
in as one further factor pair. `C` must be a `LogicalTLROperand` over a
`TLRMatrix` (a zero-copy tile-factor view); required whenever `beta != 0`.
"""
function tile_factor_list(ops::LogicalTLROperands, i::Integer, j::Integer;
                          alpha, beta=false, C=nothing, compute=nothing)
    coupling = TileCoupling(ops, i, j; alpha, compute)
    beta_term = if iszero(beta)
        nothing
    else
        C === nothing && throw(ArgumentError("C must be supplied when beta != 0"))
        Uc, Vc = logical_tile_factors(C, Int(i), Int(j))
        TileBetaTerm(Uc, Vc, eltype(coupling.S)(beta))
    end
    return (; coupling, beta_term)
end

"""
    apply_right!(Y, coupling, beta_term, Ω; compute) -> Y

`Y ← X_ij·Ω`, where `X_ij = α Σ_ℓ A_iℓB_ℓj + βC_ij` (`β C_ij` included when
`beta_term !== nothing`) and `Ω` is `b_n × s`. `Y` must be `b_m × s`.

Cost: one batched GEMM for `H_{ℓj} = (V^B_{ℓj})ᵀΩ` (Ω broadcast over `ℓ`), one
pointer-batched GEMM for `T_{iℓj} = α S_{iℓj}H_{ℓj}` (written directly into the
layout the fused reduction needs), and one plain GEMM for
`Y = Σ_ℓ U^A_{iℓ}T_{iℓj}` with contraction dimension `ρ_i = r_A q_k`
(eq:Y) -- three launches (five with `β ≠ 0`), independent of `q_k`.
"""
function apply_right!(Y::AbstractMatrix{T}, coupling::TileCoupling,
                      beta_term::Union{TileBetaTerm,Nothing}, Omega::AbstractMatrix{T};
                      compute=nothing) where {T}
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    adj = _adjoint_blas_char(T)
    qk = size(coupling.S, 3)
    rA, rB = size(coupling.S, 1), size(coupling.S, 2)
    bm, s = size(Y)
    bn = size(Omega, 1)
    size(Omega, 2) == s ||
        throw(DimensionMismatch("Omega and Y must have the same sketch width"))
    size(coupling.rowU, 1) == bm ||
        throw(DimensionMismatch("Y's row count must match the output tile's b_m"))
    size(coupling.colZ, 1) == bn ||
        throw(DimensionMismatch("Omega's row count must match the output tile's b_n"))

    if qk > 0
        H = similar(Y, T, rB, s, qk)
        # Copy Omega into a plain, freshly-allocated 3D buffer rather than
        # reshaping it in place: `reshape` of a view the caller handed in
        # (e.g. a slice of a slice, as `range_find_tile!` produces) does not
        # reliably dispatch to the batched GPU GEMM and can silently fall
        # through to a host BLAS call on device memory. The copy is one
        # bn×s transfer, negligible next to the GEMMs it feeds.
        Omega3 = similar(Y, T, bn, s, 1)
        copyto!(view(Omega3, :, :, 1), Omega)
        precision_gemm_batched!(adj, 'N', one(T), coupling.colZ,
                                Omega3, zero(T), H, mode)

        # T written straight into (rA, qk, s): ℓ is the *middle* index, matching
        # rowpanel(au,i)'s own memory order (rA fastest, then ℓ) so the fused
        # reduction below is a single zero-copy reshape on both operands.
        Tbuf = similar(Y, T, rA, qk, s)
        Sviews = [view(coupling.S, :, :, l) for l in 1:qk]
        Hviews = [view(H, :, :, l) for l in 1:qk]
        Tviews = [view(Tbuf, :, l, :) for l in 1:qk]
        precision_gemm_batched!('N', 'N', coupling.alpha, Sviews, Hviews, zero(T),
                                Tviews, mode)

        Ustack = reshape(coupling.rowU, bm, rA * qk)
        Tstack = reshape(Tbuf, rA * qk, s)
        precision_gemm!('N', 'N', one(T), Ustack, Tstack, zero(T), Y, mode)
    else
        fill!(Y, zero(T))
    end

    if beta_term !== nothing
        rC = size(beta_term.Vc, 2)
        tmp = similar(Y, T, rC, s)
        precision_gemm!(adj, 'N', one(T), beta_term.Vc, Omega, zero(T), tmp, mode)
        precision_gemm!('N', 'N', beta_term.beta, beta_term.Uc, tmp, one(T), Y, mode)
    end
    return Y
end

"""
    apply_left!(Z, coupling, beta_term, Q; compute) -> Z

`Z ← X_ijᵀ·Q`, the transpose analogue of [`apply_right!`](@ref): `Q` is
`b_m × s`, `Z` must be `b_n × s`. `G_{iℓj} = (U^A_{iℓ})ᵀQ` (batched, `Q`
broadcast over `ℓ`), `W_{iℓj} = α S_{iℓj}ᵀG_{iℓj}` (pointer-batched), then the
fused reduction `Z = Σ_ℓ V^B_{ℓj}W_{iℓj}` with contraction `γ_j = r_B q_k`.
"""
function apply_left!(Z::AbstractMatrix{T}, coupling::TileCoupling,
                     beta_term::Union{TileBetaTerm,Nothing}, Q::AbstractMatrix{T};
                     compute=nothing) where {T}
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    adj = _adjoint_blas_char(T)
    qk = size(coupling.S, 3)
    rA, rB = size(coupling.S, 1), size(coupling.S, 2)
    bn, s = size(Z)
    bm = size(Q, 1)
    size(Q, 2) == s ||
        throw(DimensionMismatch("Q and Z must have the same sketch width"))
    size(coupling.rowU, 1) == bm ||
        throw(DimensionMismatch("Q's row count must match the output tile's b_m"))
    size(coupling.colZ, 1) == bn ||
        throw(DimensionMismatch("Z's row count must match the output tile's b_n"))

    if qk > 0
        G = similar(Z, T, rA, s, qk)
        # See apply_right!'s matching comment: copy rather than reshape a
        # caller-supplied view in place, so the batched GPU GEMM dispatch
        # cannot silently fall through to a host BLAS call.
        Q3 = similar(Z, T, bm, s, 1)
        copyto!(view(Q3, :, :, 1), Q)
        precision_gemm_batched!(adj, 'N', one(T), coupling.rowU,
                                Q3, zero(T), G, mode)

        Wbuf = similar(Z, T, rB, qk, s)
        Sviews = [view(coupling.S, :, :, l) for l in 1:qk]
        Gviews = [view(G, :, :, l) for l in 1:qk]
        Wviews = [view(Wbuf, :, l, :) for l in 1:qk]
        precision_gemm_batched!(adj, 'N', coupling.alpha, Sviews, Gviews, zero(T),
                                Wviews, mode)

        Zstack = reshape(coupling.colZ, bn, rB * qk)
        Wstack = reshape(Wbuf, rB * qk, s)
        precision_gemm!('N', 'N', one(T), Zstack, Wstack, zero(T), Z, mode)
    else
        fill!(Z, zero(T))
    end

    if beta_term !== nothing
        rC = size(beta_term.Uc, 2)
        tmp = similar(Z, T, rC, s)
        precision_gemm!(adj, 'N', one(T), beta_term.Uc, Q, zero(T), tmp, mode)
        precision_gemm!('N', 'N', beta_term.beta, beta_term.Vc, tmp, one(T), Z, mode)
    end
    return Z
end

# R3 column-family sampler -----------------------------------------------------

"""
    ColumnRunCoupling(ops, rows, j; alpha, beta=false, C=nothing, block, maxrank)

Retained factor-list state for a run of output tiles `(rows, j)`. The run is
physically ordered by active ARA slot. `S[:,:,:,slot]` and the corresponding A
row panel are swap-compacted when a member retires.

The fixed output column makes `H_ℓj = V^B_ℓj'Ω` common to every active tile:
[`apply_right_run!`](@ref) forms it once per pass, then batches the remaining
coupling and fused-reduction work over the active prefix.
"""
struct ColumnRunCoupling{ST,RU,CZ,BUT,BVT,HT,TT,GT,WT,OT,BT,T}
    S::ST
    rowU::RU
    colZ::CZ
    betaU::BUT
    betaV::BVT
    H::HT
    Tbuf::TT
    G::GT
    Wbuf::WT
    Omega::OT
    beta_tmp::BT
    alpha::T
    qk::Int
end

function ColumnRunCoupling(ops::LogicalTLROperands, rows, j::Integer;
                           alpha, beta=false, C=nothing,
                           block::Int, maxrank::Int, compute=nothing)
    ids = collect(Int, rows)
    count = length(ids)
    rowU = [rowpanel(ops.au, i) for i in ids]
    rowV = [rowpanel(ops.av, i) for i in ids]
    colW = colpanel(ops.bw, j)
    colZ = colpanel(ops.bz, j)
    T = eltype(colZ)
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    qk = size(colZ, 3)
    rA = size(first(rowU), 2)
    rB = size(colZ, 2)
    bm = size(first(rowU), 1)
    bn = size(colZ, 1)
    backend = get_backend(colZ)

    S = allocate(backend, T, rA, rB, qk, count)
    if count > 0 && qk > 0
        Avec = [view(rowV[p], :, :, l) for p in 1:count for l in 1:qk]
        Bvec = [view(colW, :, :, l) for p in 1:count for l in 1:qk]
        Svec = [view(S, :, :, l, p) for p in 1:count for l in 1:qk]
        precision_gemm_batched!(_adjoint_blas_char(T), 'N', one(T),
                                Avec, Bvec, zero(T), Svec, mode)
    end

    betaU, betaV = if iszero(beta)
        (nothing, nothing)
    else
        C === nothing && throw(ArgumentError("C must be supplied when beta != 0"))
        uv = [logical_tile_factors(C, i, Int(j)) for i in ids]
        ([x[1] for x in uv], [x[2] for x in uv])
    end

    return ColumnRunCoupling(
        S, rowU, colZ, betaU, betaV,
        allocate(backend, T, rB, block, qk),
        allocate(backend, T, rA, qk, block, count),
        allocate(backend, T, rA, maxrank, qk, count),
        allocate(backend, T, rB, qk, maxrank, count),
        allocate(backend, T, bn, block, 1),
        allocate(backend, T,
                 betaV === nothing ? 0 : size(first(betaV), 2),
                 max(block, maxrank), count),
        T(alpha), qk,
    )
end

function _swap_column_run_members!(run::ColumnRunCoupling, p::Int, q::Int)
    p == q && return run
    backend = get_backend(run.S)
    S3 = reshape(run.S, :, 1, size(run.S, 4))
    _ara_swap_basis_kernel!(backend)(S3, p, q;
                                     ndrange=(size(S3, 1), 1))
    run.rowU[p], run.rowU[q] = run.rowU[q], run.rowU[p]
    if run.betaU !== nothing
        run.betaU[p], run.betaU[q] = run.betaU[q], run.betaU[p]
        run.betaV[p], run.betaV[q] = run.betaV[q], run.betaV[p]
    end
    return run
end

"""
    apply_right_run!(Y, run, width, nactive; beta, compute)

Sample the contiguous active prefix of a fixed-column run. Exactly one `H`
batch is formed for the column, independent of `nactive`; `T` and the final
reduction are batched over active members.
"""
function apply_right_run!(Y::AbstractArray{T,3}, run::ColumnRunCoupling,
                          width::Int, nactive::Int;
                          beta=false, compute=nothing) where {T}
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    adj = _adjoint_blas_char(T)
    qk = run.qk
    bm = size(Y, 1)
    rA, rB = size(run.S, 1), size(run.S, 2)

    Om = view(run.Omega, :, 1:width, :)
    Random.randn!(Om)
    H = view(run.H, :, 1:width, :)
    if qk > 0
        precision_gemm_batched!(adj, 'N', one(T), run.colZ, Om,
                                zero(T), H, mode)
        Svec = [view(run.S, :, :, l, p) for p in 1:nactive for l in 1:qk]
        Hvec = [view(H, :, :, l) for p in 1:nactive for l in 1:qk]
        Tvec = [view(run.Tbuf, :, l, 1:width, p)
                for p in 1:nactive for l in 1:qk]
        precision_gemm_batched!('N', 'N', run.alpha, Svec, Hvec,
                                zero(T), Tvec, mode)
        Uvec = [reshape(run.rowU[p], bm, rA * qk) for p in 1:nactive]
        Tstack = [reshape(view(run.Tbuf, :, :, 1:width, p), rA * qk, width)
                  for p in 1:nactive]
        Yvec = [view(Y, :, 1:width, p) for p in 1:nactive]
        precision_gemm_batched!('N', 'N', one(T), Uvec, Tstack,
                                zero(T), Yvec, mode)
    else
        fill!(view(Y, :, 1:width, :), zero(T))
    end

    if run.betaU !== nothing
        tmp = [view(run.beta_tmp, :, 1:width, p) for p in 1:nactive]
        Ovec = [view(Om, :, :, 1) for _ in 1:nactive]
        Yvec = [view(Y, :, 1:width, p) for p in 1:nactive]
        precision_gemm_batched!(adj, 'N', one(T), view(run.betaV, 1:nactive),
                                Ovec, zero(T), tmp, mode)
        precision_gemm_batched!('N', 'N', T(beta), view(run.betaU, 1:nactive),
                                tmp, one(T), Yvec, mode)
    end
    return Y
end

"""
    apply_left_run!(Z, run, Q, width; beta, compute)

Form `Z_slot = X_tile(slot)'Q_slot` for every member of the finished run in
one grouped apply. Run-local `S` and `Q` are in the same final packed order.
"""
function apply_left_run!(Z::AbstractArray{T,3}, run::ColumnRunCoupling,
                         Q::AbstractArray{T,3}, width::Int;
                         beta=false, compute=nothing) where {T}
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    adj = _adjoint_blas_char(T)
    count = size(Q, 3)
    qk = run.qk
    rA, rB = size(run.S, 1), size(run.S, 2)
    bn = size(Z, 1)

    if qk > 0
        Uvec = [view(run.rowU[p], :, :, l) for p in 1:count for l in 1:qk]
        Qvec = [view(Q, :, 1:width, p) for p in 1:count for l in 1:qk]
        Gvec = [view(run.G, :, 1:width, l, p) for p in 1:count for l in 1:qk]
        precision_gemm_batched!(adj, 'N', one(T), Uvec, Qvec,
                                zero(T), Gvec, mode)
        Svec = [view(run.S, :, :, l, p) for p in 1:count for l in 1:qk]
        Wvec = [view(run.Wbuf, :, l, 1:width, p)
                for p in 1:count for l in 1:qk]
        precision_gemm_batched!(adj, 'N', run.alpha, Svec, Gvec,
                                zero(T), Wvec, mode)
        Zleft = [reshape(run.colZ, bn, rB * qk) for _ in 1:count]
        Wstack = [reshape(view(run.Wbuf, :, :, 1:width, p), rB * qk, width)
                  for p in 1:count]
        Zvec = [view(Z, :, 1:width, p) for p in 1:count]
        precision_gemm_batched!('N', 'N', one(T), Zleft, Wstack,
                                zero(T), Zvec, mode)
    else
        fill!(Z, zero(T))
    end

    if run.betaU !== nothing
        tmp = [view(run.beta_tmp, :, 1:width, p) for p in 1:count]
        Qvec = [view(Q, :, 1:width, p) for p in 1:count]
        Zvec = [view(Z, :, 1:width, p) for p in 1:count]
        precision_gemm_batched!(adj, 'N', one(T), run.betaU, Qvec,
                                zero(T), tmp, mode)
        precision_gemm_batched!('N', 'N', T(beta), run.betaV, tmp,
                                one(T), Zvec, mode)
    end
    return Z
end
