# The factor-list sampler for one output tile (i,j) of X = α·A·B + β·C.
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
# pairs are the general-storage integration's concern (packing or a different
# fusion), not this file's.

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
    colW::CWT    # colpanel(bu, j): bk × rB × qk  (U^B_{·j})
    colZ::CZT    # colpanel(bv, j): bn × rB × qk  (V^B_{·j})
    alpha::T
end

function TileCoupling(ops::LogicalTLROperands, i::Integer, j::Integer;
                      alpha, compute=nothing)
    rowV = rowpanel(ops.av, i)
    rowU = rowpanel(ops.au, i)
    colW = colpanel(ops.bu, j)
    colZ = colpanel(ops.bv, j)
    size(rowV, 3) == size(colW, 3) ||
        throw(DimensionMismatch("A's row panel and B's column panel disagree on q_k"))
    size(rowV, 1) == size(colW, 1) ||
        throw(DimensionMismatch("A's and B's contraction tile size (b_k) disagree"))

    T = eltype(rowV)
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    rA, rB, qk = size(rowV, 2), size(colW, 2), size(rowV, 3)
    S = similar(rowV, T, rA, rB, qk)
    if qk > 0 && rA > 0 && rB > 0
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
`PaddedFTLRMatrix` (a zero-copy tile-factor view); required whenever `beta != 0`.
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

    if qk > 0 && rA > 0 && rB > 0
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
        Sviews = [view(coupling.S, :, :, kidx) for kidx in 1:qk]
        Hviews = [view(H, :, :, kidx) for kidx in 1:qk]
        Tviews = [view(Tbuf, :, kidx, :) for kidx in 1:qk]
        precision_gemm_batched!('N', 'N', coupling.alpha, Sviews, Hviews, zero(T),
                                Tviews, mode)

        Ustack = reshape(coupling.rowU, bm, rA * qk)
        coupling_sketch_stacks = reshape(Tbuf, rA * qk, s)
        precision_gemm!('N', 'N', one(T), Ustack, coupling_sketch_stacks, zero(T), Y, mode)
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

    if qk > 0 && rA > 0 && rB > 0
        G = similar(Z, T, rA, s, qk)
        # See apply_right!'s matching comment: copy rather than reshape a
        # caller-supplied view in place, so the batched GPU GEMM dispatch
        # cannot silently fall through to a host BLAS call.
        Q3 = similar(Z, T, bm, s, 1)
        copyto!(view(Q3, :, :, 1), Q)
        precision_gemm_batched!(adj, 'N', one(T), coupling.rowU,
                                Q3, zero(T), G, mode)

        Wbuf = similar(Z, T, rB, qk, s)
        Sviews = [view(coupling.S, :, :, kidx) for kidx in 1:qk]
        Gviews = [view(G, :, :, kidx) for kidx in 1:qk]
        Wviews = [view(Wbuf, :, kidx, :) for kidx in 1:qk]
        precision_gemm_batched!(adj, 'N', coupling.alpha, Sviews, Gviews, zero(T),
                                Wviews, mode)

        Zstack = reshape(coupling.colZ, bn, rB * qk)
        coupling_sketch_stacks = reshape(Wbuf, rB * qk, s)
        precision_gemm!('N', 'N', one(T), Zstack, coupling_sketch_stacks, zero(T), Z, mode)
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
