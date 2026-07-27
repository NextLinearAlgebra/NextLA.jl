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

## Pointer-batch descriptors (hot path only)

`rowU` is not representable as a strided-3D batch across members (its
entries are views at arbitrary, non-uniformly-spaced offsets into shared
operand storage), so a batched GEMM over it needs a pointer-batched call. On
backends with pointer-batched GEMM, the hot per-pass sampler
([`apply_right_run!`](@ref)'s member-only-batched final reduction and beta
accumulation) is made allocation-free by building the needed pointer tables
once here rather than rebuilding them every pass: `rowU_ptrs`,
`betaU_ptrs`, `betaV_ptrs` are swap-tracked ([`swap_batch_ptrs!`](@ref) in
[`_swap_column_run_members!`](@ref) keeps them in step with active-prefix
packing); `Tstack_ptrs`, `Om_ptrs`, and `beta_tmp_ptrs` address this run's
own scratch (`Tbuf`, `Omega`, `beta_tmp`), which is never swapped, only
reused, so they are built once and never touched again. CUBLAS/rocBLAS's
pointer-batched GEMM has no strided/pointer mixed form, so once `rowU` (or
`betaU`/`betaV`) forces a call into pointer-batched form, every operand in
that same call must be — including the otherwise-strided `Tstack`/`Om`/
`beta_tmp`, which is why they get descriptors too even though a
strided-only view of them would otherwise suffice.

`Y` (the ARA basis being grown) is the one operand this cannot cover: it is
owned by the caller's `ARAWorkspace`, not this run, so its address is only
known at the first `apply_right_run!` call and cannot be cached in this
struct without threading a descriptor through `numerics/ara.jl` — judged out
of scope for this pass (see `docs/TODO.md`'s workspace contract). Its
pointer array is therefore built once per `apply_right_run!` call (not once
per GEMM within it, since the fused reduction and the beta accumulation
share the same `Y`) and is the sampler's only remaining per-pass allocation.

The co-range apply ([`apply_left_run!`](@ref), called once after
convergence, not once per pass) is intentionally left entirely on the
`Vector`-of-views path and does not use any of these descriptors: its cost
does not compound with pass count the way the sampler's did, and its G/H
formation batches `rowU` over member × contraction-tile (`nmember*qk`
pointers, needing a `qk`-blocked swap) rather than member-only, which would
need a second, larger descriptor. See `docs/TODO.md` for this scoping
decision.
"""
struct ColumnRunCoupling{ST,RU,CZ,BUT,BVT,HT,TT,GT,WT,OT,BT,T}
    S::ST
    rowU::RU
    colZ::CZ
    betaU::BUT
    betaV::BVT
    rowU_ptrs::Union{Nothing,BatchPtrDescriptor}
    Tstack_ptrs::Union{Nothing,BatchPtrDescriptor}
    betaU_ptrs::Union{Nothing,BatchPtrDescriptor}
    betaV_ptrs::Union{Nothing,BatchPtrDescriptor}
    Om_ptrs::Union{Nothing,BatchPtrDescriptor}
    beta_tmp_ptrs::Union{Nothing,BatchPtrDescriptor}
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
                           block::Int, maxrank::Int,
                           rA::Int=rankdim(ops.au),
                           rB::Int=rankdim(ops.bv),
                           compute=nothing)
    ids = collect(Int, rows)
    nmember = length(ids)
    rowU = [_factor_row_stack(ops.au, i, rA) for i in ids]
    rowV = [_factor_row_stack(ops.av, i, rA) for i in ids]
    colW = _factor_column_stack(ops.bu, Int(j), rB)
    colZ = _factor_column_stack(ops.bv, Int(j), rB)
    T = eltype(colZ)
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    qk = size(colZ, 3)
    rA = rA
    rB = rB
    bm = size(first(rowU), 1)
    bn = size(colZ, 1)
    backend = get_backend(colZ)

    S = allocate(backend, T, rA, rB, qk, nmember)
    if nmember > 0 && qk > 0 && rA > 0 && rB > 0
        Avec = [view(rowV[p], :, :, kidx) for p in 1:nmember for kidx in 1:qk]
        Bvec = [view(colW, :, :, kidx) for p in 1:nmember for kidx in 1:qk]
        Svec = [view(S, :, :, kidx, p) for p in 1:nmember for kidx in 1:qk]
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

    Hbuf = allocate(backend, T, rB, block, qk)
    Tbuf = allocate(backend, T, rA, qk, block, nmember)
    Gbuf = allocate(backend, T, rA, maxrank, qk, nmember)
    Wbuf = allocate(backend, T, rB, qk, maxrank, nmember)
    Omega = allocate(backend, T, bn, block, 1)
    beta_tmp = allocate(backend, T,
                        betaV === nothing ? 0 : size(first(betaV), 2),
                        max(block, maxrank), nmember)

    ptrs_ok = nmember > 0 && supports_pointer_batched(backend)
    rowU_ptrs = ptrs_ok ?
        BatchPtrDescriptor([reshape(rowU[p], bm, rA * qk) for p in 1:nmember]) :
        nothing
    Tstack_ptrs = ptrs_ok ?
        BatchPtrDescriptor([view(reshape(Tbuf, rA * qk, block, nmember), :, :, p)
                           for p in 1:nmember]) :
        nothing
    has_beta_ptrs = ptrs_ok && betaU !== nothing
    betaU_ptrs = has_beta_ptrs ? BatchPtrDescriptor(betaU) : nothing
    betaV_ptrs = has_beta_ptrs ? BatchPtrDescriptor(betaV) : nothing
    Om_ptrs = has_beta_ptrs ?
        BatchPtrDescriptor([view(Omega, :, :, 1) for _ in 1:nmember]) : nothing
    beta_tmp_ptrs = has_beta_ptrs ?
        BatchPtrDescriptor([view(beta_tmp, :, :, p) for p in 1:nmember]) : nothing

    return ColumnRunCoupling(
        S, rowU, colZ, betaU, betaV,
        rowU_ptrs, Tstack_ptrs, betaU_ptrs, betaV_ptrs, Om_ptrs, beta_tmp_ptrs,
        Hbuf, Tbuf, Gbuf, Wbuf, Omega, beta_tmp,
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
    run.rowU_ptrs !== nothing && swap_batch_ptrs!(run.rowU_ptrs, p, q)
    if run.betaU !== nothing
        run.betaU[p], run.betaU[q] = run.betaU[q], run.betaU[p]
        run.betaV[p], run.betaV[q] = run.betaV[q], run.betaV[p]
        run.betaU_ptrs !== nothing && swap_batch_ptrs!(run.betaU_ptrs, p, q)
        run.betaV_ptrs !== nothing && swap_batch_ptrs!(run.betaV_ptrs, p, q)
    end
    return run
end

"""
    apply_right_run!(Y, run, sketch_width, nactive; beta, compute)

Sample the contiguous active prefix of a fixed-column run. Exactly one `H`
batch is formed for the column, independent of `nactive`; `T` and the final
reduction are batched over active members.

This is the run's hot per-pass sampler. When `run.rowU_ptrs !== nothing`
(pointer-batched GEMM available), the final reduction and beta accumulation
run entirely off the persistent descriptors built in
[`ColumnRunCoupling`](@ref) -- no `Vector`-of-views or device pointer array
is built for `rowU`/`Tstack`/beta terms. `Y`'s own pointer array is the one
exception (see the struct docstring); it is built once here and reused for
both GEMMs in this call. On backends without pointer-batched GEMM (CPU),
this falls back to the original `Vector`-of-views construction.
"""
function apply_right_run!(Y::AbstractArray{T,3}, run::ColumnRunCoupling,
                          sketch_width::Int, nactive::Int;
                          beta=false, compute=nothing) where {T}
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    adj = _adjoint_blas_char(T)
    qk = run.qk
    bm = size(Y, 1)
    rA, rB = size(run.S, 1), size(run.S, 2)
    use_ptrs = run.rowU_ptrs !== nothing

    Om = view(run.Omega, :, 1:sketch_width, :)
    Random.randn!(Om)
    H = view(run.H, :, 1:sketch_width, :)

    Yptrs = use_ptrs ?
        BatchPtrDescriptor([view(Y, :, 1:sketch_width, p) for p in 1:nactive]) :
        nothing

    if qk > 0 && rA > 0 && rB > 0
        precision_gemm_batched!(adj, 'N', one(T), run.colZ, Om,
                                zero(T), H, mode)
        Svec = [view(run.S, :, :, kidx, p) for p in 1:nactive for kidx in 1:qk]
        Hvec = [view(H, :, :, kidx) for p in 1:nactive for kidx in 1:qk]
        Tvec = [view(run.Tbuf, :, kidx, 1:sketch_width, p)
                for p in 1:nactive for kidx in 1:qk]
        precision_gemm_batched!('N', 'N', run.alpha, Svec, Hvec,
                                zero(T), Tvec, mode)
        if use_ptrs
            Uref = reshape(run.rowU[1], bm, rA * qk)
            Tref = view(reshape(run.Tbuf, rA * qk, size(run.Tbuf, 3),
                               size(run.Tbuf, 4)), :, 1:sketch_width, 1)
            precision_gemm_batched_ptrs!('N', 'N', one(T),
                run.rowU_ptrs, Uref, run.Tstack_ptrs, Tref,
                zero(T), Yptrs, view(Y, :, 1:sketch_width, 1), nactive, mode)
        else
            Uvec = [reshape(run.rowU[p], bm, rA * qk) for p in 1:nactive]
            Tstack = [reshape(view(run.Tbuf, :, :, 1:sketch_width, p), rA * qk, sketch_width)
                      for p in 1:nactive]
            Yvec = [view(Y, :, 1:sketch_width, p) for p in 1:nactive]
            precision_gemm_batched!('N', 'N', one(T), Uvec, Tstack,
                                    zero(T), Yvec, mode)
        end
    else
        fill!(view(Y, :, 1:sketch_width, :), zero(T))
    end

    if run.betaU !== nothing
        if use_ptrs
            tmpref = view(run.beta_tmp, :, 1:sketch_width, 1)
            precision_gemm_batched_ptrs!(adj, 'N', one(T),
                run.betaV_ptrs, first(run.betaV), run.Om_ptrs, view(Om, :, :, 1),
                zero(T), run.beta_tmp_ptrs, tmpref, nactive, mode)
            precision_gemm_batched_ptrs!('N', 'N', T(beta),
                run.betaU_ptrs, first(run.betaU), run.beta_tmp_ptrs, tmpref,
                one(T), Yptrs, view(Y, :, 1:sketch_width, 1), nactive, mode)
        else
            tmp = [view(run.beta_tmp, :, 1:sketch_width, p) for p in 1:nactive]
            Ovec = [view(Om, :, :, 1) for _ in 1:nactive]
            Yvec = [view(Y, :, 1:sketch_width, p) for p in 1:nactive]
            precision_gemm_batched!(adj, 'N', one(T), view(run.betaV, 1:nactive),
                                    Ovec, zero(T), tmp, mode)
            precision_gemm_batched!('N', 'N', T(beta), view(run.betaU, 1:nactive),
                                    tmp, one(T), Yvec, mode)
        end
    end
    return Y
end

"""
    apply_left_run!(Z, run, Q, sketch_width; beta, compute)

Form `Z_slot = X_tile(slot)'Q_slot` for every member of the finished run in
one grouped apply. Run-local `S` and `Q` are in the same final packed order.
"""
function apply_left_run!(Z::AbstractArray{T,3}, run::ColumnRunCoupling,
                         Q::AbstractArray{T,3}, sketch_width::Int;
                         beta=false, compute=nothing) where {T}
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    adj = _adjoint_blas_char(T)
    nmember = size(Q, 3)
    qk = run.qk
    rA, rB = size(run.S, 1), size(run.S, 2)
    bn = size(Z, 1)

    if qk > 0 && rA > 0 && rB > 0
        Uvec = [view(run.rowU[p], :, :, kidx) for p in 1:nmember for kidx in 1:qk]
        Qvec = [view(Q, :, 1:sketch_width, p) for p in 1:nmember for kidx in 1:qk]
        Gvec = [view(run.G, :, 1:sketch_width, kidx, p) for p in 1:nmember for kidx in 1:qk]
        precision_gemm_batched!(adj, 'N', one(T), Uvec, Qvec,
                                zero(T), Gvec, mode)
        Svec = [view(run.S, :, :, kidx, p) for p in 1:nmember for kidx in 1:qk]
        Wvec = [view(run.Wbuf, :, kidx, 1:sketch_width, p)
                for p in 1:nmember for kidx in 1:qk]
        precision_gemm_batched!(adj, 'N', run.alpha, Svec, Gvec,
                                zero(T), Wvec, mode)
        Zleft = [reshape(run.colZ, bn, rB * qk) for _ in 1:nmember]
        Wstack = [reshape(view(run.Wbuf, :, :, 1:sketch_width, p), rB * qk, sketch_width)
                  for p in 1:nmember]
        Zvec = [view(Z, :, 1:sketch_width, p) for p in 1:nmember]
        precision_gemm_batched!('N', 'N', one(T), Zleft, Wstack,
                                zero(T), Zvec, mode)
    else
        fill!(Z, zero(T))
    end

    if run.betaU !== nothing
        tmp = [view(run.beta_tmp, :, 1:sketch_width, p) for p in 1:nmember]
        Qvec = [view(Q, :, 1:sketch_width, p) for p in 1:nmember]
        Zvec = [view(Z, :, 1:sketch_width, p) for p in 1:nmember]
        precision_gemm_batched!(adj, 'N', one(T), run.betaU, Qvec,
                                zero(T), tmp, mode)
        precision_gemm_batched!('N', 'N', T(beta), run.betaV, tmp,
                                one(T), Zvec, mode)
    end
    return Z
end

# R3 canonical fixed-row samplers ---------------------------------------------

@kernel function _pack_factor_row_kernel!(dest, src, row::Int, qm::Int, qn::Int,
                                          row_major::Bool)
    i, k, kidx = @index(Global, NTuple)
    tile = row_major ? kidx + (row - 1) * qn : row + (kidx - 1) * qm
    @inbounds dest[i, k, kidx] = src[i, k, tile]
end

@kernel function _pack_factor_columns_kernel!(dest, src, cols, qm::Int, qn::Int,
                                              row_major::Bool)
    i, k, kidx, p = @index(Global, NTuple)
    col = Int(@inbounds cols[p])
    tile = row_major ? col + (kidx - 1) * qn : kidx + (col - 1) * qm
    @inbounds dest[i, k, kidx, p] = src[i, k, tile]
end

@kernel function _pack_factor_column_kernel!(dest, src, col::Int, qm::Int, qn::Int,
                                             row_major::Bool)
    i, k, kidx = @index(Global, NTuple)
    tile = row_major ? col + (kidx - 1) * qn : kidx + (col - 1) * qm
    @inbounds dest[i, k, kidx] = src[i, k, tile]
end

@inline _is_row_major(order) = order isa TileRowMajor

function _factor_row_stack(p::InteriorOperand, row::Int, rank::Int)
    if p.order isa TileRowMajor && rank == rankdim(p)
        return rowpanel(p, row)
    end
    backend = get_backend(p.data)
    dest = allocate(backend, eltype(p.data), size(p.data, 1), rank, p.qn)
    rank == 0 && return dest
    _pack_factor_row_kernel!(backend)(
        dest, p.data, row, p.qm, p.qn, _is_row_major(p.order);
        ndrange=size(dest),
    )
    return dest
end

function _factor_column_stack(p::InteriorOperand, col::Int, rank::Int)
    if p.order isa TileColMajor && rank == rankdim(p)
        return colpanel(p, col)
    end
    backend = get_backend(p.data)
    dest = allocate(backend, eltype(p.data), size(p.data, 1), rank, p.qm)
    rank == 0 && return dest
    _pack_factor_column_kernel!(backend)(
        dest, p.data, col, p.qm, p.qn, _is_row_major(p.order);
        ndrange=size(dest),
    )
    return dest
end

function _factor_column_stacks(p::InteriorOperand, cols::Vector{Int}, rank::Int)
    backend = get_backend(p.data)
    nmember = length(cols)
    dest = allocate(backend, eltype(p.data), size(p.data, 1), rank, p.qm, nmember)
    (rank == 0 || nmember == 0) && return dest
    cols_dev = copyto!(allocate(backend, Int32, nmember), Int32.(cols))
    _pack_factor_columns_kernel!(backend)(
        dest, p.data, cols_dev, p.qm, p.qn, _is_row_major(p.order);
        ndrange=size(dest),
    )
    return dest
end

@inline function _trimmed_tile(p::InteriorOperand, i::Int, j::Int, rank::Int)
    view(tilefactor(p, i, j), :, 1:rank)
end

"""
Fixed-row, right-sampling state for logical `A` tile-row-major. The repeated
`XΩ` apply terminates in A's zero-copy row stack. B's column stacks are
materialized once only for the final `XᵀQ` co-range.

## Pointer-batch descriptors (hot path only)

Unlike [`ColumnRunCoupling`](@ref)'s `rowU`, `Bouter` is genuinely on the hot
per-pass sampler's path (`apply_right_row_run!`'s H-formation batches it over
member × contraction-tile, `nmember*qk` entries, since each member's tiles sit
at arbitrary offsets in `B`'s storage). On backends with pointer-batched
GEMM, `Bouter_ptrs` is a `nmember*qk`-length descriptor, member-major
(`(p-1)*qk+kidx`), swap-tracked with the *block* form of
[`swap_batch_ptrs!`](@ref) (`blocklen=qk`) since retiring a member moves its
whole qk-tile block. `HOm_ptrs` and `H_ptrs` cover this same call's other two
operands: `Om` (this run's own `Omega`, repeated per contraction tile) and
`H` (this run's own scratch, reshaped so its natural `(kidx,p)` memory order
already matches `Bouter_ptrs`'s) -- both build-once, never swapped, since
CUBLAS/rocBLAS's pointer-batched GEMM has no strided/pointer mixed form and
every operand in a call with `Bouter_ptrs` must be pointer-batched too, even
though `Om`/`H` would individually be fine as strided views. `betaU_ptrs`/
`betaV_ptrs` (single-slot swap) and `betaOm_ptrs`/`beta_tmp_ptrs`
(build-once, one entry per member, no `qk` repeat) cover the beta
accumulation the same way. `Y`'s own pointer array remains the one residual
per-pass cost (see [`ColumnRunCoupling`](@ref)'s docstring for why).

The co-range apply (`apply_left_row_run!`, once per run, not once per pass)
is intentionally left on the `Vector`-of-views path; see `docs/TODO.md`.
"""
struct RowRightRunCoupling{ST,AStackT,BOT,BStackT,BUT,BVT,HT,TT,GT,WT,OT,BT,T}
    S::ST
    Astack::AStackT
    Bouter::BOT
    Bstack::BStackT
    betaU::BUT
    betaV::BVT
    Bouter_ptrs::Union{Nothing,BatchPtrDescriptor}
    HOm_ptrs::Union{Nothing,BatchPtrDescriptor}
    H_ptrs::Union{Nothing,BatchPtrDescriptor}
    betaU_ptrs::Union{Nothing,BatchPtrDescriptor}
    betaV_ptrs::Union{Nothing,BatchPtrDescriptor}
    betaOm_ptrs::Union{Nothing,BatchPtrDescriptor}
    beta_tmp_ptrs::Union{Nothing,BatchPtrDescriptor}
    H::HT
    Tbuf::TT
    G::GT
    Wbuf::WT
    Omega::OT
    beta_tmp::BT
    alpha::T
    qk::Int
end

function RowRightRunCoupling(ops::LogicalTLROperands, i::Integer, cols;
                             alpha, beta=false, C=nothing,
                             block::Int, maxrank::Int,
                             rA::Int=rankdim(ops.au),
                             rB::Int=rankdim(ops.bv),
                             compute=nothing)
    ids = collect(Int, cols)
    nmember = length(ids)
    qk = ops.au.qn
    T = eltype(ops.au.data)
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    backend = get_backend(ops.au.data)
    bm = size(ops.au.data, 1)
    bn = size(ops.bv.data, 1)

    Ainner = [_trimmed_tile(ops.av, Int(i), kidx, rA) for kidx in 1:qk]
    Bouter = [[_trimmed_tile(ops.bu, kidx, j, rB) for kidx in 1:qk] for j in ids]
    Binner = [[_trimmed_tile(ops.bv, kidx, j, rB) for kidx in 1:qk] for j in ids]
    Astack = _factor_row_stack(ops.au, Int(i), rA)
    Bstack = _factor_column_stacks(ops.bv, ids, rB)

    S = allocate(backend, T, rA, rB, qk, nmember)
    if nmember > 0 && qk > 0 && rA > 0 && rB > 0
        Avec = [Ainner[kidx] for p in 1:nmember for kidx in 1:qk]
        Bvec = [Bouter[p][kidx] for p in 1:nmember for kidx in 1:qk]
        Svec = [view(S, :, :, kidx, p) for p in 1:nmember for kidx in 1:qk]
        precision_gemm_batched!(_adjoint_blas_char(T), 'N', one(T),
                                Avec, Bvec, zero(T), Svec, mode)
    end

    betaU, betaV = if iszero(beta)
        (nothing, nothing)
    else
        C === nothing && throw(ArgumentError("C must be supplied when beta != 0"))
        uv = [logical_tile_factors(C, Int(i), j) for j in ids]
        ([x[1] for x in uv], [x[2] for x in uv])
    end
    rC = betaV === nothing ? 0 : size(first(betaV), 2)

    Hbuf = allocate(backend, T, rB, block, qk, nmember)
    Tbuf = allocate(backend, T, rA, qk, block, nmember)
    Gbuf = allocate(backend, T, rA, maxrank, qk, nmember)
    Wbuf = allocate(backend, T, rB, qk, maxrank, nmember)
    Omega = allocate(backend, T, bn, block, nmember)
    beta_tmp = allocate(backend, T, rC, max(block, maxrank), nmember)

    ptrs_ok = nmember > 0 && qk > 0 && supports_pointer_batched(backend)
    # The struct field `Bouter` is populated from `Binner` below (matching
    # apply_right_row_run!'s `run.Bouter[p][kidx]` access, which is the V-factor
    # tile, not the local `Bouter`/U-factor variable used only for S above).
    Bouter_ptrs = ptrs_ok ?
        BatchPtrDescriptor([Binner[p][kidx] for p in 1:nmember for kidx in 1:qk]) :
        nothing
    HOm_ptrs = ptrs_ok ?
        BatchPtrDescriptor([view(Omega, :, :, p) for p in 1:nmember for kidx in 1:qk]) :
        nothing
    H_ptrs = ptrs_ok ?
        BatchPtrDescriptor([view(reshape(Hbuf, rB, block, qk * nmember), :, :, k)
                           for k in 1:(qk * nmember)]) :
        nothing
    has_beta_ptrs = ptrs_ok && betaU !== nothing
    betaU_ptrs = has_beta_ptrs ? BatchPtrDescriptor(betaU) : nothing
    betaV_ptrs = has_beta_ptrs ? BatchPtrDescriptor(betaV) : nothing
    betaOm_ptrs = has_beta_ptrs ?
        BatchPtrDescriptor([view(Omega, :, :, p) for p in 1:nmember]) : nothing
    beta_tmp_ptrs = has_beta_ptrs ?
        BatchPtrDescriptor([view(beta_tmp, :, :, p) for p in 1:nmember]) : nothing

    return RowRightRunCoupling(
        S, Astack, Binner, Bstack, betaU, betaV,
        Bouter_ptrs, HOm_ptrs, H_ptrs, betaU_ptrs, betaV_ptrs, betaOm_ptrs,
        beta_tmp_ptrs,
        Hbuf, Tbuf, Gbuf, Wbuf, Omega, beta_tmp,
        T(alpha), qk,
    )
end

function _swap_row_right_members!(run::RowRightRunCoupling, p::Int, q::Int)
    p == q && return run
    backend = get_backend(run.S)
    S3 = reshape(run.S, :, 1, size(run.S, 4))
    _ara_swap_basis_kernel!(backend)(S3, p, q; ndrange=(size(S3, 1), 1))
    B3 = reshape(run.Bstack, :, 1, size(run.Bstack, 4))
    _ara_swap_basis_kernel!(backend)(B3, p, q; ndrange=(size(B3, 1), 1))
    run.Bouter[p], run.Bouter[q] = run.Bouter[q], run.Bouter[p]
    if run.Bouter_ptrs !== nothing
        swap_batch_ptrs!(run.Bouter_ptrs, p, q, run.qk)
    end
    if run.betaU !== nothing
        run.betaU[p], run.betaU[q] = run.betaU[q], run.betaU[p]
        run.betaV[p], run.betaV[q] = run.betaV[q], run.betaV[p]
        run.betaU_ptrs !== nothing && swap_batch_ptrs!(run.betaU_ptrs, p, q)
        run.betaV_ptrs !== nothing && swap_batch_ptrs!(run.betaV_ptrs, p, q)
    end
    return run
end

"""
    apply_right_row_run!(Y, run, sketch_width, nactive; beta, compute)

The run's hot per-pass sampler. When `run.Bouter_ptrs !== nothing`, the
H-formation and beta accumulation run entirely off the persistent
descriptors built in [`RowRightRunCoupling`](@ref) -- no `Vector`-of-views or
device pointer array is rebuilt for `Bouter`/`Om`/`H`/beta terms.

The T-formation call (`S`/`H`/`Tbuf`, all run-owned) is left on the
`Vector`-of-views path even when `use_ptrs`: unlike `Bouter`, none of its
operands are ragged, so it *could* be made descriptor-driven the same way,
but is left as a documented residual to bound this pass's scope -- it costs
one `Vector`+pointer-array build per pass, not one per tile, so it does not
scale with `q_k` the way the eliminated `Bouter` call did. The final
reduction never involves `Bouter` at all (`Astack`/`Tstack` are broadcast/
reshape-friendly), so it was already fully strided from an earlier pass and
needs no branching here.

`Y`'s own pointer array is the one exception the descriptor mechanism can't
remove for the beta accumulation (see [`ColumnRunCoupling`](@ref)'s
docstring for why); it is built once here and reused for both beta GEMMs.
On backends without pointer-batched GEMM (CPU), this falls back entirely to
the original `Vector`-of-views construction.
"""
function apply_right_row_run!(Y::AbstractArray{T,3}, run::RowRightRunCoupling,
                              sketch_width::Int, nactive::Int;
                              beta=false, compute=nothing) where {T}
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    adj = _adjoint_blas_char(T)
    qk = run.qk
    rA, rB = size(run.S, 1), size(run.S, 2)
    bm = size(Y, 1)
    Om = view(run.Omega, :, 1:sketch_width, 1:nactive)
    Random.randn!(Om)
    use_ptrs = run.Bouter_ptrs !== nothing

    if qk > 0 && rA > 0 && rB > 0
        H = view(run.H, :, 1:sketch_width, :, 1:nactive)
        if use_ptrs
            Bref = first(first(run.Bouter))
            precision_gemm_batched_ptrs!(adj, 'N', one(T),
                run.Bouter_ptrs, Bref, run.HOm_ptrs, view(Om, :, :, 1),
                zero(T), run.H_ptrs, view(H, :, :, 1, 1), nactive * qk, mode)
        else
            Bvec = [run.Bouter[p][kidx] for p in 1:nactive for kidx in 1:qk]
            Ovec = [view(Om, :, :, p) for p in 1:nactive for kidx in 1:qk]
            Hvec_out = [view(H, :, :, kidx, p) for p in 1:nactive for kidx in 1:qk]
            precision_gemm_batched!(adj, 'N', one(T), Bvec, Ovec,
                                    zero(T), Hvec_out, mode)
        end
        Svec = [view(run.S, :, :, kidx, p) for p in 1:nactive for kidx in 1:qk]
        Hvec = [view(H, :, :, kidx, p) for p in 1:nactive for kidx in 1:qk]
        Tvec = [view(run.Tbuf, :, kidx, 1:sketch_width, p)
                for p in 1:nactive for kidx in 1:qk]
        precision_gemm_batched!('N', 'N', run.alpha, Svec, Hvec,
                                zero(T), Tvec, mode)
        # Fused reduction Y = Astack * Tstack as one strided-batched GEMM:
        # Astack (bm, rA*qk) is the same matrix for every member (fixed output
        # row), so it broadcasts as a batch-nmember-1 strided operand; Tstack is
        # a zero-copy reshape of run.Tbuf's leading (rA,qk) dims, batched over
        # the active member axis. No pointer-batch Vector is built here.
        Astack3 = reshape(run.Astack, bm, rA * qk, 1)
        Tbuf2 = reshape(run.Tbuf, rA * qk, size(run.Tbuf, 3), size(run.Tbuf, 4))
        Tstack = view(Tbuf2, :, 1:sketch_width, 1:nactive)
        Yview = view(Y, :, 1:sketch_width, 1:nactive)
        precision_gemm_batched!('N', 'N', one(T), Astack3, Tstack,
                                zero(T), Yview, mode)
    else
        fill!(view(Y, :, 1:sketch_width, 1:nactive), zero(T))
    end

    Yptrs = (use_ptrs && run.betaU !== nothing) ?
        BatchPtrDescriptor([view(Y, :, 1:sketch_width, p) for p in 1:nactive]) :
        nothing

    if run.betaU !== nothing
        if use_ptrs
            tmpref = view(run.beta_tmp, :, 1:sketch_width, 1)
            precision_gemm_batched_ptrs!(adj, 'N', one(T),
                run.betaV_ptrs, first(run.betaV), run.betaOm_ptrs, view(Om, :, :, 1),
                zero(T), run.beta_tmp_ptrs, tmpref, nactive, mode)
            precision_gemm_batched_ptrs!('N', 'N', T(beta),
                run.betaU_ptrs, first(run.betaU), run.beta_tmp_ptrs, tmpref,
                one(T), Yptrs, view(Y, :, 1:sketch_width, 1), nactive, mode)
        else
            tmp = [view(run.beta_tmp, :, 1:sketch_width, p) for p in 1:nactive]
            Ovec = [view(Om, :, :, p) for p in 1:nactive]
            Yvec = [view(Y, :, 1:sketch_width, p) for p in 1:nactive]
            precision_gemm_batched!(adj, 'N', one(T), view(run.betaV, 1:nactive),
                                    Ovec, zero(T), tmp, mode)
            precision_gemm_batched!('N', 'N', T(beta), view(run.betaU, 1:nactive),
                                    tmp, one(T), Yvec, mode)
        end
    end
    return Y
end

function apply_left_row_run!(Z::AbstractArray{T,3}, run::RowRightRunCoupling,
                             Q::AbstractArray{T,3}, sketch_width::Int;
                             beta=false, compute=nothing) where {T}
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    adj = _adjoint_blas_char(T)
    nmember = size(Q, 3)
    qk = run.qk
    rA, rB = size(run.S, 1), size(run.S, 2)
    bn = size(Z, 1)
    if qk > 0 && rA > 0 && rB > 0
        # run.Astack[:,:,kidx] holds the same data run.Aouter[kidx] used to; A has
        # no per-member axis here, so this stays a Vector rebuild (co-range,
        # once per run, not the per-pass hot path) but no longer needs a
        # separately-retained duplicate of Astack's data.
        Uvec = [view(run.Astack, :, :, kidx) for _ in 1:nmember for kidx in 1:qk]
        Qvec = [view(Q, :, 1:sketch_width, p) for p in 1:nmember for kidx in 1:qk]
        Gvec = [view(run.G, :, 1:sketch_width, kidx, p)
                for p in 1:nmember for kidx in 1:qk]
        precision_gemm_batched!(adj, 'N', one(T), Uvec, Qvec,
                                zero(T), Gvec, mode)
        Svec = [view(run.S, :, :, kidx, p) for p in 1:nmember for kidx in 1:qk]
        Wvec = [view(run.Wbuf, :, kidx, 1:sketch_width, p)
                for p in 1:nmember for kidx in 1:qk]
        precision_gemm_batched!(adj, 'N', run.alpha, Svec, Gvec,
                                zero(T), Wvec, mode)
        Bvec = [reshape(view(run.Bstack, :, :, :, p), bn, rB * qk)
                for p in 1:nmember]
        Wstack = [reshape(view(run.Wbuf, :, :, 1:sketch_width, p),
                          rB * qk, sketch_width) for p in 1:nmember]
        Zvec = [view(Z, :, 1:sketch_width, p) for p in 1:nmember]
        precision_gemm_batched!('N', 'N', one(T), Bvec, Wstack,
                                zero(T), Zvec, mode)
    else
        fill!(Z, zero(T))
    end
    if run.betaU !== nothing
        tmp = [view(run.beta_tmp, :, 1:sketch_width, p) for p in 1:nmember]
        Qvec = [view(Q, :, 1:sketch_width, p) for p in 1:nmember]
        Zvec = [view(Z, :, 1:sketch_width, p) for p in 1:nmember]
        precision_gemm_batched!(adj, 'N', one(T), run.betaU, Qvec,
                                zero(T), tmp, mode)
        precision_gemm_batched!('N', 'N', T(beta), run.betaV, tmp,
                                one(T), Zvec, mode)
    end
    return Z
end

"""
Fixed-row, left-sampling state for logical B tile-column-major. `G` is shared
across every active output column. The opposite A row stack is zero-copy for
`NT` and materialized once for `TT`.

## Pointer-batch descriptors (hot path only)

`Bstack` (one independently-obtained array per member, from arbitrary
offsets in `B`'s storage -- like [`ColumnRunCoupling`](@ref)'s `rowU`) is
reference-swapped and used, in the hot sampler
([`apply_left_row_run!`](@ref)'s final reduction), batched over the member
axis only (no `q_k` sub-structure at that point, since each member's tiles
are already folded into one `(bn, rB*qk)` matrix) -- so it needs only the
simple, `nmember`-length, single-slot-swapped descriptor form, same as
`rowU`. `Wstack_ptrs` covers that same call's other operand (this run's own
`Wbuf`, reshaped so its natural memory order matches), build-once and never
swapped. `betaU_ptrs`/`betaV_ptrs` (single-slot swap) and
`betaOm_ptrs`/`beta_tmp_ptrs` (build-once, since `Omega` here has no
per-member axis at all -- it is one matrix shared and broadcast across every
member, unlike [`RowRightRunCoupling`](@ref)'s) cover the beta accumulation.

The W-formation call (`S`, `G`, `Wbuf`) is left on the `Vector`-of-views path
even when descriptors are available: none of its three operands are ragged
(`S` and `Wbuf` are run-owned with stable per-slot addresses, and `G` has no
member axis to be ragged in), so it is not the same kind of problem `Bstack`
is -- it is left as a documented residual, matching `T`-formation in
[`RowRightRunCoupling`](@ref), rather than adding descriptor fields for
operands that were never the source of the allocation this pass targets. The
G-formation call above it and the co-range apply
([`apply_right_row_run!`](@ref), once per run, not once per pass) are
unaffected; see [`ColumnRunCoupling`](@ref)'s docstring and `docs/TODO.md`
for the general scoping rationale.
"""
struct RowLeftRunCoupling{ST,AStackT,BStackT,BUT,BVT,GT,WT,HT,TT,OT,BT,T}
    S::ST
    Astack::AStackT
    Bstack::BStackT
    betaU::BUT
    betaV::BVT
    Bstack_ptrs::Union{Nothing,BatchPtrDescriptor}
    Wstack_ptrs::Union{Nothing,BatchPtrDescriptor}
    betaU_ptrs::Union{Nothing,BatchPtrDescriptor}
    betaV_ptrs::Union{Nothing,BatchPtrDescriptor}
    betaOm_ptrs::Union{Nothing,BatchPtrDescriptor}
    beta_tmp_ptrs::Union{Nothing,BatchPtrDescriptor}
    G::GT
    Wbuf::WT
    H::HT
    Tbuf::TT
    Omega::OT
    beta_tmp::BT
    alpha::T
    qk::Int
end

function RowLeftRunCoupling(ops::LogicalTLROperands, i::Integer, cols;
                            alpha, beta=false, C=nothing,
                            block::Int, maxrank::Int,
                            rA::Int=rankdim(ops.au),
                            rB::Int=rankdim(ops.bv),
                            compute=nothing)
    ids = collect(Int, cols)
    nmember = length(ids)
    qk = ops.au.qn
    T = eltype(ops.au.data)
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    backend = get_backend(ops.au.data)
    bm = size(ops.au.data, 1)
    bn = size(ops.bv.data, 1)
    Ainner = [_trimmed_tile(ops.av, Int(i), kidx, rA) for kidx in 1:qk]
    Astack = _factor_row_stack(ops.au, Int(i), rA)
    Bouter = [[_trimmed_tile(ops.bu, kidx, j, rB) for kidx in 1:qk] for j in ids]
    Bstack = [_factor_column_stack(ops.bv, j, rB) for j in ids]

    S = allocate(backend, T, rA, rB, qk, nmember)
    if nmember > 0 && qk > 0 && rA > 0 && rB > 0
        Avec = [Ainner[kidx] for p in 1:nmember for kidx in 1:qk]
        Bvec = [Bouter[p][kidx] for p in 1:nmember for kidx in 1:qk]
        Svec = [view(S, :, :, kidx, p) for p in 1:nmember for kidx in 1:qk]
        precision_gemm_batched!(_adjoint_blas_char(T), 'N', one(T),
                                Avec, Bvec, zero(T), Svec, mode)
    end
    betaU, betaV = if iszero(beta)
        (nothing, nothing)
    else
        C === nothing && throw(ArgumentError("C must be supplied when beta != 0"))
        uv = [logical_tile_factors(C, Int(i), j) for j in ids]
        ([x[1] for x in uv], [x[2] for x in uv])
    end
    rC = betaU === nothing ? 0 : size(first(betaU), 2)

    Gbuf = allocate(backend, T, rA, block, qk)
    Wbuf = allocate(backend, T, rB, qk, block, nmember)
    Hbuf = allocate(backend, T, rB, maxrank, qk, nmember)
    Tbuf = allocate(backend, T, rA, qk, maxrank, nmember)
    Omega = allocate(backend, T, bm, block, 1)
    beta_tmp = allocate(backend, T, rC, max(block, maxrank), nmember)

    ptrs_ok = nmember > 0 && qk > 0 && supports_pointer_batched(backend)
    Bstack_ptrs = ptrs_ok ?
        BatchPtrDescriptor([reshape(Bstack[p], bn, rB * qk) for p in 1:nmember]) :
        nothing
    Wstack_ptrs = ptrs_ok ?
        BatchPtrDescriptor([view(reshape(Wbuf, rB * qk, block, nmember), :, :, p)
                           for p in 1:nmember]) :
        nothing
    has_beta_ptrs = ptrs_ok && betaU !== nothing
    betaU_ptrs = has_beta_ptrs ? BatchPtrDescriptor(betaU) : nothing
    betaV_ptrs = has_beta_ptrs ? BatchPtrDescriptor(betaV) : nothing
    betaOm_ptrs = has_beta_ptrs ?
        BatchPtrDescriptor([view(Omega, :, :, 1) for _ in 1:nmember]) : nothing
    beta_tmp_ptrs = has_beta_ptrs ?
        BatchPtrDescriptor([view(beta_tmp, :, :, p) for p in 1:nmember]) : nothing

    return RowLeftRunCoupling(
        S, Astack, Bstack, betaU, betaV,
        Bstack_ptrs, Wstack_ptrs, betaU_ptrs, betaV_ptrs, betaOm_ptrs,
        beta_tmp_ptrs,
        Gbuf, Wbuf, Hbuf, Tbuf, Omega, beta_tmp,
        T(alpha), qk,
    )
end

function _swap_row_left_members!(run::RowLeftRunCoupling, p::Int, q::Int)
    p == q && return run
    backend = get_backend(run.S)
    S3 = reshape(run.S, :, 1, size(run.S, 4))
    _ara_swap_basis_kernel!(backend)(S3, p, q; ndrange=(size(S3, 1), 1))
    run.Bstack[p], run.Bstack[q] = run.Bstack[q], run.Bstack[p]
    run.Bstack_ptrs !== nothing && swap_batch_ptrs!(run.Bstack_ptrs, p, q)
    if run.betaU !== nothing
        run.betaU[p], run.betaU[q] = run.betaU[q], run.betaU[p]
        run.betaV[p], run.betaV[q] = run.betaV[q], run.betaV[p]
        run.betaU_ptrs !== nothing && swap_batch_ptrs!(run.betaU_ptrs, p, q)
        run.betaV_ptrs !== nothing && swap_batch_ptrs!(run.betaV_ptrs, p, q)
    end
    return run
end

function apply_left_row_run!(Y::AbstractArray{T,3}, run::RowLeftRunCoupling,
                             sketch_width::Int, nactive::Int;
                             beta=false, compute=nothing) where {T}
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    adj = _adjoint_blas_char(T)
    qk = run.qk
    rA, rB = size(run.S, 1), size(run.S, 2)
    bn = size(Y, 1)
    Om = view(run.Omega, :, 1:sketch_width, :)
    Random.randn!(Om)
    use_ptrs = run.Bstack_ptrs !== nothing
    # Built once (when needed) and reused for both the final reduction and
    # the beta accumulation below, rather than once per GEMM call.
    Yptrs = use_ptrs ?
        BatchPtrDescriptor([view(Y, :, 1:sketch_width, p) for p in 1:nactive]) :
        nothing

    if qk > 0 && rA > 0 && rB > 0
        # run.Astack (bm, rA, qk) and Om (bm, sketch_width, 1) are both already
        # plain strided-3D arrays: Om has no per-kidx axis and broadcasts as a
        # batch-nmember-1 operand, so this dispatches as one strided-batched
        # GEMM over the qk axis with no pointer-batch Vector at all.
        G = view(run.G, :, 1:sketch_width, :)
        precision_gemm_batched!(adj, 'N', one(T), run.Astack, Om,
                                zero(T), G, mode)
        # W-formation is left on the Vector-of-views path even when
        # use_ptrs: S/G/Wbuf are all run-owned, none ragged, so this is a
        # documented residual (see the struct docstring), not the problem
        # Bstack's descriptors below address.
        Svec = [view(run.S, :, :, kidx, p) for p in 1:nactive for kidx in 1:qk]
        Gvec = [view(G, :, :, kidx) for p in 1:nactive for kidx in 1:qk]
        Wvec = [view(run.Wbuf, :, kidx, 1:sketch_width, p)
                for p in 1:nactive for kidx in 1:qk]
        precision_gemm_batched!(adj, 'N', run.alpha, Svec, Gvec,
                                zero(T), Wvec, mode)
        if use_ptrs
            Bref = reshape(first(run.Bstack), bn, rB * qk)
            Wbuf2 = reshape(run.Wbuf, rB * qk, size(run.Wbuf, 3), size(run.Wbuf, 4))
            Wref = view(Wbuf2, :, 1:sketch_width, 1)
            precision_gemm_batched_ptrs!('N', 'N', one(T),
                run.Bstack_ptrs, Bref, run.Wstack_ptrs, Wref,
                zero(T), Yptrs, view(Y, :, 1:sketch_width, 1), nactive, mode)
        else
            Bvec = [reshape(run.Bstack[p], bn, rB * qk) for p in 1:nactive]
            Wstack = [reshape(view(run.Wbuf, :, :, 1:sketch_width, p),
                              rB * qk, sketch_width) for p in 1:nactive]
            Yvec = [view(Y, :, 1:sketch_width, p) for p in 1:nactive]
            precision_gemm_batched!('N', 'N', one(T), Bvec, Wstack,
                                    zero(T), Yvec, mode)
        end
    else
        fill!(view(Y, :, 1:sketch_width, 1:nactive), zero(T))
    end

    if run.betaU !== nothing
        if use_ptrs
            tmpref = view(run.beta_tmp, :, 1:sketch_width, 1)
            Yptrs = BatchPtrDescriptor([view(Y, :, 1:sketch_width, p) for p in 1:nactive])
            precision_gemm_batched_ptrs!(adj, 'N', one(T),
                run.betaU_ptrs, first(run.betaU), run.betaOm_ptrs, view(Om, :, :, 1),
                zero(T), run.beta_tmp_ptrs, tmpref, nactive, mode)
            precision_gemm_batched_ptrs!('N', 'N', T(beta),
                run.betaV_ptrs, first(run.betaV), run.beta_tmp_ptrs, tmpref,
                one(T), Yptrs, view(Y, :, 1:sketch_width, 1), nactive, mode)
        else
            tmp = [view(run.beta_tmp, :, 1:sketch_width, p) for p in 1:nactive]
            Ovec = [view(Om, :, :, 1) for p in 1:nactive]
            Yvec = [view(Y, :, 1:sketch_width, p) for p in 1:nactive]
            precision_gemm_batched!(adj, 'N', one(T), view(run.betaU, 1:nactive),
                                    Ovec, zero(T), tmp, mode)
            precision_gemm_batched!('N', 'N', T(beta), view(run.betaV, 1:nactive),
                                    tmp, one(T), Yvec, mode)
        end
    end
    return Y
end

function apply_right_row_run!(Z::AbstractArray{T,3}, run::RowLeftRunCoupling,
                              Q::AbstractArray{T,3}, sketch_width::Int;
                              beta=false, compute=nothing) where {T}
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    adj = _adjoint_blas_char(T)
    nmember = size(Q, 3)
    qk = run.qk
    rA, rB = size(run.S, 1), size(run.S, 2)
    bm = size(Z, 1)
    if qk > 0 && rA > 0 && rB > 0
        Bvec = [view(run.Bstack[p], :, :, kidx) for p in 1:nmember for kidx in 1:qk]
        Qvec = [view(Q, :, 1:sketch_width, p) for p in 1:nmember for kidx in 1:qk]
        Hvec = [view(run.H, :, 1:sketch_width, kidx, p)
                for p in 1:nmember for kidx in 1:qk]
        precision_gemm_batched!(adj, 'N', one(T), Bvec, Qvec,
                                zero(T), Hvec, mode)
        Svec = [view(run.S, :, :, kidx, p) for p in 1:nmember for kidx in 1:qk]
        Tvec = [view(run.Tbuf, :, kidx, 1:sketch_width, p)
                for p in 1:nmember for kidx in 1:qk]
        precision_gemm_batched!('N', 'N', run.alpha, Svec, Hvec,
                                zero(T), Tvec, mode)
        # See apply_right_row_run! (RowRightRunCoupling) for the identical
        # broadcast-batch-1 reshape used here.
        Astack3 = reshape(run.Astack, bm, rA * qk, 1)
        Tbuf2 = reshape(run.Tbuf, rA * qk, size(run.Tbuf, 3), size(run.Tbuf, 4))
        Tstack = view(Tbuf2, :, 1:sketch_width, 1:nmember)
        Zview = view(Z, :, 1:sketch_width, 1:nmember)
        precision_gemm_batched!('N', 'N', one(T), Astack3, Tstack,
                                zero(T), Zview, mode)
    else
        fill!(Z, zero(T))
    end
    if run.betaU !== nothing
        tmp = [view(run.beta_tmp, :, 1:sketch_width, p) for p in 1:nmember]
        Qvec = [view(Q, :, 1:sketch_width, p) for p in 1:nmember]
        Zvec = [view(Z, :, 1:sketch_width, p) for p in 1:nmember]
        precision_gemm_batched!(adj, 'N', one(T), run.betaV, Qvec,
                                zero(T), tmp, mode)
        precision_gemm_batched!('N', 'N', T(beta), run.betaU, tmp,
                                one(T), Zvec, mode)
    end
    return Z
end
