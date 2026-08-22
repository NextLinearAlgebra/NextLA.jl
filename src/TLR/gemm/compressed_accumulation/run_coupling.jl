# Run-coupling state for the canonical TLR-output GEMM's rolling traversal ----

"""
    RunCoupling{Fixed}

Retained state for a fixed-column or fixed-row run, physically ordered by ARA
slot and swap-compacted by [`swap_run_members!`](@ref). Both directions use
`S = V_A'W_B` and the same three-stage sampler:

    proj   = shared' * Om                       # H (:column) or G (:row)
    sketch = (Fixed === :column ? S : S') * proj # T (:column) or W (:row)
    Y      = member[p] * sketch[p]               # same reduction either way

Complementary packing makes the fixed factor stack zero-copy and the swept
stack member-specific. Fixed columns right-sample; fixed rows left-sample, so
[`apply_run!`](@ref) changes only the middle transpose.

Member stacks have nonuniform offsets and require pointer-batched GEMM.
Member and beta descriptors are swap-tracked; scratch descriptors are reused
without swapping; pointer APIs require a descriptor for every operand. The
caller-owned `Y` descriptor is created by `apply_run!`.
[`apply_corange!`](@ref) runs once after convergence and uses view batches.
"""
mutable struct RunCoupling{Fixed,ST,SHT,MT,BUT,BVT,PT,SKT,OT,BT,T}
    S::ST
    shared::SHT
    member::MT
    betaU::BUT
    betaV::BVT
    member_ptrs::Union{Nothing,BatchPtrDescriptor}
    sketch_ptrs::Union{Nothing,BatchPtrDescriptor}
    betaU_ptrs::Union{Nothing,BatchPtrDescriptor}
    betaV_ptrs::Union{Nothing,BatchPtrDescriptor}
    sharedOm_ptrs::Union{Nothing,BatchPtrDescriptor}
    beta_tmp_ptrs::Union{Nothing,BatchPtrDescriptor}
    proj::PT
    sketch::SKT
    Omega::OT
    beta_tmp::BT
    alpha::T
    qk::Int
end

"""
    RunCoupling(Val(:column), ops, rows, j; alpha, beta=false, C=nothing, block, maxrank)

Fixed-column constructor: the run covers output tiles `(rows, j)`.
"""
function RunCoupling(::Val{:column}, ops, rows, j::Int;
                     alpha, beta=false, C=nothing,
                     block::Int, maxrank::Int,
                     rA::Int=size(ops.au.data, 2),
                     rB::Int=size(ops.bv.data, 2),
                     compute=nothing, arena=nothing)
    ids = collect(Int, rows)
    nmember = length(ids)

    # persistent factor stacks and constructor scratch
    # Temporary coupling factors use the arena to avoid traversal allocations.
    parena = run_persistent_t_arena(arena)
    tarena = run_t_arena(arena)
    member = [
        _factor_row_stack(ops.au, i, rA; arena=parena, force_pack=true)
        for i in ids
    ]
    rowV = [_factor_row_stack(ops.av, i, rA; arena=tarena) for i in ids]
    colW = _factor_column_stack(ops.bu, j, rB; arena=tarena)
    shared = _factor_column_stack(ops.bv, j, rB; arena=parena)
    T = eltype(shared)
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    qk = size(shared, 3)
    bm = size(first(member), 1)
    bn = size(shared, 1)
    backend = get_backend(shared)

    # coupling matrix S = V_A' W_B, batched over member and contraction tile
    S = workspace_array!(parena, backend, T, rA, rB, qk, nmember)
    if nmember > 0 && qk > 0 && rA > 0 && rB > 0
        left_factor_views = [view(rowV[p], :, :, kidx) for p in 1:nmember for kidx in 1:qk]
        right_factor_views = [view(colW, :, :, kidx) for p in 1:nmember for kidx in 1:qk]
        coupling_views = [view(S, :, :, kidx, p) for p in 1:nmember for kidx in 1:qk]
        precision_gemm_batched!(adjoint_blas_char(T), 'N', one(T),
                                left_factor_views, right_factor_views, zero(T), coupling_views, mode)
    end

    # existing-output beta factors
    betaU, betaV = if iszero(beta)
        (nothing, nothing)
    else
        C === nothing && throw(ArgumentError("C must be supplied when beta != 0"))
        ([compressed_ftlr_storage_outer(C, i, j) for i in ids],
         [compressed_ftlr_storage_inner(C, i, j) for i in ids])
    end

    # sampling and truncation scratch
    # Only phase scratch is rewound; persistent coupling data remains intact.
    arena_reset_phase!(arena)
    proj = workspace_array!(tarena, backend, T, rB, block, qk)
    sketch = workspace_array!(tarena, backend, T, rA, qk, block, nmember)
    Omega = workspace_array!(tarena, backend, T, bn, block, 1)
    beta_tmp = workspace_array!(tarena, backend, T,
                        betaV === nothing ? 0 : size(first(betaV), 2),
                        max(block, maxrank), nmember)

    # pointer-batched descriptors
    ptrs_ok = nmember > 0 && supports_pointer_batched(backend)
    member_ptrs = ptrs_ok ?
        BatchPtrDescriptor([reshape(member[p], bm, rA * qk) for p in 1:nmember]) :
        nothing
    sketch_ptrs = ptrs_ok ?
        BatchPtrDescriptor([view(reshape(sketch, rA * qk, block, nmember), :, :, p)
                           for p in 1:nmember]) :
        nothing
    has_beta_ptrs = ptrs_ok && betaU !== nothing
    betaU_ptrs = has_beta_ptrs ? BatchPtrDescriptor(betaU) : nothing
    betaV_ptrs = has_beta_ptrs ? BatchPtrDescriptor(betaV) : nothing
    sharedOm_ptrs = has_beta_ptrs ?
        BatchPtrDescriptor([view(Omega, :, :, 1) for _ in 1:nmember]) : nothing
    beta_tmp_ptrs = has_beta_ptrs ?
        BatchPtrDescriptor([view(beta_tmp, :, :, p) for p in 1:nmember]) : nothing

    return RunCoupling{:column,typeof(S),typeof(shared),typeof(member),
                       typeof(betaU),typeof(betaV),typeof(proj),typeof(sketch),
                       typeof(Omega),typeof(beta_tmp),T}(
        S, shared, member, betaU, betaV,
        member_ptrs, sketch_ptrs, betaU_ptrs, betaV_ptrs, sharedOm_ptrs, beta_tmp_ptrs,
        proj, sketch, Omega, beta_tmp, T(alpha), qk)
end

"""
    RunCoupling(Val(:row), ops, i, cols; alpha, beta=false, C=nothing, block, maxrank)

Fixed-row constructor: the run covers output tiles `(i, cols)`.
"""
function RunCoupling(::Val{:row}, ops, i::Int, cols;
                     alpha, beta=false, C=nothing,
                     block::Int, maxrank::Int,
                     rA::Int=size(ops.au.data, 2),
                     rB::Int=size(ops.bv.data, 2),
                     compute=nothing, arena=nothing)
    ids = collect(Int, cols)
    nmember = length(ids)
    qk = ops.au.qn
    T = eltype(ops.au.data)
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    backend = get_backend(ops.au.data)
    bm = size(ops.au.data, 1)
    bn = size(ops.bv.data, 1)
    parena = run_persistent_t_arena(arena)
    tarena = run_t_arena(arena)

    # member/shared factor panels
    Ainner = [_trimmed_tile(ops.av, i, kidx, rA) for kidx in 1:qk]
    shared = _factor_row_stack(ops.au, i, rA; arena=parena)
    Bouter = [[_trimmed_tile(ops.bu, kidx, j, rB) for kidx in 1:qk] for j in ids]
    member = [
        _factor_column_stack(ops.bv, j, rB; arena=parena, force_pack=true)
        for j in ids
    ]

    # coupling matrix S = V_A' W_B, batched over member and contraction tile
    S = workspace_array!(parena, backend, T, rA, rB, qk, nmember)
    if nmember > 0 && qk > 0 && rA > 0 && rB > 0
        left_factor_views = [Ainner[kidx] for p in 1:nmember for kidx in 1:qk]
        right_factor_views = [Bouter[p][kidx] for p in 1:nmember for kidx in 1:qk]
        coupling_views = [view(S, :, :, kidx, p) for p in 1:nmember for kidx in 1:qk]
        precision_gemm_batched!(adjoint_blas_char(T), 'N', one(T),
                                left_factor_views, right_factor_views, zero(T), coupling_views, mode)
    end

    # existing-output beta factors
    betaU, betaV = if iszero(beta)
        (nothing, nothing)
    else
        C === nothing && throw(ArgumentError("C must be supplied when beta != 0"))
        ([compressed_ftlr_storage_outer(C, i, j) for j in ids],
         [compressed_ftlr_storage_inner(C, i, j) for j in ids])
    end
    rC = betaU === nothing ? 0 : size(first(betaU), 2)

    # phase-only sampling and truncation scratch
    arena_reset_phase!(arena)
    proj = workspace_array!(tarena, backend, T, rA, block, qk)
    sketch = workspace_array!(tarena, backend, T, rB, qk, block, nmember)
    Omega = workspace_array!(tarena, backend, T, bm, block, 1)
    beta_tmp = workspace_array!(tarena, backend, T, rC, max(block, maxrank), nmember)

    # pointer-batched descriptors
    ptrs_ok = nmember > 0 && qk > 0 && supports_pointer_batched(backend)
    member_ptrs = ptrs_ok ?
        BatchPtrDescriptor([reshape(member[p], bn, rB * qk) for p in 1:nmember]) :
        nothing
    sketch_ptrs = ptrs_ok ?
        BatchPtrDescriptor([view(reshape(sketch, rB * qk, block, nmember), :, :, p)
                           for p in 1:nmember]) :
        nothing
    has_beta_ptrs = ptrs_ok && betaU !== nothing
    betaU_ptrs = has_beta_ptrs ? BatchPtrDescriptor(betaU) : nothing
    betaV_ptrs = has_beta_ptrs ? BatchPtrDescriptor(betaV) : nothing
    sharedOm_ptrs = has_beta_ptrs ?
        BatchPtrDescriptor([view(Omega, :, :, 1) for _ in 1:nmember]) : nothing
    beta_tmp_ptrs = has_beta_ptrs ?
        BatchPtrDescriptor([view(beta_tmp, :, :, p) for p in 1:nmember]) : nothing

    return RunCoupling{:row,typeof(S),typeof(shared),typeof(member),
                       typeof(betaU),typeof(betaV),typeof(proj),typeof(sketch),
                       typeof(Omega),typeof(beta_tmp),T}(
        S, shared, member, betaU, betaV,
        member_ptrs, sketch_ptrs, betaU_ptrs, betaV_ptrs, sharedOm_ptrs, beta_tmp_ptrs,
        proj, sketch, Omega, beta_tmp, T(alpha), qk)
end

"""Swap active slots `p` and `q`; `shared` has no member axis."""
function swap_run_members!(run::RunCoupling, p::Int, q::Int)
    p == q && return run

    # coupling and member factors
    backend = get_backend(run.S)
    S3 = reshape(run.S, :, 1, size(run.S, 4))
    ara_swap_basis_kernel!(backend)(S3, p, q; ndrange=(size(S3, 1), 1))
    run.member[p], run.member[q] = run.member[q], run.member[p]
    run.member_ptrs !== nothing && swap_batch_ptrs!(run.member_ptrs, p, q)

    # beta factors
    if run.betaU !== nothing
        run.betaU[p], run.betaU[q] = run.betaU[q], run.betaU[p]
        run.betaV[p], run.betaV[q] = run.betaV[q], run.betaV[p]
        run.betaU_ptrs !== nothing && swap_batch_ptrs!(run.betaU_ptrs, p, q)
        run.betaV_ptrs !== nothing && swap_batch_ptrs!(run.betaV_ptrs, p, q)
    end
    return run
end

"""
    apply_run!(Y, run, sketch_width, nactive; beta, compute)

Hot per-pass sampler: fixed columns right-sample and fixed rows left-sample.
One projection batch is independent of `nactive`; the other stages use the
active prefix. Supported backends reuse persistent pointer descriptors, while
the caller-owned `Y` descriptor is built per call; CPU uses view batches.
"""
function apply_run!(Y::AbstractArray{T,3}, run::RunCoupling{:column},
                    sketch_width::Int, nactive::Int;
                    beta=false, compute=nothing) where {T}
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    adj = adjoint_blas_char(T)
    qk = run.qk
    bm = size(Y, 1)
    rA, rB = size(run.S, 1), size(run.S, 2)
    use_ptrs = run.member_ptrs !== nothing

    # random sketch and output descriptor
    Om = view(run.Omega, :, 1:sketch_width, :)
    Random.randn!(Om)
    proj = view(run.proj, :, 1:sketch_width, :)

    Yptrs = use_ptrs ?
        BatchPtrDescriptor([view(Y, :, 1:sketch_width, p) for p in 1:nactive]) :
        nothing

    # operator samples
    if qk > 0 && rA > 0 && rB > 0
        # shared projection
        precision_gemm_batched!(adj, 'N', one(T), run.shared, Om,
                                zero(T), proj, mode)

        # coupling contraction
        coupling_views = [view(run.S, :, :, kidx, p) for p in 1:nactive for kidx in 1:qk]
        proj_views = [view(proj, :, :, kidx) for p in 1:nactive for kidx in 1:qk]
        sketch_views = [view(run.sketch, :, kidx, 1:sketch_width, p)
                for p in 1:nactive for kidx in 1:qk]
        precision_gemm_batched!('N', 'N', run.alpha, coupling_views, proj_views,
                                zero(T), sketch_views, mode)

        # member reduction
        if use_ptrs
            Uref = reshape(run.member[1], bm, rA * qk)
            Sref = view(reshape(run.sketch, rA * qk, size(run.sketch, 3),
                               size(run.sketch, 4)), :, 1:sketch_width, 1)
            precision_gemm_batched_ptrs!('N', 'N', one(T),
                run.member_ptrs, Uref, run.sketch_ptrs, Sref,
                zero(T), Yptrs, view(Y, :, 1:sketch_width, 1), nactive, mode)
        else
            member_views = [reshape(run.member[p], bm, rA * qk) for p in 1:nactive]
            sketch_stacks = [reshape(view(run.sketch, :, :, 1:sketch_width, p), rA * qk, sketch_width)
                      for p in 1:nactive]
            output_views = [view(Y, :, 1:sketch_width, p) for p in 1:nactive]
            precision_gemm_batched!('N', 'N', one(T), member_views, sketch_stacks,
                                    zero(T), output_views, mode)
        end
    else
        fill!(view(Y, :, 1:sketch_width, :), zero(T))
    end

    # beta contribution
    if run.betaU !== nothing
        if use_ptrs
            tmpref = view(run.beta_tmp, :, 1:sketch_width, 1)
            precision_gemm_batched_ptrs!(adj, 'N', one(T),
                run.betaV_ptrs, first(run.betaV), run.sharedOm_ptrs, view(Om, :, :, 1),
                zero(T), run.beta_tmp_ptrs, tmpref, nactive, mode)
            precision_gemm_batched_ptrs!('N', 'N', T(beta),
                run.betaU_ptrs, first(run.betaU), run.beta_tmp_ptrs, tmpref,
                one(T), Yptrs, view(Y, :, 1:sketch_width, 1), nactive, mode)
        else
            tmp = [view(run.beta_tmp, :, 1:sketch_width, p) for p in 1:nactive]
            omega_views = [view(Om, :, :, 1) for _ in 1:nactive]
            output_views = [view(Y, :, 1:sketch_width, p) for p in 1:nactive]
            precision_gemm_batched!(adj, 'N', one(T), view(run.betaV, 1:nactive),
                                    omega_views, zero(T), tmp, mode)
            precision_gemm_batched!('N', 'N', T(beta), view(run.betaU, 1:nactive),
                                    tmp, one(T), output_views, mode)
        end
    end
    return Y
end

function apply_run!(Y::AbstractArray{T,3}, run::RunCoupling{:row},
                    sketch_width::Int, nactive::Int;
                    beta=false, compute=nothing) where {T}
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    adj = adjoint_blas_char(T)
    qk = run.qk
    rA, rB = size(run.S, 1), size(run.S, 2)
    bn = size(Y, 1)

    # random sketch and output descriptor
    Om = view(run.Omega, :, 1:sketch_width, :)
    Random.randn!(Om)
    use_ptrs = run.member_ptrs !== nothing
    Yptrs = use_ptrs ?
        BatchPtrDescriptor([view(Y, :, 1:sketch_width, p) for p in 1:nactive]) :
        nothing

    # operator samples
    if qk > 0 && rA > 0 && rB > 0
        # shared strided projection
        # `Om` broadcasts over the contraction-tile batch.
        proj = view(run.proj, :, 1:sketch_width, :)
        precision_gemm_batched!(adj, 'N', one(T), run.shared, Om,
                                zero(T), proj, mode)

        # coupling contraction
        coupling_views = [view(run.S, :, :, kidx, p) for p in 1:nactive for kidx in 1:qk]
        proj_views = [view(proj, :, :, kidx) for p in 1:nactive for kidx in 1:qk]
        sketch_views = [view(run.sketch, :, kidx, 1:sketch_width, p)
                for p in 1:nactive for kidx in 1:qk]
        precision_gemm_batched!(adj, 'N', run.alpha, coupling_views, proj_views,
                                zero(T), sketch_views, mode)

        # member reduction
        if use_ptrs
            Bref = reshape(first(run.member), bn, rB * qk)
            sketch2 = reshape(run.sketch, rB * qk, size(run.sketch, 3), size(run.sketch, 4))
            Sref = view(sketch2, :, 1:sketch_width, 1)
            precision_gemm_batched_ptrs!('N', 'N', one(T),
                run.member_ptrs, Bref, run.sketch_ptrs, Sref,
                zero(T), Yptrs, view(Y, :, 1:sketch_width, 1), nactive, mode)
        else
            member_views = [reshape(run.member[p], bn, rB * qk) for p in 1:nactive]
            sketch_stacks = [reshape(view(run.sketch, :, :, 1:sketch_width, p),
                              rB * qk, sketch_width) for p in 1:nactive]
            output_views = [view(Y, :, 1:sketch_width, p) for p in 1:nactive]
            precision_gemm_batched!('N', 'N', one(T), member_views, sketch_stacks,
                                    zero(T), output_views, mode)
        end
    else
        fill!(view(Y, :, 1:sketch_width, 1:nactive), zero(T))
    end

    # beta contribution
    if run.betaU !== nothing
        if use_ptrs
            tmpref = view(run.beta_tmp, :, 1:sketch_width, 1)
            precision_gemm_batched_ptrs!(adj, 'N', one(T),
                run.betaU_ptrs, first(run.betaU), run.sharedOm_ptrs, view(Om, :, :, 1),
                zero(T), run.beta_tmp_ptrs, tmpref, nactive, mode)
            precision_gemm_batched_ptrs!('N', 'N', T(beta),
                run.betaV_ptrs, first(run.betaV), run.beta_tmp_ptrs, tmpref,
                one(T), Yptrs, view(Y, :, 1:sketch_width, 1), nactive, mode)
        else
            tmp = [view(run.beta_tmp, :, 1:sketch_width, p) for p in 1:nactive]
            omega_views = [view(Om, :, :, 1) for p in 1:nactive]
            output_views = [view(Y, :, 1:sketch_width, p) for p in 1:nactive]
            precision_gemm_batched!(adj, 'N', one(T), view(run.betaU, 1:nactive),
                                    omega_views, zero(T), tmp, mode)
            precision_gemm_batched!('N', 'N', T(beta), view(run.betaV, 1:nactive),
                                    tmp, one(T), output_views, mode)
        end
    end
    return Y
end

"""
    apply_corange!(Z, run, Q, sketch_width; beta, compute, proj, sketch, beta_tmp, slot0=1)

Form `Z_slot = X_tile(slot)'Q_slot` after convergence. `S` and `Q` share the
final slot order. External `proj` and `sketch` use the opposite fixed-axis
shape because this apply builds the complementary basis; see [`apply_run!`](@ref).
"""
function apply_corange!(Z::AbstractArray{T,3}, run::RunCoupling{:column},
                        Q::AbstractArray{T,3}, sketch_width::Int;
                        beta=false, compute=nothing,
                        proj, sketch, beta_tmp, slot0::Int=1) where {T}
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    adj = adjoint_blas_char(T)
    nmember = size(Q, 3)
    qk = run.qk
    rA, rB = size(run.S, 1), size(run.S, 2)
    bn = size(Z, 1)

    # complementary operator samples
    if qk > 0 && rA > 0 && rB > 0
        member_views = [view(run.member[slot0 + p - 1], :, :, kidx) for p in 1:nmember for kidx in 1:qk]
        input_views = [view(Q, :, 1:sketch_width, p) for p in 1:nmember for kidx in 1:qk]
        proj_views = [view(proj, :, 1:sketch_width, kidx, p) for p in 1:nmember for kidx in 1:qk]
        precision_gemm_batched!(adj, 'N', one(T), member_views, input_views,
                                zero(T), proj_views, mode)

        coupling_views = [view(run.S, :, :, kidx, slot0 + p - 1) for p in 1:nmember for kidx in 1:qk]
        sketch_views = [view(sketch, :, kidx, 1:sketch_width, p)
                for p in 1:nmember for kidx in 1:qk]
        precision_gemm_batched!(adj, 'N', run.alpha, coupling_views, proj_views,
                                zero(T), sketch_views, mode)

        shared_stack = reshape(run.shared, bn, rB * qk)
        tasks = [GroupedGemmTask(
            'N', 'N', one(T), shared_stack,
            reshape(view(sketch, :, :, 1:sketch_width, p),
                    rB * qk, sketch_width),
            zero(T), view(Z, :, 1:sketch_width, p)) for p in 1:nmember]
        precision_gemm_grouped!(tasks, mode)
    else
        fill!(Z, zero(T))
    end

    # beta contribution
    if run.betaU !== nothing
        tmp = [view(beta_tmp, :, 1:sketch_width, p) for p in 1:nmember]
        input_views = [view(Q, :, 1:sketch_width, p) for p in 1:nmember]
        out_views = [view(Z, :, 1:sketch_width, p) for p in 1:nmember]
        betaU = view(run.betaU, slot0:(slot0 + nmember - 1))
        betaV = view(run.betaV, slot0:(slot0 + nmember - 1))
        precision_gemm_batched!(adj, 'N', one(T), betaU, input_views,
                                zero(T), tmp, mode)
        precision_gemm_batched!('N', 'N', T(beta), betaV, tmp,
                                one(T), out_views, mode)
    end
    return Z
end

function apply_corange!(Z::AbstractArray{T,3}, run::RunCoupling{:row},
                        Q::AbstractArray{T,3}, sketch_width::Int;
                        beta=false, compute=nothing,
                        proj, sketch, beta_tmp, slot0::Int=1) where {T}
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    adj = adjoint_blas_char(T)
    nmember = size(Q, 3)
    qk = run.qk
    rA, rB = size(run.S, 1), size(run.S, 2)
    bm = size(Z, 1)

    # complementary operator samples
    if qk > 0 && rA > 0 && rB > 0
        member_views = [view(run.member[slot0 + p - 1], :, :, kidx) for p in 1:nmember for kidx in 1:qk]
        input_views = [view(Q, :, 1:sketch_width, p) for p in 1:nmember for kidx in 1:qk]
        proj_views = [view(proj, :, 1:sketch_width, kidx, p)
                for p in 1:nmember for kidx in 1:qk]
        precision_gemm_batched!(adj, 'N', one(T), member_views, input_views,
                                zero(T), proj_views, mode)

        coupling_views = [view(run.S, :, :, kidx, slot0 + p - 1) for p in 1:nmember for kidx in 1:qk]
        sketch_views = [view(sketch, :, kidx, 1:sketch_width, p)
                for p in 1:nmember for kidx in 1:qk]
        precision_gemm_batched!('N', 'N', run.alpha, coupling_views, proj_views,
                                zero(T), sketch_views, mode)

        # grouped reshaped arena subviews
        shared_stack = reshape(run.shared, bm, rA * qk)
        tasks = [GroupedGemmTask(
            'N', 'N', one(T), shared_stack,
            reshape(view(sketch, :, :, 1:sketch_width, p),
                    rA * qk, sketch_width),
            zero(T), view(Z, :, 1:sketch_width, p)) for p in 1:nmember]
        precision_gemm_grouped!(tasks, mode)
    else
        fill!(Z, zero(T))
    end

    # beta contribution
    if run.betaU !== nothing
        tmp = [view(beta_tmp, :, 1:sketch_width, p) for p in 1:nmember]
        input_views = [view(Q, :, 1:sketch_width, p) for p in 1:nmember]
        out_views = [view(Z, :, 1:sketch_width, p) for p in 1:nmember]
        betaU = view(run.betaU, slot0:(slot0 + nmember - 1))
        betaV = view(run.betaV, slot0:(slot0 + nmember - 1))
        precision_gemm_batched!(adj, 'N', one(T), betaV, input_views,
                                zero(T), tmp, mode)
        precision_gemm_batched!('N', 'N', T(beta), betaU, tmp,
                                one(T), out_views, mode)
    end
    return Z
end

# Factor-stack packing helpers ------------------------------------------------

@kernel function _pack_factor_row_kernel!(dest, src, row::Int, qm::Int, qn::Int,
                                          row_major::Bool)
    i, k, kidx = @index(Global, NTuple)
    tile = row_major ? kidx + (row - 1) * qn : row + (kidx - 1) * qm
    @inbounds dest[i, k, kidx] = src[i, k, tile]
end

@kernel function _pack_factor_column_kernel!(dest, src, col::Int, qm::Int, qn::Int,
                                             row_major::Bool)
    i, k, kidx = @index(Global, NTuple)
    tile = row_major ? col + (kidx - 1) * qn : kidx + (col - 1) * qm
    @inbounds dest[i, k, kidx] = src[i, k, tile]
end

function _factor_row_stack(p, row::Int, rank::Int;
                           arena=nothing, force_pack::Bool=false)
    if !force_pack && p.order isa TileRowMajor && rank == size(p.data, 2)
        return view(p.data, :, :,
                    (row - 1) * p.qn + 1:row * p.qn)
    end

    # packed fallback
    backend = get_backend(p.data)
    dest = workspace_array!(arena, backend, eltype(p.data), size(p.data, 1), rank, p.qn)
    rank == 0 && return dest
    _pack_factor_row_kernel!(backend)(
        dest, p.data, row, p.qm, p.qn, p.order isa TileRowMajor;
        ndrange=size(dest),
    )
    return dest
end

function _factor_column_stack(p, col::Int, rank::Int;
                              arena=nothing, force_pack::Bool=false)
    if !force_pack && p.order isa TileColMajor && rank == size(p.data, 2)
        return view(p.data, :, :,
                    (col - 1) * p.qm + 1:col * p.qm)
    end

    # packed fallback
    backend = get_backend(p.data)
    dest = workspace_array!(arena, backend, eltype(p.data), size(p.data, 1), rank, p.qm)
    rank == 0 && return dest
    _pack_factor_column_kernel!(backend)(
        dest, p.data, col, p.qm, p.qn, p.order isa TileRowMajor;
        ndrange=size(dest),
    )
    return dest
end

@inline function _trimmed_tile(p, i::Int, j::Int, rank::Int)
    slot = tile_linear_index(p.order, p.qm, p.qn, i, j)
    return view(p.data, :, 1:rank, slot)
end

function _pack_factor_row_into!(dest, p, row::Int)
    isempty(dest) && return dest
    _pack_factor_row_kernel!(get_backend(dest))(
        dest, p.data, row, p.qm, p.qn, p.order isa TileRowMajor;
        ndrange=size(dest),
    )
    return dest
end

function _pack_factor_column_into!(dest, p, col::Int)
    isempty(dest) && return dest
    _pack_factor_column_kernel!(get_backend(dest))(
        dest, p.data, col, p.qm, p.qn, p.order isa TileRowMajor;
        ndrange=size(dest),
    )
    return dest
end

# Rolling-admission support -------------------------------------------------

function _update_beta_slot!(run::RunCoupling{:column}, C, fixed::Int, member::Int, slot::Int)
    run.betaU === nothing && return
    u = compressed_ftlr_storage_outer(C, member, fixed)
    v = compressed_ftlr_storage_inner(C, member, fixed)
    run.betaU[slot] = u
    run.betaV[slot] = v
    run.betaU_ptrs !== nothing &&
        set_batch_ptrs!(run.betaU_ptrs, slot, [u])
    run.betaV_ptrs !== nothing &&
        set_batch_ptrs!(run.betaV_ptrs, slot, [v])
end

function _update_beta_slot!(run::RunCoupling{:row}, C, fixed::Int, member::Int, slot::Int)
    run.betaU === nothing && return
    u = compressed_ftlr_storage_outer(C, fixed, member)
    v = compressed_ftlr_storage_inner(C, fixed, member)
    run.betaU[slot] = u
    run.betaV[slot] = v
    run.betaU_ptrs !== nothing &&
        set_batch_ptrs!(run.betaU_ptrs, slot, [u])
    run.betaV_ptrs !== nothing &&
        set_batch_ptrs!(run.betaV_ptrs, slot, [v])
end

function admit_wave!(run::RunCoupling{:column}, ops, rows::AbstractVector{Int},
                     slots::UnitRange{Int}, j::Int, arena;
                     C=nothing, compute=nothing)
    isempty(slots) && return run
    length(rows) == length(slots) || throw(DimensionMismatch("rows/slots mismatch"))

    # coupling inputs
    T = eltype(run.S)
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    tarena = run_t_arena(arena)
    rA, rB, qk = size(run.S, 1), size(run.S, 2), run.qk
    rowV = [
        _factor_row_stack(ops.av, row, rA; arena=tarena, force_pack=true)
        for row in rows
    ]
    colW = _factor_column_stack(
        ops.bu, j, rB; arena=tarena, force_pack=true)

    # member and beta factors
    for (slot, row) in zip(slots, rows)
        _pack_factor_row_into!(run.member[slot], ops.au, row)
        C === nothing || _update_beta_slot!(run, C, j, row, slot)
    end

    # admitted coupling matrices
    if qk > 0 && rA > 0 && rB > 0
        left = [view(rowV[p], :, :, k) for p in eachindex(rows) for k in 1:qk]
        right = [view(colW, :, :, k) for _ in rows for k in 1:qk]
        dest = [view(run.S, :, :, k, slot) for slot in slots for k in 1:qk]
        precision_gemm_batched!(
            adjoint_blas_char(T), 'N', one(T), left, right, zero(T), dest, mode)
    end
    return run
end

function admit_wave!(run::RunCoupling{:row}, ops, cols::AbstractVector{Int},
                     slots::UnitRange{Int}, i::Int, arena;
                     C=nothing, compute=nothing)
    isempty(slots) && return run
    length(cols) == length(slots) || throw(DimensionMismatch("cols/slots mismatch"))

    # coupling inputs
    T = eltype(run.S)
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    rA, rB, qk = size(run.S, 1), size(run.S, 2), run.qk
    Ainner = [_trimmed_tile(ops.av, i, k, rA) for k in 1:qk]

    # member and beta factors
    for (slot, col) in zip(slots, cols)
        _pack_factor_column_into!(run.member[slot], ops.bv, col)
        C === nothing || _update_beta_slot!(run, C, i, col, slot)
    end

    # admitted coupling matrices
    if qk > 0 && rA > 0 && rB > 0
        left = [Ainner[k] for _ in cols for k in 1:qk]
        right = [
            _trimmed_tile(ops.bu, k, col, rB) for col in cols for k in 1:qk
        ]
        dest = [view(run.S, :, :, k, slot) for slot in slots for k in 1:qk]
        precision_gemm_batched!(
            adjoint_blas_char(T), 'N', one(T), left, right, zero(T), dest, mode)
    end
    return run
end

"""Rebind sampling arrays and descriptors after phase-arena reset."""
function rebind_sampling_scratch!(run::RunCoupling, arena)
    a = run_t_arena(arena)
    backend = get_backend(run.S)

    # phase arrays
    run.proj = workspace_array!(a, backend, eltype(run.S), size(run.proj)...)
    run.sketch = workspace_array!(a, backend, eltype(run.S), size(run.sketch)...)
    run.Omega = workspace_array!(a, backend, eltype(run.S), size(run.Omega)...)
    run.beta_tmp = workspace_array!(
        a, backend, eltype(run.S), size(run.beta_tmp)...)

    # refreshed descriptors
    cap = size(run.S, 4)
    run.sketch_ptrs !== nothing && set_batch_ptrs!(
        run.sketch_ptrs, 1,
        [view(reshape(run.sketch, :, size(run.sketch, 3), size(run.sketch, 4)),
              :, :, p) for p in 1:size(run.sketch, 4)])
    run.sharedOm_ptrs !== nothing && set_batch_ptrs!(
        run.sharedOm_ptrs, 1,
        [view(run.Omega, :, :, 1) for _ in 1:cap])
    run.beta_tmp_ptrs !== nothing && set_batch_ptrs!(
        run.beta_tmp_ptrs, 1,
        [view(run.beta_tmp, :, :, p) for p in 1:cap])
    return run
end

# Arena sizing ------------------------------------------------------------

"""
    ara_run_workspace_bytes(family, rA, rB, qk, nmember, block, maxrank, bm, bn,
                            ::Type{T}, ::Type{Thi})
        -> (persistent_t_bytes, phase_t_bytes, phase_thi_bytes)

Return persistent and peak phase bytes for one `ARARunArena`. Sampling and
finalization share a rewound phase arena; `Q`, `S`, and packed factor stacks
persist. `maxrank` bounds both achieved and beta-term ranks, so the result is
an upper bound enforced by `workspace_array!`.
"""
function ara_run_workspace_bytes(family::Symbol, rA::Int, rB::Int, qk::Int,
                                 nmember::Int, block::Int, maxrank::Int,
                                 bm::Int, bn::Int, ::Type{T}, ::Type{Thi}) where {T,Thi}
    blk = min(block, max(maxrank, 1))

    # family-independent ARA scratch, parameterized by basis height
    ara_sample_t(m) = m * blk * nmember +         # Yblk
               max(maxrank, 1) * blk * nmember +  # Dproj
               2 * blk * blk * nmember            # R1, R2
    ara_thi(m) = m * blk * nmember + blk * blk * nmember  # Y_hi, G_hi

    persistent, phase, thi_elems = if family === :column
        persist_t = nmember * bm * rA * qk +      # member (rowU)
                bn * rB * qk +                     # shared (colZ)
                rA * rB * qk * nmember +           # S
                bm * maxrank * nmember             # Q
        constructor_t = nmember * bm * rA * qk +  # rowV packing
                        bn * rB * qk                # colW packing
        sample_t = rB * block * qk +               # proj (Hbuf)
                   rA * qk * block * nmember +     # sketch (Tbuf)
                   bn * block +                    # Omega
                   maxrank * max(block, maxrank) * nmember + # beta_tmp
                   ara_sample_t(bm)
        final_t = rA * maxrank * qk * nmember +    # proj (G)
                  rB * qk * maxrank * nmember +    # sketch (Wbuf)
                  maxrank * maxrank * nmember +    # beta_tmp
                  bn * maxrank * nmember +         # Z
                  bm * maxrank * nmember +         # Uh
                  bn * maxrank * nmember           # Vh
        (persist_t, max(constructor_t, sample_t, final_t), ara_thi(bm))
    elseif family === :row_left
        persist_t = bm * rA * qk +                # shared (Astack)
                    bn * rB * qk * nmember +       # member (Bstack)
                    rA * rB * qk * nmember +       # S
                    bn * maxrank * nmember         # right basis Q
        sample_t = rA * block * qk +              # proj (Gbuf)
                   rB * qk * block * nmember +     # sketch (Wbuf)
                   bm * block +                    # Omega
                   maxrank * max(block, maxrank) * nmember + # beta_tmp
                   ara_sample_t(bn)
        final_t = rB * maxrank * qk * nmember +   # proj (H)
                  rA * qk * maxrank * nmember +   # sketch (Tbuf)
                  maxrank * maxrank * nmember +   # beta_tmp
                  bm * maxrank * nmember +        # L
                  bm * maxrank * nmember +        # Uh
                  bn * maxrank * nmember          # Vh
        (persist_t, max(sample_t, final_t), ara_thi(bn))
    else
        throw(ArgumentError("unknown run family $family"))
    end

    return (
        persistent_t_bytes=persistent * sizeof(T),
        phase_t_bytes=phase * sizeof(T),
        phase_thi_bytes=thi_elems * sizeof(Thi),
    )
end
