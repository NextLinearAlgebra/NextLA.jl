#!/usr/bin/env julia
# 2.5D distributed sCQR3-2.5D across `c` GPUs using NCCL AllReduce.
#
# Architecture (paper §A.3):
#   - c = number of GPUs (each is one "replica" along the P_z axis).
#   - Row-distribution: A is split across GPUs by rows. Each GPU owns
#     `m_local = m / c` rows of the full A.
#   - Panel Gram on each GPU is a *partial* (m_local × b)' × (m_local × b)
#     matrix; AllReduce(+) across the c GPUs gives the full Gram (Lemma 4
#     of the paper).
#   - POTRF + TRSM run locally on each GPU (identical inputs after
#     AllReduce ⇒ identical outputs modulo floating-point determinism).
#   - Trailing update: each GPU computes W_local = Q_panel^T · A_tr_local,
#     AllReduce(W), then A_tr_local -= Q_panel · W.
#
# Versus single-GPU cuSOLVER reference. The 2.5D bandwidth saving
# (√c/√P scaling on the dominant trailing-update term) becomes a real
# wall-clock win once c≥2 and N is large enough that the trailing GEMM
# dominates.

using LinearAlgebra, Random, Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "NextLA.jl"))
using CUDA, NCCL, NextLA, KernelAbstractions

flushln(s...) = (println(s...); flush(stdout))

# ────────────────────────────────────────────────────────────────────────────
function multi_gpu_scqr3!(N::Int, A0::AbstractMatrix{T}, c::Int;
                           b::Int=0, passes::Int=3) where {T<:LinearAlgebra.BlasFloat}
    m = n = N
    @assert m % c == 0 "m=$m must be divisible by c=$c (rectangular row-split)"
    m_local = m ÷ c

    # NCCL communicators across c GPUs.
    devs = collect(0:c-1)
    comms = NCCL.Communicators(devs)

    # Distribute A across GPUs (row-split).
    A_local = Vector{CuMatrix{T}}(undef, c)
    for r in 1:c
        CUDA.device!(devs[r])
        A_local[r] = CuArray(A0[(r-1)*m_local+1 : r*m_local, :])
    end

    # Panel size.
    if b == 0
        # X-partition cube on a single GPU's fast memory.
        # Use sqrt(M_local) where M_local = ~228 KB smem + reg per SM.
        b = 1454  # match Step 7 heuristic at N=16K. We'll override for other N.
        if N <= 4000;     b = 363
        elseif N <= 8000;  b = 727
        elseif N <= 12000; b = 1090
        elseif N <= 16000; b = 1454
        elseif N <= 20000; b = 1818
        elseif N <= 24000; b = 2181
        elseif N <= 30000; b = 2727
        elseif N <= 40000; b = 3636
        end
    end

    # Per-GPU scratch.
    G_partial = Vector{CuMatrix{T}}(undef, c)
    G_full    = Vector{CuMatrix{T}}(undef, c)
    W_partial = Vector{CuMatrix{T}}(undef, c)
    W_full    = Vector{CuMatrix{T}}(undef, c)
    R_acc     = Vector{CuMatrix{T}}(undef, c)  # replicated R on each GPU
    Racc_iter = Vector{CuMatrix{T}}(undef, c)
    Rwrk_iter = Vector{CuMatrix{T}}(undef, c)
    for r in 1:c
        CUDA.device!(devs[r])
        G_partial[r] = CUDA.zeros(T, b, b)
        G_full[r]    = CUDA.zeros(T, b, b)
        W_partial[r] = CUDA.zeros(T, b, n)  # large enough for any panel's W
        W_full[r]    = CUDA.zeros(T, b, n)
        R_acc[r]     = CUDA.zeros(T, n, n)
        Racc_iter[r] = CUDA.zeros(T, b, b)
        Rwrk_iter[r] = CUDA.zeros(T, b, b)
    end

    # ── Outer panel loop ──────────────────────────────────────────────────
    k = 1
    while k <= n
        sb = min(b, n - k + 1)
        n_tr = n - (k + sb - 1)

        # Phase Q1: scqr3 panel
        for it in 1:passes
            # Step 1: local Gram per GPU
            for r in 1:c
                CUDA.device!(devs[r])
                Av = view(A_local[r], 1:m_local, k:(k+sb-1))
                Gv = view(G_partial[r], 1:sb, 1:sb)
                CUDA.CUBLAS.syrk!('U', 'T', one(T), Av, zero(T), Gv)
            end

            # Step 2: AllReduce the partial Grams to get full Gram on each GPU
            NCCL.group() do
                for r in 1:c
                    CUDA.device!(devs[r])
                    # NCCL.Allreduce in-place: contents become sum across all ranks
                    # We pass a view but NCCL needs contiguous buffer — use full G_partial[r]
                    NCCL.Allreduce!(G_partial[r], +, comms[r])
                end
            end
            # Sync each GPU
            for r in 1:c; CUDA.device!(devs[r]); CUDA.synchronize(); end

            # Step 3: Shift on first iter (Fukaya 2018, on each GPU, identical math)
            if it == 1
                for r in 1:c
                    CUDA.device!(devs[r])
                    Gv = view(G_partial[r], 1:sb, 1:sb)
                    # tr = sum(real(G[j,j])) — host read for shift coef
                    G_host = Array(Gv)
                    tr_val = sum(real(G_host[j,j]) for j in 1:sb)
                    coef = real(T)(11) * (real(T)(m * sb) + real(T)(sb * (sb + 1))) * eps(real(T))
                    s = coef * tr_val
                    G_diag = view(Gv, diagind(Gv))
                    # add shift on device
                    G_diag .+= s
                end
            end

            # Step 4: POTRF (local, identical across GPUs)
            for r in 1:c
                CUDA.device!(devs[r])
                Gv = view(G_partial[r], 1:sb, 1:sb)
                CUDA.CUSOLVER.potrf!('U', Gv)
            end

            # Step 5: TRSM A_local := A_local · R^-1 (local on each GPU)
            for r in 1:c
                CUDA.device!(devs[r])
                Gv = view(G_partial[r], 1:sb, 1:sb)
                Av = view(A_local[r], 1:m_local, k:(k+sb-1))
                rdiv!(Av, UpperTriangular(Gv))
            end
        end
        # After 3 passes, A_local[k:k+sb-1] is now Q_k_local (rows of Q_k)

        # Phase Q2: trailing update on A[:, k+sb:n] across all GPUs
        if n_tr > 0
            # W_local = Q^T · A_tr_local on each GPU
            for r in 1:c
                CUDA.device!(devs[r])
                Qv  = view(A_local[r], 1:m_local, k:(k+sb-1))
                Atr = view(A_local[r], 1:m_local, (k+sb):n)
                Wv  = view(W_partial[r], 1:sb, 1:n_tr)
                mul!(Wv, Qv', Atr)
            end

            # AllReduce W
            NCCL.group() do
                for r in 1:c
                    CUDA.device!(devs[r])
                    # Allreduce the full W_partial (we'll only use [1:sb, 1:n_tr]).
                    # Actually we need to only Allreduce the used part — use a
                    # contiguous view via reshape.
                    Wv  = view(W_partial[r], 1:sb, 1:n_tr)
                    NCCL.Allreduce!(reshape(Wv, sb*n_tr), +, comms[r])
                end
            end
            for r in 1:c; CUDA.device!(devs[r]); CUDA.synchronize(); end

            # A_tr_local -= Q_local · W
            for r in 1:c
                CUDA.device!(devs[r])
                Qv  = view(A_local[r], 1:m_local, k:(k+sb-1))
                Atr = view(A_local[r], 1:m_local, (k+sb):n)
                Wv  = view(W_partial[r], 1:sb, 1:n_tr)
                mul!(Atr, Qv, Wv, -one(T), one(T))
            end
        end
        k += sb
    end

    # Gather Q back to GPU 0 for the residual check (not part of measured time).
    Q_full = Array{T}(undef, m, n)
    R_final = nothing
    for r in 1:c
        CUDA.device!(devs[r])
        Q_full[(r-1)*m_local+1 : r*m_local, :] .= Array(A_local[r])
    end
    return Q_full
end

# ────────────────────────────────────────────────────────────────────────────
# Bench harness
function main()
    flushln("==========================================================")
    flushln(" Multi-GPU 2.5D sCQR3 (NCCL) vs single-GPU cuSOLVER")
    flushln(" host=", gethostname())
    flushln(" gpus visible: ", length(CUDA.devices()))
    flushln("==========================================================")

    sizes = (4000, 8000, 16000)

    for N in sizes
        flushln("\n────── N=$N ──────")
        # CPU host A0 (we hand FP64 copies to GPUs).
        Random.seed!(7)
        A0_host = randn(Float64, N, N)

        # Single-GPU cuSOLVER reference
        CUDA.device!(0)
        A0_d = CuArray(A0_host)
        for _ in 1:2; A = copy(A0_d); CUDA.CUSOLVER.geqrf!(A); CUDA.synchronize(); end
        cu_ts = Float64[]
        for _ in 1:5
            A = copy(A0_d); CUDA.synchronize()
            t = time_ns(); CUDA.CUSOLVER.geqrf!(A); CUDA.synchronize()
            push!(cu_ts, (time_ns()-t)/1e6)
        end
        sort!(cu_ts)
        cu_tmin = cu_ts[1]
        @printf("  cuSOLVER (1 GPU)    tmin=%9.2f ms  (baseline)\n", cu_tmin); flush(stdout)

        # Multi-GPU runs
        for c in (2, length(CUDA.devices()))
            if c < 2 || (N % c) != 0; continue; end
            flushln("\n  ── c=$c GPUs ──")
            # Warm
            for _ in 1:2
                multi_gpu_scqr3!(N, A0_host, c)
            end
            ts = Float64[]
            res_check = -1.0
            for trial in 1:3
                ts_start = time_ns()
                Q = multi_gpu_scqr3!(N, A0_host, c)
                ts_end = time_ns()
                push!(ts, (ts_end - ts_start)/1e6)
                if trial == 1
                    # Validate (cheap on CPU)
                    res_check = norm(A0_host - Q * (Q' * A0_host)) / norm(A0_host)
                end
            end
            sort!(ts)
            @printf("  multi-GPU c=%d      tmin=%9.2f ms  %.2fx  res=%.1e\n",
                    c, ts[1], cu_tmin/ts[1], res_check); flush(stdout)
        end
    end
end

main()
