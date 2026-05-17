#!/usr/bin/env julia
# Fully-async multi-GPU 2.5D sCQR3 — zero per-iteration host syncs.
#
# All inter-GPU dependencies are expressed via CUDA events on a *compute*
# stream and a *comm* stream per GPU. NCCL collectives run on the comm stream
# and the compute stream `wait()`s on the comm event when it needs the
# reduced result. The host issues ALL panels' work without ever blocking on
# `CUDA.synchronize()` until the final result is needed.
#
# Per-GPU streams:
#   - compute_stream[r]: cuBLAS SYRK / GEMM / POTRF / TRSM live here
#   - comm_stream[r]:    NCCL collectives (Allreduce of G and W) live here
#
# Per panel iter, the chain is:
#   compute_stream:  SYRK(G_partial) → record evt_compute
#   comm_stream:     wait(evt_compute) → NCCL.AllReduce(G_partial) → record evt_allreduce
#   compute_stream:  wait(evt_allreduce) → POTRF → TRSM
# All concurrently across the c GPUs; no `device!(...); synchronize()` in the
# inner loop.

using LinearAlgebra, Random, Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "NextLA.jl"))
using CUDA, NCCL, NextLA, KernelAbstractions

flushln(s...) = (println(s...); flush(stdout))

# ────────────────────────────────────────────────────────────────────────────
function multi_gpu_async_scqr3!(N::Int, A0::AbstractMatrix{T}, c::Integer;
                                 b::Int=0, passes::Int=3) where {T<:LinearAlgebra.BlasFloat}
    c = Int(c)
    m = n = N
    @assert m % c == 0 "m=$m must be divisible by c=$c"
    m_local = m ÷ c

    devs = collect(0:c-1)
    comms = NCCL.Communicators(devs)

    # Choose b heuristically (b_max = N / sqrt(P₁) where P₁=132 SMs).
    if b == 0
        b = N <= 4000  ? 363  :
            N <= 8000  ? 727  :
            N <= 12000 ? 1090 :
            N <= 16000 ? 1454 :
            N <= 20000 ? 1818 :
            N <= 24000 ? 2181 :
            N <= 30000 ? 2727 :
            N <= 40000 ? 3636 : 5400
    end

    # Per-GPU buffers, streams, events.
    A_local   = Vector{CuMatrix{T}}(undef, c)
    G_partial = Vector{CuMatrix{T}}(undef, c)
    W_partial = Vector{CuMatrix{T}}(undef, c)
    Racc_buf  = Vector{CuMatrix{T}}(undef, c)
    Rwrk_buf  = Vector{CuMatrix{T}}(undef, c)
    compute_streams = Vector{CuStream}(undef, c)
    comm_streams    = Vector{CuStream}(undef, c)
    evt_compute     = Vector{CuEvent}(undef, c)
    evt_allreduce   = Vector{CuEvent}(undef, c)
    R_acc_local     = Vector{CuMatrix{T}}(undef, c)

    for r in 1:c
        CUDA.device!(devs[r])
        A_local[r]   = CuArray(A0[(r-1)*m_local+1 : r*m_local, :])
        G_partial[r] = CUDA.zeros(T, b, b)
        W_partial[r] = CUDA.zeros(T, b, n)
        Racc_buf[r]  = CUDA.zeros(T, b, b)
        Rwrk_buf[r]  = CUDA.zeros(T, b, b)
        R_acc_local[r] = CUDA.zeros(T, n, n)
        compute_streams[r] = CuStream()
        comm_streams[r]    = CuStream()
        evt_compute[r]     = CuEvent()
        evt_allreduce[r]   = CuEvent()
    end

    # Convenience: helper to run a closure inside a particular device's
    # stream without touching the task-default stream.
    function on_stream(body::Function, r::Int, stream::CuStream)
        CUDA.device!(devs[r])
        CUDA.stream!(stream) do; body(); end
    end

    # ── Outer panel loop ──────────────────────────────────────────────────
    k = 1
    while k <= n
        sb = min(b, n - k + 1)
        n_tr = n - (k + sb - 1)

        # Phase Q1 — 3 sCQR3 iterations.
        for it in 1:passes
            # 1. SYRK on each GPU's compute stream.
            for r in 1:c
                on_stream(r, compute_streams[r]) do
                    Av = view(A_local[r], 1:m_local, k:(k+sb-1))
                    Gv = view(G_partial[r], 1:sb, 1:sb)
                    CUDA.CUBLAS.syrk!('U', 'T', one(T), Av, zero(T), Gv)
                end
            end

            # 2. Record event on compute_stream; comm_stream waits on it.
            for r in 1:c
                CUDA.device!(devs[r])
                CUDA.record(evt_compute[r], compute_streams[r])
                CUDA.wait(evt_compute[r], comm_streams[r])
            end

            # 3. NCCL Allreduce on comm_streams (group call → batched).
            NCCL.group() do
                for r in 1:c
                    CUDA.device!(devs[r])
                    CUDA.stream!(comm_streams[r]) do
                        NCCL.Allreduce!(G_partial[r], +, comms[r])
                    end
                end
            end

            # 4. Compute stream waits on allreduce done.
            for r in 1:c
                CUDA.device!(devs[r])
                CUDA.record(evt_allreduce[r], comm_streams[r])
                CUDA.wait(evt_allreduce[r], compute_streams[r])
            end

            # 5. POTRF + TRSM on compute_streams (still no host sync).
            for r in 1:c
                on_stream(r, compute_streams[r]) do
                    Gv = view(G_partial[r], 1:sb, 1:sb)
                    CUDA.CUSOLVER.potrf!('U', Gv)
                    Av = view(A_local[r], 1:m_local, k:(k+sb-1))
                    rdiv!(Av, UpperTriangular(Gv))
                end
            end
        end  # end sCQR3 iterations

        # Phase Q2 — trailing update on A[:, k+sb:n]
        if n_tr > 0
            # 1. W_local = Q^T · A_tr on compute_streams.
            for r in 1:c
                on_stream(r, compute_streams[r]) do
                    Qv  = view(A_local[r], 1:m_local, k:(k+sb-1))
                    Atr = view(A_local[r], 1:m_local, (k+sb):n)
                    Wv  = view(W_partial[r], 1:sb, 1:n_tr)
                    mul!(Wv, Qv', Atr)
                end
            end
            for r in 1:c
                CUDA.device!(devs[r])
                CUDA.record(evt_compute[r], compute_streams[r])
                CUDA.wait(evt_compute[r], comm_streams[r])
            end

            # 2. NCCL Allreduce W (full b × n_tr slab).
            NCCL.group() do
                for r in 1:c
                    CUDA.device!(devs[r])
                    CUDA.stream!(comm_streams[r]) do
                        Wv = view(W_partial[r], 1:sb, 1:n_tr)
                        NCCL.Allreduce!(reshape(Wv, sb*n_tr), +, comms[r])
                    end
                end
            end
            for r in 1:c
                CUDA.device!(devs[r])
                CUDA.record(evt_allreduce[r], comm_streams[r])
                CUDA.wait(evt_allreduce[r], compute_streams[r])
            end

            # 3. A_tr -= Q · W locally.
            for r in 1:c
                on_stream(r, compute_streams[r]) do
                    Qv  = view(A_local[r], 1:m_local, k:(k+sb-1))
                    Atr = view(A_local[r], 1:m_local, (k+sb):n)
                    Wv  = view(W_partial[r], 1:sb, 1:n_tr)
                    mul!(Atr, Qv, Wv, -one(T), one(T))
                end
            end
        end
        k += sb
    end

    # Now sync at the very end (this is the FIRST host sync).
    for r in 1:c
        CUDA.device!(devs[r])
        CUDA.synchronize(compute_streams[r])
        CUDA.synchronize(comm_streams[r])
    end

    # Gather Q on host for correctness.
    Q_full = Array{T}(undef, m, n)
    for r in 1:c
        CUDA.device!(devs[r])
        Q_full[(r-1)*m_local+1 : r*m_local, :] .= Array(A_local[r])
    end
    return Q_full
end

# ────────────────────────────────────────────────────────────────────────────
function main()
    flushln("==========================================================")
    flushln(" Fully-async multi-GPU 2.5D sCQR3 (NCCL+events) vs cuSOLVER")
    flushln(" host=", gethostname(), "  visible GPUs: ", length(CUDA.devices()))
    flushln("==========================================================")

    sizes = (8000, 16000, 32000)
    for N in sizes
        flushln("\n────── N=$N ──────")
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
        sort!(cu_ts); cu_tmin = cu_ts[1]
        @printf("  cuSOLVER (1 GPU)        tmin=%9.2f ms  (baseline)\n", cu_tmin); flush(stdout)

        # Async multi-GPU sweep.
        for c in (2, 4)
            c > length(CUDA.devices()) && continue
            (N % c) != 0 && continue
            for _ in 1:2; multi_gpu_async_scqr3!(N, A0_host, c); end
            ts = Float64[]
            res = -1.0
            for trial in 1:3
                t0 = time_ns()
                Q = multi_gpu_async_scqr3!(N, A0_host, c)
                t1 = time_ns()
                push!(ts, (t1-t0)/1e6)
                if trial == 1
                    CUDA.device!(0); Qd = CuArray(Q); res = norm(Qd'*Qd - I)
                end
            end
            sort!(ts)
            @printf("  async-NCCL c=%d         tmin=%9.2f ms  %.2fx  |Q'Q-I|=%.1e\n",
                    c, ts[1], cu_tmin/ts[1], res); flush(stdout)
        end
    end
end

main()
