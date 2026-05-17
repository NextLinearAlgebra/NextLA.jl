#!/usr/bin/env julia
# Look-ahead depth sweep on H200 single-GPU.
# Tests n_streams ∈ {1, 2, 3, 4, 8} for the look-ahead variant of sCQR3-2.5D.
# Reports timing + speedup vs n_streams=1 (sequential reference).

using LinearAlgebra, Random, Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "NextLA.jl"))
using NextLA, KernelAbstractions, CUDA

ENV["NEXTLA_FORCE_C1"] = "1"
ENV["NEXTLA_USE_GRAPH"] = "0"

flushln(s...) = (println(s...); flush(stdout))
flushln("===========================================================")
flushln(" H200 look-ahead depth-sweep:  n_streams ∈ {1, 2, 3, 4, 8}")
flushln("===========================================================")

for N in (4000, 8000, 16000, 30000)
    flushln("\n────── N=$N ──────")
    A0 = CUDA.randn(Float64, N, N)

    for s in (1, 2, 3, 4, 8)
        # warmup
        for _ in 1:2
            A = copy(A0); R = CUDA.zeros(Float64,N,N); tau = CUDA.zeros(Float64,N)
            NextLA.geqrf_2p5d_lookahead!(N,N,A,R,tau; n_streams=s)
            CUDA.synchronize()
        end
        Av = copy(A0); Rv = CUDA.zeros(Float64,N,N); tauv = CUDA.zeros(Float64,N)
        NextLA.geqrf_2p5d_lookahead!(N,N,Av,Rv,tauv; n_streams=s)
        CUDA.synchronize()
        res = norm(A0 - Av*Rv)/norm(A0)
        ort = norm(Av' * Av - I)
        # timed
        ts = Float64[]
        for _ in 1:5
            A = copy(A0); R = CUDA.zeros(Float64,N,N); tau = CUDA.zeros(Float64,N)
            CUDA.synchronize()
            t = time_ns(); NextLA.geqrf_2p5d_lookahead!(N,N,A,R,tau; n_streams=s); CUDA.synchronize()
            push!(ts, (time_ns()-t)/1e6)
        end
        sort!(ts)
        @printf("  n_streams=%d   tmin=%9.2f ms   res=%.1e  ortho=%.1e\n",
                s, ts[1], res, ort)
        flush(stdout)
    end
end
