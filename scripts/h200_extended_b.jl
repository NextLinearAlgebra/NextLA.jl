#!/usr/bin/env julia
# Extended b-sweep at N=8000 (and an N=4000 control) to see if pushing b to
# the admissible-window upper bound (or beyond with a larger M proxy) changes
# the NextLA / cuSOLVER ratio.

using LinearAlgebra, Random, Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "NextLA.jl"))
using NextLA, KernelAbstractions, CUDA

flushln(s...) = (println(s...); flush(stdout))
ENV["NEXTLA_FORCE_C1"] = "1"
CUDA.math_mode!(CUDA.DEFAULT_MATH)

function bench_pair(::Type{T}, m, n; b, c=1, nwarm=2, nrun=5) where {T}
    A0 = CUDA.randn(T, m, n)
    be = CUDABackend()
    p = compute_params(be, T, n; b=b, c=c)
    # cuSOLVER
    for _ in 1:nwarm; A = copy(A0); CUDA.CUSOLVER.geqrf!(A); CUDA.synchronize(); end
    cu_ts = Float64[]
    for _ in 1:nrun
        A = copy(A0); CUDA.synchronize()
        t = time_ns(); CUDA.CUSOLVER.geqrf!(A); CUDA.synchronize()
        push!(cu_ts, (time_ns() - t) / 1e6)
    end
    sort!(cu_ts)
    # NextLA
    for _ in 1:nwarm
        A = copy(A0); R = CUDA.zeros(T,n,n); tau = CUDA.zeros(T,n)
        NextLA.geqrf_2p5d!(m, n, A, R, tau; params=p)
        CUDA.synchronize()
    end
    nl_ts = Float64[]
    for _ in 1:nrun
        A = copy(A0); R = CUDA.zeros(T,n,n); tau = CUDA.zeros(T,n)
        CUDA.synchronize()
        t = time_ns(); NextLA.geqrf_2p5d!(m, n, A, R, tau; params=p); CUDA.synchronize()
        push!(nl_ts, (time_ns() - t) / 1e6)
    end
    sort!(nl_ts)
    return (cu_min=cu_ts[1], nl_min=nl_ts[1], b_used=p.b, c=p.c)
end

flushln("Extended b-sweep on H200 FP64 (c=1 forced):\n")
for N in (4000, 8000)
    flushln("─ N=$N ─────")
    @printf("%4s %5s  %8s %8s %7s  %6s\n", "b", "b*", "cu(ms)", "nl(ms)", "ratio", "c")
    for b in (256, 384, 512, 640, 727, 1024)
        # request b explicitly; compute_params will clamp if needed
        r = bench_pair(Float64, N, N; b=b)
        ratio = r.cu_min / r.nl_min
        @printf("%4d %5d  %8.2f %8.2f %6.2fx %5d\n", b, r.b_used, r.cu_min, r.nl_min, ratio, r.c)
    end
    flushln()
end

# Also try with a HUGE M so c>1 at N=8000, allowing b > b_max(c=1)
flushln("\n--- HUGE M (forces c>1 at N=8000) ---")
ENV["NEXTLA_M_BYTES"] = "200000000"  # 200 MB → ~25M FP64 words. PM=132*25e6 = 3.3e9
ENV["NEXTLA_FORCE_C1"] = "1"  # keep fanout suppressed
for N in (8000,), b in (512, 727, 1024, 1500)
    r = bench_pair(Float64, N, N; b=b, c=nothing)
    @printf("N=%d b=%d  cu=%.2f ms  nl=%.2f ms  ratio=%.2fx  (b_used=%d c=%d)\n",
            N, b, r.cu_min, r.nl_min, r.cu_min/r.nl_min, r.b_used, r.c)
end
