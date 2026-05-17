#!/usr/bin/env julia
# Final 4-variant H200 benchmark — flat, no closures (closure boxing tanks
# Julia perf on this hot path).
#
#   1. sCQR3 (3-pass FP64)            — DAAP-proper panel
#   2. Householder-2.5D               — Quasi-DAAP panel (cuSOLVER geqrf)
#   3. Mixed-prec sCQR3 (FP32 trail)
#   4. Mixed-prec CQR2 (FP32 trail)
#
# vs CUDA.CUSOLVER.geqrf!.

using LinearAlgebra, Random, Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "NextLA.jl"))
using NextLA, KernelAbstractions, CUDA

ENV["NEXTLA_FORCE_C1"] = "1"
ENV["NEXTLA_USE_GRAPH"] = "0"
CUDA.math_mode!(CUDA.DEFAULT_MATH)

flushln(s...) = (println(s...); flush(stdout))
flushln("==========================================================")
flushln(" H200 4-variant NextLA benchmark vs cuSOLVER.geqrf!")
flushln(" host=", gethostname(), "  gpu=", CUDA.name(CUDA.device()))
flushln("==========================================================")

const RESULTS = Vector{NamedTuple}()

for N in (4000, 8000, 16000)
    flushln("\n────── N=$N ──────")
    A0 = CUDA.randn(Float64, N, N)

    # cuSOLVER
    for _ in 1:2; A = copy(A0); CUDA.CUSOLVER.geqrf!(A); CUDA.synchronize(); end
    cu_ts = Float64[]
    for _ in 1:5
        A = copy(A0); CUDA.synchronize()
        t = time_ns(); CUDA.CUSOLVER.geqrf!(A); CUDA.synchronize()
        push!(cu_ts, (time_ns()-t)/1e6)
    end
    sort!(cu_ts)
    cu_tmin = cu_ts[1]
    @printf("  cuSOLVER          tmin=%8.2f ms  (baseline)\n", cu_tmin); flush(stdout)

    # 1. sCQR3
    CUDA.math_mode!(CUDA.DEFAULT_MATH)
    for _ in 1:2
        A = copy(A0); R = CUDA.zeros(Float64,N,N); tau = CUDA.zeros(Float64,N)
        NextLA.geqrf_2p5d!(N,N,A,R,tau)
        CUDA.synchronize()
    end
    Av = copy(A0); Rv = CUDA.zeros(Float64,N,N); tauv = CUDA.zeros(Float64,N)
    NextLA.geqrf_2p5d!(N,N,Av,Rv,tauv); CUDA.synchronize()
    res1 = norm(A0 - Av*Rv)/norm(A0); ort1 = norm(Av'*Av - I)
    ts = Float64[]
    for _ in 1:5
        A = copy(A0); R = CUDA.zeros(Float64,N,N); tau = CUDA.zeros(Float64,N); CUDA.synchronize()
        t = time_ns(); NextLA.geqrf_2p5d!(N,N,A,R,tau); CUDA.synchronize()
        push!(ts, (time_ns()-t)/1e6)
    end
    sort!(ts); t1 = ts[1]
    @printf("  1. sCQR3          tmin=%8.2f ms  %.2fx  res=%.1e ortho=%.1e\n", t1, cu_tmin/t1, res1, ort1); flush(stdout)

    # 2. Householder
    for _ in 1:2
        A = copy(A0); R = CUDA.zeros(Float64,N,N); tau = CUDA.zeros(Float64,N)
        NextLA.geqrf_2p5d_householder!(N,N,A,R,tau)
        CUDA.synchronize()
    end
    Av = copy(A0); Rv = CUDA.zeros(Float64,N,N); tauv = CUDA.zeros(Float64,N)
    NextLA.geqrf_2p5d_householder!(N,N,Av,Rv,tauv); CUDA.synchronize()
    res2 = norm(A0 - Av*Rv)/norm(A0); ort2 = norm(Av'*Av - I)
    ts = Float64[]
    for _ in 1:5
        A = copy(A0); R = CUDA.zeros(Float64,N,N); tau = CUDA.zeros(Float64,N); CUDA.synchronize()
        t = time_ns(); NextLA.geqrf_2p5d_householder!(N,N,A,R,tau); CUDA.synchronize()
        push!(ts, (time_ns()-t)/1e6)
    end
    sort!(ts); t2 = ts[1]
    @printf("  2. Householder    tmin=%8.2f ms  %.2fx  res=%.1e ortho=%.1e\n", t2, cu_tmin/t2, res2, ort2); flush(stdout)

    # 3. Mixed-precision sCQR3 (TF32 off)
    CUDA.math_mode!(CUDA.DEFAULT_MATH)
    for _ in 1:2
        A = copy(A0); R = CUDA.zeros(Float64,N,N); tau = CUDA.zeros(Float64,N)
        NextLA.geqrf_2p5d!(N,N,A,R,tau; mixed_precision=true)
        CUDA.synchronize()
    end
    Av = copy(A0); Rv = CUDA.zeros(Float64,N,N); tauv = CUDA.zeros(Float64,N)
    NextLA.geqrf_2p5d!(N,N,Av,Rv,tauv; mixed_precision=true); CUDA.synchronize()
    res3 = norm(A0 - Av*Rv)/norm(A0); ort3 = norm(Av'*Av - I)
    ts = Float64[]
    for _ in 1:5
        A = copy(A0); R = CUDA.zeros(Float64,N,N); tau = CUDA.zeros(Float64,N); CUDA.synchronize()
        t = time_ns(); NextLA.geqrf_2p5d!(N,N,A,R,tau; mixed_precision=true); CUDA.synchronize()
        push!(ts, (time_ns()-t)/1e6)
    end
    sort!(ts); t3 = ts[1]
    @printf("  3. MP sCQR3       tmin=%8.2f ms  %.2fx  res=%.1e ortho=%.1e\n", t3, cu_tmin/t3, res3, ort3); flush(stdout)

    # 3b. Mixed-precision sCQR3 + TF32 on
    CUDA.math_mode!(CUDA.FAST_MATH)
    for _ in 1:2
        A = copy(A0); R = CUDA.zeros(Float64,N,N); tau = CUDA.zeros(Float64,N)
        NextLA.geqrf_2p5d!(N,N,A,R,tau; mixed_precision=true)
        CUDA.synchronize()
    end
    Av = copy(A0); Rv = CUDA.zeros(Float64,N,N); tauv = CUDA.zeros(Float64,N)
    NextLA.geqrf_2p5d!(N,N,Av,Rv,tauv; mixed_precision=true); CUDA.synchronize()
    res3b = norm(A0 - Av*Rv)/norm(A0); ort3b = norm(Av'*Av - I)
    ts = Float64[]
    for _ in 1:5
        A = copy(A0); R = CUDA.zeros(Float64,N,N); tau = CUDA.zeros(Float64,N); CUDA.synchronize()
        t = time_ns(); NextLA.geqrf_2p5d!(N,N,A,R,tau; mixed_precision=true); CUDA.synchronize()
        push!(ts, (time_ns()-t)/1e6)
    end
    sort!(ts); t3b = ts[1]
    @printf("  3b. MP sCQR3+TF32 tmin=%8.2f ms  %.2fx  res=%.1e ortho=%.1e\n", t3b, cu_tmin/t3b, res3b, ort3b); flush(stdout)
    CUDA.math_mode!(CUDA.DEFAULT_MATH)

    # 4. Mixed-precision CQR2 (TF32 off)
    for _ in 1:2
        A = copy(A0); R = CUDA.zeros(Float64,N,N); tau = CUDA.zeros(Float64,N)
        NextLA.geqrf_2p5d!(N,N,A,R,tau; passes=2, mixed_precision=true)
        CUDA.synchronize()
    end
    Av = copy(A0); Rv = CUDA.zeros(Float64,N,N); tauv = CUDA.zeros(Float64,N)
    NextLA.geqrf_2p5d!(N,N,Av,Rv,tauv; passes=2, mixed_precision=true); CUDA.synchronize()
    res4 = norm(A0 - Av*Rv)/norm(A0); ort4 = norm(Av'*Av - I)
    ts = Float64[]
    for _ in 1:5
        A = copy(A0); R = CUDA.zeros(Float64,N,N); tau = CUDA.zeros(Float64,N); CUDA.synchronize()
        t = time_ns(); NextLA.geqrf_2p5d!(N,N,A,R,tau; passes=2, mixed_precision=true); CUDA.synchronize()
        push!(ts, (time_ns()-t)/1e6)
    end
    sort!(ts); t4 = ts[1]
    @printf("  4. MP CQR2        tmin=%8.2f ms  %.2fx  res=%.1e ortho=%.1e\n", t4, cu_tmin/t4, res4, ort4); flush(stdout)

    # 4b. Mixed-precision CQR2 + TF32 on
    CUDA.math_mode!(CUDA.FAST_MATH)
    for _ in 1:2
        A = copy(A0); R = CUDA.zeros(Float64,N,N); tau = CUDA.zeros(Float64,N)
        NextLA.geqrf_2p5d!(N,N,A,R,tau; passes=2, mixed_precision=true)
        CUDA.synchronize()
    end
    Av = copy(A0); Rv = CUDA.zeros(Float64,N,N); tauv = CUDA.zeros(Float64,N)
    NextLA.geqrf_2p5d!(N,N,Av,Rv,tauv; passes=2, mixed_precision=true); CUDA.synchronize()
    res4b = norm(A0 - Av*Rv)/norm(A0); ort4b = norm(Av'*Av - I)
    ts = Float64[]
    for _ in 1:5
        A = copy(A0); R = CUDA.zeros(Float64,N,N); tau = CUDA.zeros(Float64,N); CUDA.synchronize()
        t = time_ns(); NextLA.geqrf_2p5d!(N,N,A,R,tau; passes=2, mixed_precision=true); CUDA.synchronize()
        push!(ts, (time_ns()-t)/1e6)
    end
    sort!(ts); t4b = ts[1]
    @printf("  4b. MP CQR2+TF32  tmin=%8.2f ms  %.2fx  res=%.1e ortho=%.1e\n", t4b, cu_tmin/t4b, res4b, ort4b); flush(stdout)
    CUDA.math_mode!(CUDA.DEFAULT_MATH)

    push!(RESULTS, (N=N, cu=cu_tmin, t1=t1, t2=t2, t3=t3, t3b=t3b, t4=t4, t4b=t4b,
                    res=[res1, res2, res3, res3b, res4, res4b],
                    ortho=[ort1, ort2, ort3, ort3b, ort4, ort4b]))
end

flushln("\n", "="^88)
flushln(" Summary — speedup vs cuSOLVER (>1 means NextLA wins; bold-relevant)")
flushln("="^88)
@printf("%5s %9s %8s %8s %8s %8s %8s %8s\n",
        "N", "cu(ms)", "sCQR3", "Hldr", "MP-s", "MP-s+TF", "MP-CQR2", "MP-C2+TF")
println("-"^88)
for r in RESULTS
    @printf("%5d %9.2f %7.2fx %7.2fx %7.2fx %7.2fx %7.2fx %7.2fx\n",
            r.N, r.cu, r.cu/r.t1, r.cu/r.t2, r.cu/r.t3, r.cu/r.t3b, r.cu/r.t4, r.cu/r.t4b)
end
println("="^88)
