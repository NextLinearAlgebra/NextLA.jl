#!/usr/bin/env julia
# Single-GPU H200 benchmark for all 6 NextLA variants vs cuSOLVER.geqrf!.
#
#   1. sCQR3 (3-pass FP64)                — DAAP-proper panel  (Path s)
#   2. Householder-2.5D                    — Quasi-DAAP panel  (Path h, cuSOLVER geqrf)
#   3. MP sCQR3 (FP32 trailing GEMM)
#   4. MP CQR2 (2-pass + FP32 trailing)
#   5. Look-ahead 2-stream sCQR3          — Phase Q5 (paper §A.1)
#   6. Mixed-prec sCQR3 + FP64 IR n_ir=1   — Phase Q6 (paper §A.1)
#
# Tested at N ∈ {4K, 8K, 16K, 30K} with both TF32-off and TF32-on for the
# variants that use mixed precision.

using LinearAlgebra, Random, Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "NextLA.jl"))
using NextLA, KernelAbstractions, CUDA

ENV["NEXTLA_FORCE_C1"] = "1"
ENV["NEXTLA_USE_GRAPH"] = "0"
CUDA.math_mode!(CUDA.DEFAULT_MATH)

flushln(s...) = (println(s...); flush(stdout))
flushln("==========================================================")
flushln(" H200 6-variant NextLA benchmark vs cuSOLVER.geqrf!")
flushln(" host=", gethostname(), "  gpu=", CUDA.name(CUDA.device()))
flushln("==========================================================")

const RESULTS = Vector{NamedTuple}()
sizes = (4000, 8000, 16000, 30000)

for N in sizes
    flushln("\n────── N=$N ──────")
    A0 = CUDA.randn(Float64, N, N)

    # cuSOLVER reference.
    for _ in 1:2; A = copy(A0); CUDA.CUSOLVER.geqrf!(A); CUDA.synchronize(); end
    cu_ts = Float64[]
    for _ in 1:5
        A = copy(A0); CUDA.synchronize()
        t = time_ns(); CUDA.CUSOLVER.geqrf!(A); CUDA.synchronize()
        push!(cu_ts, (time_ns()-t)/1e6)
    end
    sort!(cu_ts); cu_tmin = cu_ts[1]
    @printf("  cuSOLVER             tmin=%9.2f ms  (baseline)\n", cu_tmin); flush(stdout)

    # 1. sCQR3
    CUDA.math_mode!(CUDA.DEFAULT_MATH)
    for _ in 1:2; A=copy(A0); R=CUDA.zeros(Float64,N,N); tau=CUDA.zeros(Float64,N); NextLA.geqrf_2p5d!(N,N,A,R,tau); CUDA.synchronize(); end
    Av=copy(A0); Rv=CUDA.zeros(Float64,N,N); tauv=CUDA.zeros(Float64,N); NextLA.geqrf_2p5d!(N,N,Av,Rv,tauv); CUDA.synchronize()
    res1 = norm(A0-Av*Rv)/norm(A0); ort1 = norm(Av'*Av-I)
    ts=Float64[]; for _ in 1:5; A=copy(A0); R=CUDA.zeros(Float64,N,N); tau=CUDA.zeros(Float64,N); CUDA.synchronize(); t=time_ns(); NextLA.geqrf_2p5d!(N,N,A,R,tau); CUDA.synchronize(); push!(ts,(time_ns()-t)/1e6); end; sort!(ts); t1=ts[1]
    @printf("  1. sCQR3             tmin=%9.2f ms  %.2fx  res=%.1e ortho=%.1e\n", t1, cu_tmin/t1, res1, ort1); flush(stdout)

    # 2. Householder
    for _ in 1:2; A=copy(A0); R=CUDA.zeros(Float64,N,N); tau=CUDA.zeros(Float64,N); NextLA.geqrf_2p5d_householder!(N,N,A,R,tau); CUDA.synchronize(); end
    Av=copy(A0); Rv=CUDA.zeros(Float64,N,N); tauv=CUDA.zeros(Float64,N); NextLA.geqrf_2p5d_householder!(N,N,Av,Rv,tauv); CUDA.synchronize()
    res2 = norm(A0-Av*Rv)/norm(A0); ort2 = norm(Av'*Av-I)
    ts=Float64[]; for _ in 1:5; A=copy(A0); R=CUDA.zeros(Float64,N,N); tau=CUDA.zeros(Float64,N); CUDA.synchronize(); t=time_ns(); NextLA.geqrf_2p5d_householder!(N,N,A,R,tau); CUDA.synchronize(); push!(ts,(time_ns()-t)/1e6); end; sort!(ts); t2=ts[1]
    @printf("  2. Householder       tmin=%9.2f ms  %.2fx  res=%.1e ortho=%.1e\n", t2, cu_tmin/t2, res2, ort2); flush(stdout)

    # 3. MP sCQR3 (TF32 off)
    CUDA.math_mode!(CUDA.DEFAULT_MATH)
    for _ in 1:2; A=copy(A0); R=CUDA.zeros(Float64,N,N); tau=CUDA.zeros(Float64,N); NextLA.geqrf_2p5d!(N,N,A,R,tau; mixed_precision=true); CUDA.synchronize(); end
    Av=copy(A0); Rv=CUDA.zeros(Float64,N,N); tauv=CUDA.zeros(Float64,N); NextLA.geqrf_2p5d!(N,N,Av,Rv,tauv; mixed_precision=true); CUDA.synchronize()
    res3 = norm(A0-Av*Rv)/norm(A0); ort3 = norm(Av'*Av-I)
    ts=Float64[]; for _ in 1:5; A=copy(A0); R=CUDA.zeros(Float64,N,N); tau=CUDA.zeros(Float64,N); CUDA.synchronize(); t=time_ns(); NextLA.geqrf_2p5d!(N,N,A,R,tau; mixed_precision=true); CUDA.synchronize(); push!(ts,(time_ns()-t)/1e6); end; sort!(ts); t3=ts[1]
    @printf("  3. MP sCQR3 (TF32-)  tmin=%9.2f ms  %.2fx  res=%.1e ortho=%.1e\n", t3, cu_tmin/t3, res3, ort3); flush(stdout)

    # 4. MP CQR2 (TF32 off)
    for _ in 1:2; A=copy(A0); R=CUDA.zeros(Float64,N,N); tau=CUDA.zeros(Float64,N); NextLA.geqrf_2p5d!(N,N,A,R,tau; passes=2, mixed_precision=true); CUDA.synchronize(); end
    Av=copy(A0); Rv=CUDA.zeros(Float64,N,N); tauv=CUDA.zeros(Float64,N); NextLA.geqrf_2p5d!(N,N,Av,Rv,tauv; passes=2, mixed_precision=true); CUDA.synchronize()
    res4 = norm(A0-Av*Rv)/norm(A0); ort4 = norm(Av'*Av-I)
    ts=Float64[]; for _ in 1:5; A=copy(A0); R=CUDA.zeros(Float64,N,N); tau=CUDA.zeros(Float64,N); CUDA.synchronize(); t=time_ns(); NextLA.geqrf_2p5d!(N,N,A,R,tau; passes=2, mixed_precision=true); CUDA.synchronize(); push!(ts,(time_ns()-t)/1e6); end; sort!(ts); t4=ts[1]
    @printf("  4. MP CQR2 (TF32-)   tmin=%9.2f ms  %.2fx  res=%.1e ortho=%.1e\n", t4, cu_tmin/t4, res4, ort4); flush(stdout)

    # 5. Look-ahead sCQR3 (2 streams)
    for _ in 1:2; A=copy(A0); R=CUDA.zeros(Float64,N,N); tau=CUDA.zeros(Float64,N); NextLA.geqrf_2p5d_lookahead!(N,N,A,R,tau); CUDA.synchronize(); end
    Av=copy(A0); Rv=CUDA.zeros(Float64,N,N); tauv=CUDA.zeros(Float64,N); NextLA.geqrf_2p5d_lookahead!(N,N,Av,Rv,tauv); CUDA.synchronize()
    res5 = norm(A0-Av*Rv)/norm(A0); ort5 = norm(Av'*Av-I)
    ts=Float64[]; for _ in 1:5; A=copy(A0); R=CUDA.zeros(Float64,N,N); tau=CUDA.zeros(Float64,N); CUDA.synchronize(); t=time_ns(); NextLA.geqrf_2p5d_lookahead!(N,N,A,R,tau); CUDA.synchronize(); push!(ts,(time_ns()-t)/1e6); end; sort!(ts); t5=ts[1]
    @printf("  5. Look-ahead-2-str  tmin=%9.2f ms  %.2fx  res=%.1e ortho=%.1e\n", t5, cu_tmin/t5, res5, ort5); flush(stdout)

    # 6. IR (MP sCQR3 + n_ir=1 FP64 refinement)
    for _ in 1:2; A=copy(A0); R=CUDA.zeros(Float64,N,N); tau=CUDA.zeros(Float64,N); A0_kept=copy(A0); NextLA.geqrf_2p5d_ir!(N,N,A,R,tau,A0_kept; n_ir=1); CUDA.synchronize(); end
    Av=copy(A0); Rv=CUDA.zeros(Float64,N,N); tauv=CUDA.zeros(Float64,N); A0_kept=copy(A0); NextLA.geqrf_2p5d_ir!(N,N,Av,Rv,tauv,A0_kept; n_ir=1); CUDA.synchronize()
    res6 = norm(A0-Av*Rv)/norm(A0); ort6 = norm(Av'*Av-I)
    ts=Float64[]; for _ in 1:5; A=copy(A0); R=CUDA.zeros(Float64,N,N); tau=CUDA.zeros(Float64,N); A0_kept=copy(A0); CUDA.synchronize(); t=time_ns(); NextLA.geqrf_2p5d_ir!(N,N,A,R,tau,A0_kept; n_ir=1); CUDA.synchronize(); push!(ts,(time_ns()-t)/1e6); end; sort!(ts); t6=ts[1]
    @printf("  6. MP sCQR3 + IR     tmin=%9.2f ms  %.2fx  res=%.1e ortho=%.1e\n", t6, cu_tmin/t6, res6, ort6); flush(stdout)

    push!(RESULTS, (N=N, cu=cu_tmin, t1=t1, t2=t2, t3=t3, t4=t4, t5=t5, t6=t6,
                    res=[res1,res2,res3,res4,res5,res6], ortho=[ort1,ort2,ort3,ort4,ort5,ort6]))
end

flushln("\n", "="^88)
flushln(" Summary — speedup vs cuSOLVER (>1 = NextLA wins)")
flushln("="^88)
@printf("%6s %10s %8s %8s %8s %8s %8s %8s\n",
        "N", "cu(ms)", "sCQR3", "Hldr", "MP-s", "MP-C2", "LA-2-s", "IR-1")
println("-"^88)
for r in RESULTS
    @printf("%6d %10.2f %7.2fx %7.2fx %7.2fx %7.2fx %7.2fx %7.2fx\n",
            r.N, r.cu, r.cu/r.t1, r.cu/r.t2, r.cu/r.t3, r.cu/r.t4, r.cu/r.t5, r.cu/r.t6)
end
println("="^88)

# Print residual / ortho gates so the reader can see which variants are
# numerically stable.
flushln("\nNumerical stability (residual & orthogonality at N=", sizes[end], "):")
last = RESULTS[end]
labels = ["sCQR3", "Householder", "MP-sCQR3", "MP-CQR2", "Look-ahead", "IR-1"]
for (i, lbl) in enumerate(labels)
    @printf("  %-12s  res=%.2e   |QtQ - I|=%.2e\n", lbl, last.res[i], last.ortho[i])
end
