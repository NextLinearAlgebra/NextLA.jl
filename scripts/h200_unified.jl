#!/usr/bin/env julia
# Single-process H200 bench: amortizes JIT compile across all configs.
#
# Loads NextLA + CUDA + cuSOLVER once, runs through every (config, T, N, b)
# cell, prints a CSV-friendly summary at the end. Configs vary by env knobs
# that are checked at runtime inside NextLA (NEXTLA_FORCE_C1, NEXTLA_USE_SYRK,
# NEXTLA_M_BYTES). NEXTLA_TF32 is special — its toggle has to happen before
# the cudaext __init__ runs; we set it via `CUDA.math_mode!` directly inside
# each config block.

using LinearAlgebra, Random, Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "NextLA.jl"))
using NextLA, KernelAbstractions, CUDA

flushln(s...) = (println(s...); flush(stdout))

flushln("==========================================================")
flushln(" H200 unified NextLA bench   ", string(time()))
flushln("  host=", gethostname(), "  gpu=", CUDA.name(CUDA.device()),
        "  CUDA functional=", CUDA.functional())
flushln("==========================================================")

# --- Probe / Phase A --------------------------------------------------------
flushln("\n### Phase A — regime probe ###")
let be = CUDABackend()
    for N in (1000, 2000, 4000, 8000), T in (Float64, Float32)
        p = compute_params(be, T, N)
        flushln("  T=", T, " N=", N, "  c=", p.c, " b=", p.b,
                " M=", p.M, " words  P=", p.P, "  Px=", p.Px)
    end
end

# --- Bench primitives ------------------------------------------------------
function make_A(::Type{T}, m, n; cnd=10.0, seed=7) where {T}
    rng = MersenneTwister(seed)
    k = min(m, n)
    sv = T[T(cnd)^(-(i-1)/(k-1)) for i in 1:k]
    U, _ = qr(randn(rng, T, m, m)); U = Matrix(U)
    V, _ = qr(randn(rng, T, n, n)); V = Matrix(V)
    return CuArray(T.(U[:, 1:k] * Diagonal(sv) * V[:, 1:k]'))
end

function time_cusolver(::Type{T}, m, n; nwarm=2, nrun=5) where {T}
    A0 = make_A(T, m, n)
    for _ in 1:nwarm
        A = copy(A0); CUDA.CUSOLVER.geqrf!(A); CUDA.synchronize()
    end
    ts = Float64[]
    for _ in 1:nrun
        A = copy(A0); CUDA.synchronize()
        t = time_ns()
        CUDA.CUSOLVER.geqrf!(A); CUDA.synchronize()
        push!(ts, (time_ns() - t) / 1e6)
    end
    sort!(ts); ts[1], ts[nrun÷2+1]
end

function time_nextla(::Type{T}, m, n; b, c=1, nwarm=2, nrun=5, ortho=:fast) where {T}
    A0 = make_A(T, m, n)
    be = CUDABackend()
    p = compute_params(be, T, n; b=b, c=c)
    for _ in 1:nwarm
        A = copy(A0); R = CUDA.zeros(T,n,n); tau = CUDA.zeros(T,n)
        NextLA.geqrf_2p5d!(m, n, A, R, tau; params=p, ortho=ortho)
        CUDA.synchronize()
    end
    ts = Float64[]
    A_keep = copy(A0); R_keep = CUDA.zeros(T,n,n); tau_keep = CUDA.zeros(T,n)
    NextLA.geqrf_2p5d!(m, n, A_keep, R_keep, tau_keep; params=p, ortho=ortho)
    CUDA.synchronize()
    res = norm(A0 - A_keep * R_keep) / norm(A0)
    ort = norm(A_keep' * A_keep - I)
    for _ in 1:nrun
        A = copy(A0); R = CUDA.zeros(T,n,n); tau = CUDA.zeros(T,n)
        CUDA.synchronize()
        t = time_ns()
        NextLA.geqrf_2p5d!(m, n, A, R, tau; params=p, ortho=ortho)
        CUDA.synchronize()
        push!(ts, (time_ns() - t) / 1e6)
    end
    sort!(ts)
    return (b_used=p.b, c=p.c, tmin=ts[1], tmed=ts[nrun÷2+1], res=res, orth=ort)
end

# --- Configurations ---------------------------------------------------------
sizes = (2000, 4000, 8000)
bs = (128, 256, 512)
results = NamedTuple[]

function run_config(label, types, ; c=1, tf32=:default, use_syrk=true,
                    m_bytes=nothing, force_c1=false)
    # Knobs that NextLA reads at every call: ENV
    ENV["NEXTLA_USE_SYRK"] = use_syrk ? "1" : "0"
    ENV["NEXTLA_FORCE_C1"] = force_c1 ? "1" : "0"
    if m_bytes === nothing
        delete!(ENV, "NEXTLA_M_BYTES")
    else
        ENV["NEXTLA_M_BYTES"] = string(m_bytes)
    end
    # TF32 is a CUDA-task-scope flag.
    if tf32 === :on
        CUDA.math_mode!(CUDA.FAST_MATH)
    else
        CUDA.math_mode!(CUDA.DEFAULT_MATH)
    end
    flushln("\n", "─"^66)
    flushln(" Config: ", label)
    flushln(" knobs: ortho=:fast c=$c tf32=$tf32 use_syrk=$use_syrk m_bytes=$m_bytes force_c1=$force_c1")
    flushln("─"^66)
    for T in types
        for N in sizes
            cu_min, cu_med = time_cusolver(T, N, N)
            for b in bs
                if b > N; continue; end
                r = time_nextla(T, N, N; b=b, c=c)
                speedup = cu_min / r.tmin
                @printf("  %-10s N=%-5d b=%-3d   cu=%6.2f ms  nl=%6.2f ms  %4.2fx  c=%d  res=%.1e ortho=%.1e\n",
                        T, N, b, cu_min, r.tmin, speedup, r.c, r.res, r.orth)
                flush(stdout)
                push!(results, (label=label, T=T, N=N, b=b,
                                cu_min=cu_min, cu_med=cu_med,
                                nl_min=r.tmin, nl_med=r.tmed,
                                speedup=speedup, c=r.c, res=r.res, ortho=r.orth))
            end
        end
    end
end

# 1. Default (post-Steps 1-6 integrated). FP64 only first.
run_config("Default (M=smem+regs, SYRK, FORCE_C1=0, TF32=off)", (Float64,);
           c=1, tf32=:default, use_syrk=true, m_bytes=nothing, force_c1=false)

# 2. Same default but with c=auto so c>1 fanout active where PM>N²
run_config("Default + c=auto (fanout active at N=1000/2000)", (Float64,);
           c=nothing, tf32=:default, use_syrk=true, m_bytes=nothing, force_c1=false)

# 3. SYRK OFF — A/B for Step 4
run_config("Step 4 baseline (SYRK off — uses cuBLAS GEMM)", (Float64,);
           c=1, tf32=:default, use_syrk=false, m_bytes=nothing, force_c1=true)

# 4. OLD smem-only M — A/B for Step 3
run_config("Step 3 baseline (M = smem only, no register file)", (Float64,);
           c=1, tf32=:default, use_syrk=true, m_bytes=233472, force_c1=true)

# 5. FP32 TF32 off
run_config("FP32 TF32 off (FP32 ALU)", (Float32,);
           c=1, tf32=:off, use_syrk=true, m_bytes=nothing, force_c1=true)

# 6. FP32 TF32 on
run_config("FP32 TF32 on (Tensor Cores)", (Float32,);
           c=1, tf32=:on, use_syrk=true, m_bytes=nothing, force_c1=true)

# Phase D large-N safe ortho check
flushln("\n### Phase D — N=4096 :safe orthogonality gate ###")
let N = 4096, T = Float64
    A0 = CUDA.randn(T, N, N)
    A  = copy(A0); R = CUDA.zeros(T, N, N); tau = CUDA.zeros(T, N)
    NextLA.geqrf_2p5d!(N, N, A, R, tau; ortho=:safe)
    CUDA.synchronize()
    flushln("  N=", N, " FP64 :safe  |Q'Q - I|_F = ", norm(A' * A - I),
            "  residual = ", norm(A0 - A * R) / norm(A0))
end

# --- Summary table ---------------------------------------------------------
flushln("\n", "="^90)
flushln(" Summary (best b per Config × T × N):")
flushln("="^90)
@printf("%-58s %-9s %-6s %8s %8s %7s %3s\n", "Config", "T", "N", "cu(ms)", "nl(ms)", "speedup", "c")
println("-"^90)
groups = Dict()
for r in results
    push!(get!(groups, (r.label, r.T, r.N), typeof(r)[]), r)
end
for ((label, T, N), rs) in sort(collect(groups); by=x->(x[1][1], string(x[1][2]), x[1][3]))
    best = rs[argmax(getfield.(rs, :speedup))]
    @printf("%-58s %-9s %-6d %8.2f %8.2f %6.2fx %3d\n",
            label[1:min(end, 58)], T, N, best.cu_min, best.nl_min, best.speedup, best.c)
end
println("="^90)
flushln(" Done at ", string(time()))
