#!/usr/bin/env julia
# bench_gpu_vs_cusolver.jl
# Compare NextLA.geqrf_2p5d! vs CUDA.CUSOLVER.geqrf! on a 1000×1000 matrix.
# Sweeps b across the paper's admissible window [c, N/√P₁] for both Float32 and Float64.
#
# Run: julia --project=NextLA.jl scripts/bench_gpu_vs_cusolver.jl

using LinearAlgebra, Random, Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "NextLA.jl"))
using NextLA, KernelAbstractions, CUDA

function make_cu_matrix(::Type{T}, m, n; cnd=10.0, seed=42) where {T}
    rng = MersenneTwister(seed)
    k = min(m, n)
    sv = T[T(cnd)^(-(i-1)/(k-1)) for i in 1:k]
    U, _ = qr(randn(rng, T, m, m)); U = Matrix(U)
    V, _ = qr(randn(rng, T, n, n)); V = Matrix(V)
    A_cpu = U[:,1:k] * Diagonal(sv) * V[:,1:k]'
    return CuArray(T.(A_cpu))
end

function fukaya_metrics_gpu(A0::CuArray{T}, A_fact::CuArray{T}, R::CuArray{T}) where {T}
    n = size(A_fact, 2)
    Q = A_fact[:, 1:n]
    G = Q' * Q - I
    denom = max(norm(A0), eps(real(T)))
    res = norm(A0 - Q * R) / denom
    orth = norm(G)
    return (; res, orth)
end

function _bench_c_override()
    s = get(ENV, "BENCH_C", "1")
    s == "auto" && return nothing
    return parse(Int, s)
end

function bench_nextla(::Type{T}, m, n, b; ortho::Symbol=:fast, nwarm=2, nrun=10) where {T}
    A0 = make_cu_matrix(T, m, n; cnd=10.0, seed=7)
    be = CUDABackend()
    c_ovr = _bench_c_override()

    for _ in 1:nwarm
        A = copy(A0); R = CUDA.zeros(T, n, n); tau = CUDA.zeros(T, n)
        p = compute_params(be, T, n; b=b, c=c_ovr)
        NextLA.geqrf_2p5d!(m, n, A, R, tau; params=p, ortho=ortho)
        CUDA.synchronize()
    end

    A = copy(A0); R = CUDA.zeros(T, n, n); tau = CUDA.zeros(T, n)
    p = compute_params(be, T, n; b=b, c=c_ovr)
    NextLA.geqrf_2p5d!(m, n, A, R, tau; params=p, ortho=ortho)
    CUDA.synchronize()
    met = fukaya_metrics_gpu(A0, A, R)

    times = Float64[]
    for _ in 1:nrun
        A2 = copy(A0); R2 = CUDA.zeros(T, n, n); tau2 = CUDA.zeros(T, n)
        CUDA.synchronize()
        t0 = time_ns()
        NextLA.geqrf_2p5d!(m, n, A2, R2, tau2; params=p, ortho=ortho)
        CUDA.synchronize()
        push!(times, (time_ns() - t0) / 1e6)
    end
    sort!(times)
    return (; b_used=p.b, c=p.c, Px=p.Px,
              tmin=times[1], tmed=times[nrun÷2+1], tmax=times[end],
              res=met.res, orth=met.orth)
end

function bench_cusolver(::Type{T}, m, n; nwarm=2, nrun=10) where {T}
    A0 = make_cu_matrix(T, m, n; cnd=10.0, seed=7)
    for _ in 1:nwarm
        A = copy(A0); CUDA.CUSOLVER.geqrf!(A); CUDA.synchronize()
    end
    times = Float64[]
    for _ in 1:nrun
        A2 = copy(A0); CUDA.synchronize()
        t0 = time_ns()
        CUDA.CUSOLVER.geqrf!(A2)
        CUDA.synchronize()
        push!(times, (time_ns() - t0) / 1e6)
    end
    sort!(times)
    return (; tmin=times[1], tmed=times[nrun÷2+1], tmax=times[end])
end

function fp_peak_gflops(::Type{Float64})
    # NEXTLA_PEAK_FP64 in GF/s; default tuned for NVIDIA H200 (FP64 ALU ≈ 34 TFLOPS).
    # H200 FP64 Tensor Core peak is ~67 TFLOPS; set to 67000 if comparing against TC.
    return parse(Float64, get(ENV, "NEXTLA_PEAK_FP64", "34000"))
end
function fp_peak_gflops(::Type{Float32})
    # H200 FP32 ALU ≈ 67 TFLOPS; with TF32 Tensor Cores ≈ 989 TFLOPS.
    return parse(Float64, get(ENV, "NEXTLA_PEAK_FP32", "67000"))
end

function run_for_size(m::Int, n::Int)
    println("═══════════════════════════════════════════════════════")
    @printf(" geqrf_2p5d! vs CUSOLVER.geqrf!   m=%d, n=%d\n", m, n)
    println("═══════════════════════════════════════════════════════")

    type_list = if get(ENV, "BENCH_T", "") != ""
                    Tuple(s == "f64" ? Float64 : s == "f32" ? Float32 :
                          s == "c64" ? ComplexF64 : ComplexF32 for s in split(ENV["BENCH_T"], ","))
                else (Float64, Float32) end
    for T in type_list
        println("── Type: $T ─────────────────────────────────────────")
        peak = fp_peak_gflops(T)
        flops = 4.0 * n^3 / 3.0
        cu = bench_cusolver(T, m, n)
        cu_gflops = flops / (cu.tmin * 1e6)
        @printf("  cuSOLVER.geqrf!       min=%7.2f ms  med=%7.2f ms   (%.0f GF/s, %.1f%% peak)\n",
                cu.tmin, cu.tmed, cu_gflops, 100*cu_gflops/peak)

        # Admissible window from the paper: c ≤ b ≤ N/√P. Build the b sweep adaptive to N.
        # Extended to b=1024 for large N on H200 where the latency-optimal b★ ≈ √(γ P log P / β)
        # is much larger than the X-partition cube √M.
        sweep_default = n <= 1024  ? (32, 64, 96, 128, 160, 192, 224, 250) :
                        n <= 2048  ? (64, 96, 128, 160, 192, 224, 256, 320, 384) :
                        n <= 4096  ? (64, 128, 192, 256, 320, 384, 512, 640) :
                        n <= 8192  ? (128, 192, 256, 384, 512, 640, 768, 896, 1024) :
                                     (128, 192, 256, 384, 512, 768, 1024)
        sweep = if get(ENV, "BENCH_BS", "") != ""
                    Tuple(parse(Int, s) for s in split(ENV["BENCH_BS"], ","))
                else sweep_default end

        ortho_list = if get(ENV, "BENCH_ORTHO", "") != ""
                         Tuple(Symbol(s) for s in split(ENV["BENCH_ORTHO"], ","))
                     else (:fast, :safe) end
        for ortho in ortho_list
            println("  -- ortho=$ortho --")
            for b in sweep
                try
                    r = bench_nextla(T, m, n, b; ortho=ortho, nwarm=1, nrun=5)
                    gflops = flops / (r.tmin * 1e6)
                    speedup = cu.tmin / r.tmin
                    @printf("  geqrf_2p5d! b=%-3d      min=%7.2f ms  med=%7.2f ms   (%.0f GF/s, %.1f%% peak, %.2fx)  c=%d  res=%.1e orth=%.1e\n",
                            r.b_used, r.tmin, r.tmed, gflops, 100*gflops/peak, speedup,
                            r.c, r.res, r.orth)
                catch e
                    println("  geqrf_2p5d! b=$b      FAILED ", typeof(e))
                end
            end
        end
        println()
    end
end

function main()
    println(" GPU: ", CUDA.name(CUDA.device()))
    P  = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    sm = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
    println(" P=$P SMs, M=$(sm÷8) FP64 words/SM\n")

    # Sweep the regimes:
    #  N=1000  (PM/N²≈0.3, c=1) — 2D
    #  N=2000  (PM/N²≈0.08, c=1) — still 2D
    #  N=4000  (PM/N²≈0.02, c=1) — surface 2.5D not applicable
    # For RTX 4060 Laptop the per-SM smem model gives small M; the 2.5D regime won't kick in
    # without overriding M. We still time large N to verify the trailing-GEMM-dominated regime.
    sizes = get(ENV, "BENCH_SIZES", "") != "" ?
            [parse(Int, s) for s in split(ENV["BENCH_SIZES"], ",")] :
            [1000, 2000, 4000]
    for N in sizes
        run_for_size(N, N)
    end
end

main()
