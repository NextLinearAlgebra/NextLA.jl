# NextLA TLR × TLR → dense GEMM — boundary vs. padded benchmark
#
# Run (CPU only):
#   julia --project=. scripts/benchmark_boundary_vs_padded.jl
# Run (CPU + GPU):
#   julia --project=../gpuenv scripts/benchmark_boundary_vs_padded.jl
#
# Compares two ways of handling a non-uniform tiling (n % b ≠ 0):
#   * boundary  — `gemm!` on the n×n matrix directly (interior + right/bottom/corner
#                 regions, run on four streams).
#   * padded    — `gemm!` on a uniform n2×n2 matrix, n2 = cld(n,b)·b (the smallest
#                 multiple of b above n): no boundary terms, one interior product.
# We build a fresh uniform TLR at n2 rather than literally zero-padding the n×n one —
# the FLOP counts match, so it's a fair proxy for "pad, then run the uniform path".
#
# ratio = boundary / padded  (< 1 ⇒ decomposition is faster than padding).

using LinearAlgebra, Printf, Random, Statistics, KernelAbstractions

const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

using NextLA
const M = NextLA.TLRmodule

gpu_sync() = HAS_CUDA ? CUDA.synchronize() : nothing

const COMBOS = (
    ("kj", M.TileRowMajor, M.TileRowMajor),
    ("kk", M.TileRowMajor, M.TileColMajor),
    ("ik", M.TileColMajor, M.TileColMajor),
    ("ij", M.TileColMajor, M.TileRowMajor),
)

# (b = tile size, nt = interior tiles per side, tail = n % b, r = maxrank).
# n = nt·b + tail;  padded n2 = (nt+1)·b.  Sweeping `tail` shows the crossover:
# small tail ⇒ padding wastes ~a whole tile layer; tail→b ⇒ padding is nearly free.
const CONFIGS = (
    (b=64,  nt=16, tail=1,  r=16),   # tiny tail — worst case for padding
    (b=64,  nt=16, tail=32, r=16),   # half tail
    (b=64,  nt=32, tail=48, r=16),   # ¾ tail, larger grid
    (b=128, nt=16, tail=8,  r=32),   # tiny tail, big tiles
    (b=128, nt=16, tail=96, r=32),   # big tail, big tiles
    (b=32,  nt=64, tail=5,  r=8),    # many tiny tiles, small tail
)

const T = Float64
const NREPS = 5
const WARMUP = 2

function make_tlr(backend, m, b, r, order)
    X = M.TLRMatrix(backend, T, m, m, b, r; tile_order=order)
    randn!(X.int_U); randn!(X.int_V); randn!(X.D)
    size(X.D_corner, 3) != 0 && randn!(X.D_corner)
    size(X.right_U, 3)  != 0 && (randn!(X.right_U);  randn!(X.right_V))
    size(X.bottom_U, 3) != 0 && (randn!(X.bottom_U); randn!(X.bottom_V))
    X.ranks .= r
    return X
end

function time_dense(backend, m)
    Ad = backend isa CPU ? randn(T, m, m) : CUDA.CuArray(randn(T, m, m))
    Bd = backend isa CPU ? randn(T, m, m) : CUDA.CuArray(randn(T, m, m))
    Cd = backend isa CPU ? zeros(T, m, m) : CUDA.CuArray(zeros(T, m, m))
    for _ in 1:WARMUP; mul!(Cd, Ad, Bd); gpu_sync(); end
    ts = Float64[]
    for _ in 1:NREPS
        t = @elapsed begin; mul!(Cd, Ad, Bd); gpu_sync(); end
        push!(ts, t)
    end
    return minimum(ts) * 1e3
end

function time_gemm(backend, m, b, r, oA, oB)
    A = make_tlr(backend, m, b, r, oA)
    B = make_tlr(backend, m, b, r, oB)
    C = backend isa CPU ? randn(T, m, m) : CUDA.CuArray(randn(T, m, m))
    budget = M.gemm_maximum_workspace_bytes(A, B)
    for _ in 1:WARMUP
        M.gemm!(C, A, B; alpha=1.0, beta=0.5, max_workspace=budget); gpu_sync()
    end
    ts = Float64[]
    for _ in 1:NREPS
        t = @elapsed begin
            M.gemm!(C, A, B; alpha=1.0, beta=0.5, max_workspace=budget); gpu_sync()
        end
        push!(ts, t)
    end
    return minimum(ts) * 1e3
end

function main()
    backend = HAS_CUDA ? CUDA.CUDABackend() : NextLA.KernelAbstractions.CPU()
    @printf("NextLA TLR gemm! — boundary vs. padded — backend: %s, eltype: %s\n\n",
            HAS_CUDA ? "CUDA" : "CPU", T)
    for cfg in CONFIGS
        n  = cfg.nt * cfg.b + cfg.tail
        n2 = (cfg.nt + 1) * cfg.b
        @printf("=== b=%d nt=%d tail=%d r=%d   n=%d  n2=%d  (pad waste %.0f%% FLOPs) ===\n",
                cfg.b, cfg.nt, cfg.tail, cfg.r, n, n2, 100 * ((n2^3 - n^3) / n^3))
        @printf("%-11s%10s", "", "dense")
        for (name, _, _) in COMBOS; @printf("%10s", name); end
        println()
        # boundary row (size n), padded row (size n2), ratio row.
        bnd = Float64[]; pad = Float64[]
        @printf("%-11s%10.3f", "boundary", time_dense(backend, n))
        for (_, oA, oB) in COMBOS
            t = time_gemm(backend, n, cfg.b, cfg.r, oA, oB); push!(bnd, t); @printf("%10.3f", t)
        end
        println()
        @printf("%-11s%10.3f", "padded", time_dense(backend, n2))
        for (_, oA, oB) in COMBOS
            t = time_gemm(backend, n2, cfg.b, cfg.r, oA, oB); push!(pad, t); @printf("%10.3f", t)
        end
        println()
        @printf("%-11s%10s", "ratio b/p", "")
        for k in eachindex(COMBOS); @printf("%10.2f", bnd[k] / pad[k]); end
        println("\n")
    end
end

main()
