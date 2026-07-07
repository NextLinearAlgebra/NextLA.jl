# NextLA TLR × TLR → dense GEMM benchmark
#
# Run (CPU only):
#   julia --project=. benchmark_gemm.jl
#
# Run (CPU + GPU):
#   julia --project=../gpuenv benchmark_gemm.jl
#
# Times the hard-term-dominated `gemm!(C, A, B)` across problem sizes and the four
# operand-layout combinations (kj, kk, ik, ij). A large `max_workspace` is used so
# the scheduler batches maximally (row family → 3 launches, column → 2+nt).

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

# The four layout combinations: (name, A order, B order).
const COMBOS = (
    ("kj", M.TileRowMajor, M.TileRowMajor),   # row family (fusable Stage 1)
    ("kk", M.TileRowMajor, M.TileColMajor),   # row family (tilewise Stage 1)
    ("ik", M.TileColMajor, M.TileColMajor),   # column family
    ("ij", M.TileColMajor, M.TileRowMajor),   # column family
)

# (b = tile size, nt = tiles per side, r = maxrank)
const CONFIGS = (
    (b=64,  nt=16, r=16),
    (b=64,  nt=32, r=16),
    (b=64,  nt=48, r=24),
    (b=128, nt=16, r=32),   # big tiles — most compute-bound
    (b=256, nt=16, r=64),
    (b=32,  nt=64, r=8),    # many tiny tiles — most launch-bound
)

const T = Float64
const NREPS = 5
const WARMUP = 2

function make_tlr(backend, m, b, r, order)
    X = M.TLRMatrix(backend, T, m, m, b, r; tile_order=order)
    randn!(X.int_U); randn!(X.int_V); randn!(X.D)
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

function time_gemm(backend, b, nt, r, oA, oB)
    m = b * nt
    A = make_tlr(backend, m, b, r, oA)
    B = make_tlr(backend, m, b, r, oB)
    C = backend isa CPU ? randn(T, m, m) : CUDA.CuArray(randn(T, m, m))
    budget = M.gemm_workspace_bytes(A, B)          # full batching
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
    return minimum(ts) * 1e3   # ms
end

function main()
    backend = HAS_CUDA ? CUDA.CUDABackend() : NextLA.KernelAbstractions.CPU()
    @printf("NextLA TLR gemm! benchmark — backend: %s, eltype: %s\n\n",
            HAS_CUDA ? "CUDA" : "CPU", T)
    @printf("%-22s%10s", "config (b×nt, r)", "dense")
    for (name, _, _) in COMBOS; @printf("%10s", name); end
    println()
    println("-"^(22 + 10 * (length(COMBOS) + 1)))
    for cfg in CONFIGS
        @printf("%-22s", @sprintf("b=%d nt=%d r=%d", cfg.b, cfg.nt, cfg.r))
        @printf("%10.3f", time_dense(backend, cfg.b * cfg.nt))
        for (_, oA, oB) in COMBOS
            t = time_gemm(backend, cfg.b, cfg.nt, cfg.r, oA, oB)
            @printf("%10.3f", t)
        end
        println("  (ms)")
    end
end

main()
