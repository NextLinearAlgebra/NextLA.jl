# NextLA TLR × TLR → dense GEMM benchmark
#
# Run (CPU only):
#   julia --project=. benchmark_gemm.jl
#
# Run (CPU + GPU):
#   julia --project=../gpuenv benchmark_gemm.jl
#
# Times `gemm!(C, A, B)` across problem sizes and the four operand-layout combinations
# (kj, kk, ik, ij). A large `max_workspace` is used so the scheduler batches maximally
# (row and column traversals both use their widest legal runs).
#
# Configs come in two families. `tail=0` is tile-aligned and hard-term-dominated: it
# measures the regular interior alone. `tail≠0` switches on the direct right/bottom
# panel and corner kernels.

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
    ("ik", M.TileColMajor, M.TileColMajor),   # FoldLeft row family
    ("ij", M.TileColMajor, M.TileRowMajor),   # column family
)

# (b = tile size, nt = tiles per side, r = maxrank, tail = m % b)
#
# `tail = 0` is tile-aligned: the product is the interior term alone. A non-zero tail
# switches on the right/bottom panels and the corner — six of the eight terms — which
# were otherwise never measured. Keep both: the aligned rows isolate the regular core,
# and the tailed rows are the signal for explicit boundary helpers.
const CONFIGS = (
    (b=64,  nt=16, r=16, tail=0),
    (b=64,  nt=32, r=16, tail=0),
    (b=64,  nt=48, r=24, tail=0),
    (b=128, nt=16, r=32, tail=0),   # big tiles — most compute-bound
    (b=256, nt=16, r=64, tail=0),
    (b=32,  nt=64, r=8,  tail=0),   # many tiny tiles — most launch-bound
    (b=64,  nt=32, r=16, tail=37),  # panels + corner, tail < b
    (b=32,  nt=64, r=8,  tail=17),  # panels + corner, launch-bound
    (b=128, nt=16, r=32, tail=65),  # panels + corner, big tiles
)

const T = Float32
const NREPS = 10
const WARMUP = 1

function make_tlr(backend, m, b, r, order)
    X = M.TLRMatrix(backend, T, m, m, (b, b), r; tile_order=order)
    # Panel/corner factors are empty when m % b == 0 and must be filled when it isn't;
    # `length == 0` covers both without branching on the tail.
    for f in (X.int_U, X.int_V, X.right_U, X.right_V, X.bottom_U, X.bottom_V,
              X.corner_U, X.corner_V)
        length(f) == 0 && continue
        randn!(f)
    end
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

function time_gemm(backend, b, nt, r, tail, oA, oB)
    m = b * nt + tail
    A = make_tlr(backend, m, b, r, oA)
    B = make_tlr(backend, m, b, r, oB)
    C = backend isa CPU ? randn(T, m, m) : CUDA.CuArray(randn(T, m, m))
    # Full-width budget for every direct regular and boundary kernel.
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
    return minimum(ts) * 1e3   # ms
end

function main()
    backend = HAS_CUDA ? CUDA.CUDABackend() : NextLA.KernelAbstractions.CPU()
    @printf("NextLA TLR gemm! benchmark — backend: %s, eltype: %s\n\n",
            HAS_CUDA ? "CUDA" : "CPU", T)
    @printf("%-30s%10s", "config (b×nt, r, tail)", "dense")
    for (name, _, _) in COMBOS; @printf("%10s", name); end
    println()
    println("-"^(30 + 10 * (length(COMBOS) + 1)))
    for cfg in CONFIGS
        @printf("%-30s", @sprintf("b=%d nt=%d r=%d tail=%d", cfg.b, cfg.nt, cfg.r, cfg.tail))
        @printf("%10.3f", time_dense(backend, cfg.b * cfg.nt + cfg.tail))
        for (_, oA, oB) in COMBOS
            t = time_gemm(backend, cfg.b, cfg.nt, cfg.r, cfg.tail, oA, oB)
            @printf("%10.3f", t)
        end
        println("  (ms)")
    end
end

main()
