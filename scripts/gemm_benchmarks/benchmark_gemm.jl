# Large TLR × TLR → dense GEMM benchmark. Run this file directly or use
# `run_gemm_benchmark.sh`; both use the shared configuration in `config.jl`.

using LinearAlgebra
using Printf
using Random
using Statistics
using KernelAbstractions

if !isdefined(@__MODULE__, :GemmBenchmarksConfig)
    include(joinpath(@__DIR__, "config.jl"))
end
using .GemmBenchmarksConfig

const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

using NextLA
const TLRM = NextLA.TLRmodule

if !isdefined(@__MODULE__, :CONFIG)
    const CONFIG = load_config(; default_benchmark=:dense)
end
const NREPS = CONFIG.reps
const WARMUP = CONFIG.warmup
const OUTPUT = output_path(CONFIG, :dense)
const SHARD_COUNT = CONFIG.shard_count
const SHARD_INDEX = CONFIG.shard_index
const CASE_REGEX = CONFIG.case_regex
const SEED = CONFIG.seed

# (m, k, n): four square products and four rectangular-A products against a
# square B. All dimensions divide exactly under both tile ratios.
const SHAPES = (
    (name="square_1024", m=1024, k=1024, n=1024),
    (name="square_2048", m=2048, k=2048, n=2048),
    (name="square_4096", m=4096, k=4096, n=4096),
    (name="square_8192", m=8192, k=8192, n=8192),
    (name="rect_512x1024", m=512, k=1024, n=1024),
    (name="rect_1024x2048", m=1024, k=2048, n=2048),
    (name="rect_2048x4096", m=2048, k=4096, n=4096),
    (name="rect_4096x8192", m=4096, k=8192, n=8192),
)

# bm/m = bk/k = bn/n. The denominator is also the number of tiles along
# every dimension.
const TILE_DENOMINATORS = (8, 4)
const RANK_DENOMINATORS = (2, 4, 8)

# The four physical-axis combinations. These are storage layouts, not
# transpose operations.
const AXIS_COMBINATIONS = (
    (name="kj", orderA=TLRM.TileRowMajor, orderB=TLRM.TileRowMajor),
    (name="kk", orderA=TLRM.TileRowMajor, orderB=TLRM.TileColMajor),
    (name="ik", orderA=TLRM.TileColMajor, orderB=TLRM.TileColMajor),
    (name="ij", orderA=TLRM.TileColMajor, orderB=TLRM.TileRowMajor),
)

# Actual ranks are deterministic, strictly below maxrank, and vary by tile.
# Factor columns above the actual rank are explicitly zeroed. Override this
# distribution without changing the case grid if another rank profile is
# desired.
const ACTUAL_RANK_FRACTIONS = (0.25, 0.50, 0.75)

const PRECISIONS = (
    (name="fp32", T=Float32, compute=NextLA.GEMMCompute{Float32}()),
    #(name="fp64", T=Float64, compute=NextLA.GEMMCompute{Float64}()),
    #(name="fp32_tf32", T=Float32, compute=NextLA.TF32()),
)

const CSV_COLUMNS = (
    "case_id", "shape", "m", "k", "n", "tile_ratio", "bm", "bk", "bn",
    "rank_ratio_A", "rank_ratio_B", "maxrank_A", "maxrank_B",
    "axis", "precision", "nreps", "workspace_bytes",
    "dense_ms", "tlr_ms", "speedup",
    "tlr_work_ratio_pct", "tlr_arithmetic_reduction",
    "tlr_rate_ratio_pct", "wasted_flops_pct",
    "dense_gflops", "tlr_executed_gflops", "tlr_dense_equiv_gflops",
    "padded_flops", "ideal_rank_flops",
)

@inline backend_sync(backend) =
    backend isa KernelAbstractions.CPU ? nothing :
    KernelAbstractions.synchronize(backend)

@inline backend_randn(backend, ::Type{T}, dims...) where {T} =
    backend isa KernelAbstractions.CPU ? randn(T, dims...) :
    CUDA.randn(T, dims...)

@inline backend_zeros(backend, ::Type{T}, dims...) where {T} =
    backend isa KernelAbstractions.CPU ? zeros(T, dims...) :
    CUDA.zeros(T, dims...)

function release_backend_memory!()
    GC.gc(true)
    HAS_CUDA && CUDA.reclaim()
    return nothing
end

@inline function physical_slot(order, qm, qn, i, j)
    return TLRM.tile_linear_index(order(), qm, qn, i, j)
end

@inline function actual_rank(maxrank::Int, i::Int, j::Int, seed::Int)
    f = ACTUAL_RANK_FRACTIONS[mod1(3i + 5j + seed, length(ACTUAL_RANK_FRACTIONS))]
    return clamp(round(Int, f * maxrank), 1, maxrank - 1)
end

function make_tlr(backend, ::Type{T}, m, n, bm, bn, maxrank, order;
                  seed) where {T}
    X = TLRM.TLRMatrix(
        backend, T, m, n, (bm, bn), maxrank; tile_order=order)
    randn!(X.int_U)
    randn!(X.int_V)
    qm, qn = NextLA.grid_size(X)
    ranks = Matrix{Int}(undef, qm, qn)
    @inbounds for i in 1:qm, j in 1:qn
        r = actual_rank(maxrank, i, j, seed)
        ranks[i, j] = r
        slot = physical_slot(order, qm, qn, i, j)
        X.ranks[slot] = r
        fill!(view(X.int_U, :, (r + 1):maxrank, slot), zero(T))
        fill!(view(X.int_V, :, (r + 1):maxrank, slot), zero(T))
    end
    backend_sync(backend)
    return X, ranks
end

function best_time_ms(f, backend)
    for _ in 1:WARMUP
        f()
        backend_sync(backend)
    end
    best = Inf
    for _ in 1:NREPS
        backend_sync(backend)
        t0 = time_ns()
        f()
        backend_sync(backend)
        best = min(best, (time_ns() - t0) / 1e6)
    end
    return best
end

@inline dense_flops(m, k, n) = 2.0 * m * k * n

function benchmark_dense(backend, case)
    T = case.T
    A = backend_randn(backend, T, case.m, case.k)
    B = backend_randn(backend, T, case.k, case.n)
    C = backend_zeros(backend, T, case.m, case.n)
    oneT, zeroT = one(T), zero(T)
    f = () -> NextLA._gemm_compute!(
        case.compute, 'N', 'N', oneT, A, B, zeroT, C)
    ms = best_time_ms(f, backend)
    flops = dense_flops(case.m, case.k, case.n)
    result = (; ms, gflops=flops / (ms * 1e6))
    A = B = C = nothing
    release_backend_memory!()
    return result
end

@inline function tile_product_flops(bm, bk, bn, rA, rB, fold_left)
    core = 2.0 * bk * rA * rB
    if fold_left
        return core + 2.0 * bm * rA * rB + 2.0 * bm * rB * bn
    end
    return core + 2.0 * rA * rB * bn + 2.0 * bm * rA * bn
end

function tlr_flop_model(case, ranksA, ranksB)
    qm, qk = size(ranksA)
    qkB, qn = size(ranksB)
    qk == qkB || error("rank grids have incompatible contraction dimensions")

    # This is the global fold predicate used by choose_fold for the canonical
    # factors. Keeping it fixed in the ideal-rank count isolates work wasted
    # by padding from work changed by selecting another algorithm.
    fold_left = case.bm * case.maxrank_B <
                case.maxrank_A * case.bn
    per_padded = tile_product_flops(
        case.bm, case.bk, case.bn,
        case.maxrank_A, case.maxrank_B, fold_left)
    padded = qm * qk * qn * per_padded
    ideal = 0.0
    @inbounds for i in 1:qm, kk in 1:qk, j in 1:qn
        ideal += tile_product_flops(
            case.bm, case.bk, case.bn,
            ranksA[i, kk], ranksB[kk, j], fold_left)
    end
    waste = padded == 0 ? 0.0 : 100 * (padded - ideal) / padded
    return (; padded, ideal, waste, fold_left)
end

function benchmark_tlr(backend, case)
    T = case.T
    A, ranksA = make_tlr(
        backend, T, case.m, case.k, case.bm, case.bk,
        case.maxrank_A, case.orderA; seed=SEED + 11)
    B, ranksB = make_tlr(
        backend, T, case.k, case.n, case.bk, case.bn,
        case.maxrank_B, case.orderB; seed=SEED + 29)
    C = backend_zeros(backend, T, case.m, case.n)
    bytes = TLRM.gemm_maximum_workspace_bytes(A, B)
    workspace = NextLA.DenseGemmWorkspace(A, B; bytes)
    f = () -> TLRM.gemm!(
        C, A, B; alpha=one(T), beta=zero(T),
        compute=case.compute, workspace)
    ms = best_time_ms(f, backend)
    model = tlr_flop_model(case, ranksA, ranksB)
    result = (; ms, bytes, model)
    A = B = C = workspace = nothing
    release_backend_memory!()
    return result
end

function case_id(case)
    return join((
        case.shape, "t$(case.tile_den)", "ra$(case.rank_den_A)",
        "rb$(case.rank_den_B)", case.axis, case.precision,
    ), "__")
end

function all_cases()
    cases = NamedTuple[]
    for shape in SHAPES
        for tile_den in TILE_DENOMINATORS
            bm, bk, bn = shape.m ÷ tile_den, shape.k ÷ tile_den, shape.n ÷ tile_den
            minb = min(bm, bk, bn)
            for rank_den_A in RANK_DENOMINATORS
                for rank_den_B in RANK_DENOMINATORS
                    maxrank_A = minb ÷ rank_den_A
                    maxrank_B = minb ÷ rank_den_B
                    for axes in AXIS_COMBINATIONS
                        for precision in PRECISIONS
                            push!(cases, (
                                shape=shape.name, m=shape.m, k=shape.k, n=shape.n,
                                tile_den, bm, bk, bn,
                                rank_den_A, rank_den_B, maxrank_A, maxrank_B,
                                axis=axes.name, orderA=axes.orderA,
                                orderB=axes.orderB, precision=precision.name,
                                T=precision.T, compute=precision.compute,
                            ))
                        end
                    end
                end
            end
        end
    end
    return cases
end

function completed_cases(path)
    isfile(path) || return Set{String}()
    ids = Set{String}()
    open(path, "r") do io
        eof(io) && return
        readline(io)
        for line in eachline(io)
            isempty(line) || push!(ids, first(split(line, ',')))
        end
    end
    return ids
end

function write_header_if_needed(path)
    expected = join(CSV_COLUMNS, ',')
    if isfile(path) && filesize(path) > 0
        actual = open(readline, path)
        actual == expected || error(
            "benchmark CSV schema mismatch in $path; choose a new " *
            "output directory or remove the obsolete file")
    else
        open(path, "w") do io
            println(io, expected)
        end
    end
end

function append_result(path, case, dense, tlr)
    nominal = dense_flops(case.m, case.k, case.n)
    tlr_executed = tlr.model.padded / (tlr.ms * 1e6)
    tlr_equiv = nominal / (tlr.ms * 1e6)
    speedup = dense.ms / tlr.ms
    work_ratio = tlr.model.padded / nominal
    arithmetic_reduction = nominal / tlr.model.padded
    rate_ratio = tlr_executed / dense.gflops
    row = (
        case_id(case), case.shape, case.m, case.k, case.n,
        "1/$(case.tile_den)", case.bm, case.bk, case.bn,
        "1/$(case.rank_den_A)", "1/$(case.rank_den_B)",
        case.maxrank_A, case.maxrank_B, case.axis, case.precision,
        NREPS, tlr.bytes, dense.ms, tlr.ms, speedup,
        100 * work_ratio, arithmetic_reduction, 100 * rate_ratio,
        tlr.model.waste, dense.gflops, tlr_executed, tlr_equiv,
        tlr.model.padded, tlr.model.ideal,
    )
    open(path, "a") do io
        println(io, join(row, ','))
        flush(io)
    end
    @printf(
        "%-58s dense=%8.3f ms  TLR=%8.3f ms  speedup=%6.2fx  work=%5.1f%%  rate=%5.1f%%  waste=%5.1f%%\n",
        case_id(case), dense.ms, tlr.ms, speedup,
        100 * work_ratio, 100 * rate_ratio, tlr.model.waste,
    )
end

function main()
    backend_name = CONFIG.backend === :auto ? (HAS_CUDA ? "cuda" : "cpu") :
                    string(CONFIG.backend)
    backend = if backend_name == "cuda"
        HAS_CUDA || error("CUDA was requested but is not functional")
        CUDA.CUDABackend()
    elseif backend_name == "cpu"
        KernelAbstractions.CPU()
    else
        error("backend must be `auto`, `cuda`, or `cpu`")
    end

    write_header_if_needed(OUTPUT)
    done = completed_cases(OUTPUT)
    cases = all_cases()
    selected = [
        case for (index, case) in enumerate(cases)
        if mod1(index, SHARD_COUNT) == SHARD_INDEX &&
           occursin(CASE_REGEX, case_id(case)) &&
           !(case_id(case) in done) &&
           !(backend isa KernelAbstractions.CPU && case.compute isa NextLA.TF32)
    ]
    @printf(
        "NextLA dense-output TLR GEMM benchmark\nbackend=%s cases=%d/%d reps=%d warmup=%d shard=%d/%d output=%s\n",
        backend_name, length(selected), length(cases), NREPS, WARMUP,
        SHARD_INDEX, SHARD_COUNT, OUTPUT,
    )

    dense_cache = Dict{Tuple{Int,Int,Int,String},NamedTuple}()
    for case in selected
        dense_key = (case.m, case.k, case.n, case.precision)
        dense = get!(dense_cache, dense_key) do
            benchmark_dense(backend, case)
        end
        tlr = benchmark_tlr(backend, case)
        append_result(OUTPUT, case, dense, tlr)
    end
    return nothing
end

isdefined(@__MODULE__, :GEMM_BENCHMARK_LIBRARY_ONLY) || main()
