"""Shared configuration, timing, and execution code for dense-output sweeps."""
module DenseGemmCommon

using LinearAlgebra
using KernelAbstractions
using NextLA: DenseGemmWorkspace, GEMMCompute, TF32,
              gemm_minimum_workspace_bytes, precision_gemm!, uncompress!
using NextLA.TLRmodule: gemm!, get_factors, grid_size, tile_size

const _HAS_CUDA = try
    @eval import CUDA
    CUDA.functional()
catch
    false
end

include(joinpath(@__DIR__, "matrix_generation.jl"))
using .ExperimentMatrixGeneration: generate_ftlr_operands

export PrecisionConfig, RunConfig, MatrixCase, GemmTiming, GemmMetrics,
       GemmResult, run_cases, write_dense_csv, GEMMCompute, TF32

struct PrecisionConfig
    name::Symbol
    storage_type::DataType
    compute
end

PrecisionConfig(name, storage_type::Type, compute) =
    PrecisionConfig(Symbol(name), storage_type, compute)

_precision_config(::Type{T}) where {T} = PrecisionConfig(Symbol(T), T, GEMMCompute{T}())

struct RunConfig{B}
    precisions::Vector{PrecisionConfig}
    workspace_factor::Int
    nreps::Int
    nwarmup::Int
    seed::Int
    backend::B
end

function RunConfig(precisions, workspace_factor::Integer, nreps::Integer,
                   nwarmup::Integer, seed::Integer, backend)
    modes = PrecisionConfig[
        p isa PrecisionConfig ? p : _precision_config(p)
        for p in precisions
    ]
    return RunConfig(modes, Int(workspace_factor), Int(nreps), Int(nwarmup),
                     Int(seed), backend)
end

"""A complete storage/rank-distribution choice for one experiment."""
struct MatrixCase
    name::Symbol
    format::Symbol
    distribution::Symbol
    min_rank::Union{Nothing,Int}
    max_rank::Union{Nothing,Int}
end

function MatrixCase(name, format, distribution, min_rank, max_rank)
    return MatrixCase(Symbol(name), Symbol(format), Symbol(distribution),
                      isnothing(min_rank) ? nothing : Int(min_rank),
                      isnothing(max_rank) ? nothing : Int(max_rank))
end

struct GemmTiming
    tlr_dense_ms::Float64
    dense_tlr_ms::Float64
    tlr_tlr_ms::Float64
    dense_dense_ms::Float64
end

struct GemmMetrics
    dense_gflops::Float64
    tlr_dense_gflops::Float64
    dense_tlr_gflops::Float64
    tlr_tlr_gflops::Float64
    tlr_dense_speedup::Float64
    dense_tlr_speedup::Float64
    tlr_tlr_speedup::Float64
    tlr_dense_efficiency::Float64
    dense_tlr_efficiency::Float64
    tlr_tlr_efficiency::Float64
    tlr_dense_rel_error::Float64
    dense_tlr_rel_error::Float64
    tlr_tlr_rel_error::Float64
    dense_dense_rel_error::Float64
end

struct GemmResult
    experiment::Symbol
    case::Symbol
    precision::Symbol
    dtype::DataType
    m::Int
    k::Int
    n::Int
    tile_size::Int
    rank_A::Int
    rank_B::Int
    format::Symbol
    rank_distribution::Symbol
    min_rank::Int
    max_rank::Int
    timing::GemmTiming
    metrics::GemmMetrics
end

function run_cases(experiment::Symbol, shapes, tile_size::Int, ranks::NTuple{2,Int},
                   cases, run::RunConfig; square=false)
    b = tile_size
    b > 0 || throw(ArgumentError("tile_size must be positive"))
    run.workspace_factor >= 1 || throw(ArgumentError("workspace_factor must be positive"))
    run.nreps >= 1 || throw(ArgumentError("nreps must be positive"))
    run.nwarmup >= 0 || throw(ArgumentError("nwarmup must be nonnegative"))
    results = GemmResult[]

    for case in cases, precision in run.precisions, (shape_index, shape) in enumerate(shapes)
        T = precision.storage_type
        m, k, n = shape
        (!square || m == k == n) || throw(ArgumentError("square experiment received $shape"))
        m % b == 0 && k % b == 0 && n % b == 0 ||
            throw(ArgumentError("$shape must be divisible by tile_size=$b"))
        lo = isnothing(case.min_rank) ? min(ranks...) : case.min_rank
        hi = isnothing(case.max_rank) ? max(ranks...) : case.max_rank
        lo >= 0 && lo <= hi <= b || throw(ArgumentError("invalid rank range in $(case.name)"))
        ranks[1] < b && ranks[2] < b || throw(ArgumentError("ranks must be smaller than tile_size"))

        A_tlr, B_tlr = generate_ftlr_operands(
            m, k, n, b, ranks, T; seed=run.seed + shape_index,
            backend=run.backend, format=case.format,
            rank_distribution=case.distribution, min_rank=lo, max_rank=hi)
        C = _backend_zeros(run.backend, T, m, n)
        reference = _reference_result(run.backend, T, A_tlr, B_tlr, m, n)
        reference_host = Array(reference)
        dense_flops, tlr_dense_flops, dense_tlr_flops, tlr_tlr_flops =
            _flop_counts(A_tlr, B_tlr, m, k, n)

        workspace = DenseGemmWorkspace(A_tlr, B_tlr;
            bytes=run.workspace_factor * gemm_minimum_workspace_bytes(A_tlr, B_tlr))
        tlr_tlr_ms = _time_gemm!(C, T, run) do
            gemm!(C, A_tlr, B_tlr; workspace, alpha=one(T), beta=one(T),
                  compute=precision.compute)
        end
        tlr_tlr_error = _relative_error(C, reference_host)
        workspace = nothing
        _collect_large_temporaries!()

        B_dense = _uncompress(run.backend, T, B_tlr)
        workspace = DenseGemmWorkspace(A_tlr, run.workspace_factor * 3 * sizeof(T))
        tlr_dense_ms = _time_gemm!(C, T, run) do
            gemm!(C, A_tlr, B_dense; workspace, alpha=one(T), beta=one(T),
                  compute=precision.compute)
        end
        tlr_dense_error = _relative_error(C, reference_host)
        workspace = nothing; B_dense = nothing; _collect_large_temporaries!()

        A_dense = _uncompress(run.backend, T, A_tlr)
        workspace = DenseGemmWorkspace(B_tlr, run.workspace_factor * 3 * sizeof(T))
        dense_tlr_ms = _time_gemm!(C, T, run) do
            gemm!(C, A_dense, B_tlr; workspace, alpha=one(T), beta=one(T),
                  compute=precision.compute)
        end
        dense_tlr_error = _relative_error(C, reference_host)
        workspace = nothing; A_dense = nothing; _collect_large_temporaries!()

        A_dense = _uncompress(run.backend, T, A_tlr); A_tlr = nothing
        _collect_large_temporaries!()
        B_dense = _uncompress(run.backend, T, B_tlr); B_tlr = nothing
        _collect_large_temporaries!()
        dense_dense_ms = _time_gemm!(C, T, run) do
            precision_gemm!('N', 'N', one(T), A_dense, B_dense, one(T), C,
                            precision.compute)
        end
        dense_dense_error = _relative_error(C, reference_host)

        dense_gflops = _gflops(dense_flops, dense_dense_ms)
        tlr_dense_gflops = _gflops(tlr_dense_flops, tlr_dense_ms)
        dense_tlr_gflops = _gflops(dense_tlr_flops, dense_tlr_ms)
        tlr_tlr_gflops = _gflops(tlr_tlr_flops, tlr_tlr_ms)
        metrics = GemmMetrics(
            dense_gflops, tlr_dense_gflops, dense_tlr_gflops, tlr_tlr_gflops,
            dense_dense_ms / tlr_dense_ms, dense_dense_ms / dense_tlr_ms,
            dense_dense_ms / tlr_tlr_ms,
            tlr_dense_gflops / dense_gflops,
            dense_tlr_gflops / dense_gflops,
            tlr_tlr_gflops / dense_gflops,
            tlr_dense_error, dense_tlr_error, tlr_tlr_error, dense_dense_error)

        push!(results, GemmResult(experiment, case.name, precision.name, T, m, k, n, b,
            ranks[1], ranks[2], case.format, case.distribution, lo, hi,
            GemmTiming(tlr_dense_ms, dense_tlr_ms, tlr_tlr_ms, dense_dense_ms), metrics))
        A_dense = nothing; B_dense = nothing; C = nothing; reference = nothing
        _collect_large_temporaries!()
    end
    return results
end

function _reference_result(backend, ::Type{T}, A, B, m, n) where {T}
    A_storage = _uncompress(backend, T, A)
    B_storage = _uncompress(backend, T, B)
    A_ref = _backend_zeros(backend, Float32, size(A)...)
    B_ref = _backend_zeros(backend, Float32, size(B)...)
    copyto!(A_ref, A_storage); copyto!(B_ref, B_storage)
    C_ref = _backend_zeros(backend, Float32, m, n)
    fill!(C_ref, 1.0f0)
    mul!(C_ref, A_ref, B_ref, 1.0f0, 1.0f0)
    return C_ref
end

function _flop_counts(A, B, m, k, n)
    mt, kt = grid_size(A); _, nt = grid_size(B)
    tlr_dense = dense_tlr = tlr_tlr = 0.0
    for i in 1:mt, l in 1:kt, j in 1:nt
        tm = tile_size(A, i, l)[1]
        tk = tile_size(A, i, l)[2]
        tn = tile_size(B, l, j)[2]
        ra = size(get_factors(A, i, l)[1], 2)
        rb = size(get_factors(B, l, j)[1], 2)
        tlr_dense += 2.0 * ra * (tk * n + tm * n)
        dense_tlr += 2.0 * rb * (m * tk + m * tn)
        tlr_tlr += 2.0 * (tk * ra * rb + tm * ra * rb + tm * rb * tn)
    end
    return 2.0 * m * k * n, tlr_dense, dense_tlr, tlr_tlr
end

@inline _gflops(flops, milliseconds) = flops / (milliseconds * 1.0e6)

function _relative_error(C, reference_host)
    result = Array(C)
    denominator = max(norm(reference_host), eps(Float32))
    norm(result .- reference_host) / denominator
end

function _time_gemm!(f, C, ::Type{T}, run::RunConfig) where {T}
    for _ in 1:run.nwarmup
        fill!(C, one(T)); f(); _synchronize(run.backend)
    end
    best = Inf
    for _ in 1:run.nreps
        fill!(C, one(T)); _synchronize(run.backend)
        start = time_ns(); f(); _synchronize(run.backend)
        best = min(best, (time_ns() - start) / 1.0e6)
    end
    return best
end

function _uncompress(backend, ::Type{T}, A) where {T}
    dense = _backend_zeros(backend, T, size(A)...)
    uncompress!(dense, A); _synchronize(backend); return dense
end

_backend_zeros(::KernelAbstractions.CPU, ::Type{T}, dims...) where {T} = zeros(T, dims...)
function _backend_zeros(backend, ::Type{T}, dims...) where {T}
    _HAS_CUDA || throw(ArgumentError("non-CPU backend requires CUDA"))
    CUDA.zeros(T, dims...)
end
_synchronize(::KernelAbstractions.CPU) = nothing
_synchronize(backend) = KernelAbstractions.synchronize(backend)

function _collect_large_temporaries!()
    GC.gc(true); _HAS_CUDA && CUDA.reclaim(); nothing
end

function write_dense_csv(path, results)
    header = ["experiment", "case", "precision", "dtype", "m", "k", "n", "tile_size",
              "rank_A", "rank_B", "format", "rank_distribution", "min_rank",
              "max_rank", "tlr_dense_ms", "dense_tlr_ms", "tlr_tlr_ms",
              "dense_dense_ms", "dense_gflops", "tlr_dense_gflops",
              "dense_tlr_gflops", "tlr_tlr_gflops", "tlr_dense_speedup",
              "dense_tlr_speedup", "tlr_tlr_speedup", "tlr_dense_efficiency",
              "dense_tlr_efficiency", "tlr_tlr_efficiency", "tlr_dense_rel_error",
              "dense_tlr_rel_error", "tlr_tlr_rel_error", "dense_dense_rel_error"]
    csv_value(x) = begin
        s = string(x)
        occursin(',', s) || occursin('"', s) ?
            "\"" * replace(s, '"' => "\"\"") * "\"" : s
    end
    open(path, "w") do io
        println(io, join(header, ','))
        for r in results
            row = (r.experiment, r.case, r.precision, r.dtype, r.m, r.k, r.n, r.tile_size,
                   r.rank_A, r.rank_B, r.format, r.rank_distribution, r.min_rank,
                   r.max_rank, r.timing.tlr_dense_ms, r.timing.dense_tlr_ms,
                   r.timing.tlr_tlr_ms, r.timing.dense_dense_ms,
                   r.metrics.dense_gflops, r.metrics.tlr_dense_gflops,
                   r.metrics.dense_tlr_gflops, r.metrics.tlr_tlr_gflops,
                   r.metrics.tlr_dense_speedup, r.metrics.dense_tlr_speedup,
                   r.metrics.tlr_tlr_speedup, r.metrics.tlr_dense_efficiency,
                   r.metrics.dense_tlr_efficiency, r.metrics.tlr_tlr_efficiency,
                   r.metrics.tlr_dense_rel_error, r.metrics.dense_tlr_rel_error,
                   r.metrics.tlr_tlr_rel_error, r.metrics.dense_dense_rel_error)
            println(io, join(csv_value.(row), ','))
        end
    end
    path
end

end
