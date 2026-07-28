"""Configuration-driven GEMM experiments."""
module StrongScalingExperiment

using LinearAlgebra
using KernelAbstractions
using NextLA: DenseGemmWorkspace, gemm_minimum_workspace_bytes, uncompress!
using NextLA.TLRmodule: gemm!

const _HAS_CUDA = try
    @eval import CUDA
    CUDA.functional()
catch
    false
end

include(joinpath(@__DIR__, "matrix_generation.jl"))
using .ExperimentMatrixGeneration: generate_tlr_operands

export RunConfig, StrongScalingConfig, RankSweepConfig, TileSizeSweepConfig,
       MatrixShapeSweepConfig, GemmTiming, GemmResult
export strong_scaling, rank_sweep, tile_size_sweep, matrix_shape_sweep

struct RunConfig{B}
    dtypes::Vector{DataType}
    workspace_factor::Int
    nreps::Int
    nwarmup::Int
    seed::Int
    backend::B
end

function RunConfig(dtypes, workspace_factor::Integer, nreps::Integer,
                   nwarmup::Integer, seed::Integer, backend)
    return RunConfig(DataType[dtypes...], Int(workspace_factor), Int(nreps),
                     Int(nwarmup), Int(seed), backend)
end

struct StrongScalingConfig{B}
    sizes::Vector{NTuple{3,Int}}
    tile_size::Int
    ranks::NTuple{2,Int}
    run::RunConfig{B}
end

function StrongScalingConfig(sizes, tile_size, ranks, run::RunConfig)
    shapes = sizes isa AbstractVector{<:Integer} ?
        [(Int(s), Int(s), Int(s)) for s in sizes] :
        [Tuple(Int.(shape)) for shape in sizes]
    return StrongScalingConfig(shapes, Int(tile_size), Int.(ranks), run)
end

struct RankSweepConfig{B}
    matrix_size::Int
    tile_size::Int
    ranks::Vector{Int}
    run::RunConfig{B}
end

RankSweepConfig(matrix_size, tile_size, ranks, run::RunConfig) =
    RankSweepConfig(Int(matrix_size), Int(tile_size), Int.(ranks), run)

struct TileSizeSweepConfig{B}
    matrix_size::Int
    tile_sizes::Vector{Int}
    rank::Int
    run::RunConfig{B}
end

TileSizeSweepConfig(matrix_size, tile_sizes, rank, run::RunConfig) =
    TileSizeSweepConfig(Int(matrix_size), Int.(tile_sizes), Int(rank), run)

struct MatrixShapeSweepConfig{B}
    base_size::Int
    tile_size::Int
    rank::Int
    ratios::Vector{NTuple{3,Float64}}
    run::RunConfig{B}
end

function MatrixShapeSweepConfig(base_size, tile_size, rank, ratios, run::RunConfig)
    return MatrixShapeSweepConfig(Int(base_size), Int(tile_size), Int(rank),
                                  [Tuple(Float64.(r)) for r in ratios], run)
end

struct GemmTiming
    tlr_dense_ms::Float64
    dense_tlr_ms::Float64
    tlr_tlr_ms::Float64
    dense_dense_ms::Float64
end

struct GemmResult
    experiment::Symbol
    dtype::DataType
    m::Int
    k::Int
    n::Int
    tile_size::Int
    rank_A::Int
    rank_B::Int
    timing::GemmTiming
end

function strong_scaling(config::StrongScalingConfig)
    return _run_cases(:strong_scaling, config.sizes, config.tile_size,
                      config.ranks, config.run; square=true)
end

function rank_sweep(config::RankSweepConfig)
    results = GemmResult[]
    for rank in config.ranks
        rank < config.tile_size ||
            throw(ArgumentError("rank=$rank must be smaller than tile_size=$(config.tile_size)"))
        shape = (config.matrix_size, config.matrix_size, config.matrix_size)
        append!(results, _run_cases(:rank_sweep, [shape], config.tile_size,
                                    (rank, rank), config.run; square=true))
    end
    return results
end

function tile_size_sweep(config::TileSizeSweepConfig)
    results = GemmResult[]
    for tile_size in config.tile_sizes
        config.rank < tile_size ||
            throw(ArgumentError("rank=$(config.rank) must be smaller than tile_size=$tile_size"))
        shape = (config.matrix_size, config.matrix_size, config.matrix_size)
        append!(results, _run_cases(:tile_size_sweep, [shape], tile_size,
                                    (config.rank, config.rank), config.run; square=true))
    end
    return results
end

function matrix_shape_sweep(config::MatrixShapeSweepConfig)
    shapes = NTuple{3,Int}[]
    for ratio in config.ratios
        length(ratio) == 3 || throw(ArgumentError("shape ratios must have length three"))
        scale = config.base_size / cbrt(prod(ratio))
        shape = ntuple(i -> max(config.tile_size,
            round(Int, scale * ratio[i] / config.tile_size) * config.tile_size), 3)
        push!(shapes, shape)
    end
    return _run_cases(:matrix_shape_sweep, shapes, config.tile_size,
                      (config.rank, config.rank), config.run; square=false)
end

function _run_cases(experiment::Symbol, shapes::Vector{NTuple{3,Int}},
                    tile_size::Int, ranks::NTuple{2,Int}, run::RunConfig;
                    square::Bool)
    b = tile_size
    rA, rB = ranks
    b > 0 || throw(ArgumentError("tile_size must be positive"))
    rA < b && rB < b || throw(ArgumentError("ranks must be smaller than tile_size"))
    run.workspace_factor >= 1 || throw(ArgumentError("workspace_factor must be positive"))
    run.nreps >= 1 || throw(ArgumentError("nreps must be positive"))
    run.nwarmup >= 0 || throw(ArgumentError("nwarmup must be nonnegative"))

    results = GemmResult[]
    for T in run.dtypes
        for (case_index, shape) in enumerate(shapes)
            m, k, n = shape
            (!square || m == k == n) ||
                throw(ArgumentError("square experiment received shape $shape"))
            m % b == 0 && k % b == 0 && n % b == 0 ||
                throw(ArgumentError("$shape must be divisible by tile_size=$b"))

            A_tlr, B_tlr = generate_tlr_operands(
                m, k, n, b, ranks, T; seed=run.seed + case_index, backend=run.backend)
            C = _backend_zeros(run.backend, T, m, n)

            workspace = DenseGemmWorkspace(
                A_tlr, B_tlr;
                bytes=run.workspace_factor * gemm_minimum_workspace_bytes(A_tlr, B_tlr))
            tlr_tlr_ms = _time_gemm!(C, T, run) do
                gemm!(C, A_tlr, B_tlr; workspace, alpha=one(T), beta=one(T))
            end
            workspace = nothing
            _collect_large_temporaries!()

            B_dense = _uncompress(run.backend, T, B_tlr)
            workspace = DenseGemmWorkspace(A_tlr,
                run.workspace_factor * 3 * sizeof(T))
            tlr_dense_ms = _time_gemm!(C, T, run) do
                gemm!(C, A_tlr, B_dense; workspace, alpha=one(T), beta=one(T))
            end
            workspace = nothing
            B_dense = nothing
            _collect_large_temporaries!()

            A_dense = _uncompress(run.backend, T, A_tlr)
            workspace = DenseGemmWorkspace(B_tlr,
                run.workspace_factor * 3 * sizeof(T))
            dense_tlr_ms = _time_gemm!(C, T, run) do
                gemm!(C, A_dense, B_tlr; workspace, alpha=one(T), beta=one(T))
            end
            workspace = nothing
            A_dense = nothing
            _collect_large_temporaries!()

            A_dense = _uncompress(run.backend, T, A_tlr)
            A_tlr = nothing
            _collect_large_temporaries!()
            B_dense = _uncompress(run.backend, T, B_tlr)
            B_tlr = nothing
            _collect_large_temporaries!()
            dense_dense_ms = _time_gemm!(C, T, run) do
                mul!(C, A_dense, B_dense, one(T), one(T))
            end

            push!(results, GemmResult(
                experiment, T, m, k, n, b, rA, rB,
                GemmTiming(tlr_dense_ms, dense_tlr_ms, tlr_tlr_ms, dense_dense_ms)))
            A_dense = nothing
            B_dense = nothing
            C = nothing
            _collect_large_temporaries!()
        end
    end
    return results
end

function _time_gemm!(f, C, ::Type{T}, run::RunConfig{B}) where {T,B}
    for _ in 1:run.nwarmup
        _reset_output!(C, T)
        f()
        _synchronize(run.backend)
    end
    best_ms = Inf
    for _ in 1:run.nreps
        _reset_output!(C, T)
        _synchronize(run.backend)
        start = time_ns()
        f()
        _synchronize(run.backend)
        best_ms = min(best_ms, (time_ns() - start) / 1.0e6)
    end
    return best_ms
end

function _uncompress(backend, ::Type{T}, A) where {T}
    dense = _backend_zeros(backend, T, size(A)...)
    uncompress!(dense, A)
    _synchronize(backend)
    return dense
end

@inline _reset_output!(C, ::Type{T}) where {T} = fill!(C, one(T))
@inline _backend_zeros(::KernelAbstractions.CPU, ::Type{T}, dims...) where {T} = zeros(T, dims...)

function _backend_zeros(backend, ::Type{T}, dims...) where {T}
    _HAS_CUDA || throw(ArgumentError("non-CPU backend requires CUDA"))
    return CUDA.zeros(T, dims...)
end

@inline _synchronize(::KernelAbstractions.CPU) = nothing
@inline _synchronize(backend) = KernelAbstractions.synchronize(backend)

function _collect_large_temporaries!()
    GC.gc(true)
    _HAS_CUDA && CUDA.reclaim()
    return nothing
end

end
