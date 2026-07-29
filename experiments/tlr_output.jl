"""Experiments for TLR-output GEMM and output-quality measurements."""
module TLROutputExperiment

using LinearAlgebra
using KernelAbstractions
using NextLA: DenseGemmWorkspace, TLRGemmWorkspace, GEMMCompute, TF32,
              alloc_workspace, compress!, gemm_minimum_workspace_bytes,
              tlr_gemm_minimum_workspace_bytes, precision_gemm!, uncompress!
using NextLA.TLRmodule: gemm!, grid_size, get_factors, tile_size

const _HAS_CUDA = try
    @eval import CUDA
    CUDA.functional()
catch
    false
end

include(joinpath(@__DIR__, "matrix_generation.jl"))
using .ExperimentMatrixGeneration: generate_tlr_matrix, generate_tlr_operands

if !isdefined(Main, :DenseGemmCommon)
    include(joinpath(@__DIR__, "common.jl"))
end
using Main.DenseGemmCommon: PrecisionConfig

export TLROutputRunConfig, TLROutputStrongScalingConfig,
       TLROutputOverlapConfig, TLROutputTiming, TLROutputMetrics, TLROutputResult
export tlr_output_strong_scaling, tlr_output_overlap_sweep

struct TLROutputRunConfig{B}
    precisions::Vector{PrecisionConfig}
    workspace_factor::Int
    nreps::Int
    nwarmup::Int
    seed::Int
    backend::B
    block::Int
    tol::Float64
    rel::Bool
end

function TLROutputRunConfig(precisions, workspace_factor, nreps, nwarmup, seed,
                            backend; block=32, tol=0.0, rel=false)
    return TLROutputRunConfig(
        PrecisionConfig[p isa PrecisionConfig ? p :
                       PrecisionConfig(Symbol(p), p, GEMMCompute{p}())
                       for p in precisions],
        Int(workspace_factor), Int(nreps), Int(nwarmup),
        Int(seed), backend, Int(block), Float64(tol), rel)
end

struct TLROutputStrongScalingConfig{B}
    sizes::Vector{Int}
    tile_size::Int
    ranks::NTuple{2,Int}
    output_rank::Int
    run::TLROutputRunConfig{B}
end

function TLROutputStrongScalingConfig(sizes, tile_size, ranks, output_rank,
                                      run::TLROutputRunConfig)
    return TLROutputStrongScalingConfig(Int.(sizes), Int(tile_size), Int.(ranks),
                                        Int(output_rank), run)
end

struct TLROutputOverlapConfig{B}
    matrix_size::Int
    tile_size::Int
    ranks::NTuple{2,Int}
    output_rank::Int
    shared_ranks::Vector{Int}
    run::TLROutputRunConfig{B}
end

function TLROutputOverlapConfig(matrix_size, tile_size, ranks, output_rank,
                                shared_ranks, run::TLROutputRunConfig)
    return TLROutputOverlapConfig(Int(matrix_size), Int(tile_size), Int.(ranks),
                                  Int(output_rank), Int.(shared_ranks), run)
end

struct TLROutputTiming
    tlr_tlr_ms::Float64
    dense_compress_ms::Float64
    dense_dense_ms::Float64
    tlr_tlr_rel_fro_error::Float64
    dense_compress_rel_fro_error::Float64
end

struct TLROutputMetrics
    dense_gflops::Float64
    tlr_tlr_gflops::Float64
    dense_compress_gflops::Float64
    tlr_tlr_speedup::Float64
    dense_compress_speedup::Float64
    tlr_tlr_efficiency::Float64
    dense_compress_efficiency::Float64
end

struct TLROutputResult
    experiment::Symbol
    precision::Symbol
    dtype::DataType
    m::Int
    k::Int
    n::Int
    tile_size::Int
    rank_A::Int
    rank_B::Int
    output_rank::Int
    shared_rank::Int
    timing::TLROutputTiming
    metrics::TLROutputMetrics
end

function tlr_output_strong_scaling(config::TLROutputStrongScalingConfig)
    return _run_cases(:tlr_output_strong_scaling, config.sizes, config.tile_size,
                      config.ranks, config.output_rank, 0, config.run)
end

function tlr_output_overlap_sweep(config::TLROutputOverlapConfig)
    results = TLROutputResult[]
    for shared_rank in config.shared_ranks
        shared_rank <= min(config.ranks...) ||
            throw(ArgumentError("shared_rank must not exceed either operand rank"))
        append!(results, _run_cases(
            :tlr_output_overlap_sweep,
            [config.matrix_size], config.tile_size, config.ranks,
            config.output_rank, shared_rank, config.run))
    end
    return results
end

function _run_cases(experiment, sizes, tile_size, ranks, output_rank, shared_rank, run)
    b = Int(tile_size)
    rA, rB = ranks
    b > 0 && rA < b && rB < b && output_rank < b ||
        throw(ArgumentError("all ranks must be smaller than tile_size"))
    run.block >= 1 || throw(ArgumentError("block must be positive"))
    run.workspace_factor >= 1 || throw(ArgumentError("workspace_factor must be positive"))
    run.nreps >= 1 || throw(ArgumentError("nreps must be positive"))
    run.nwarmup >= 0 || throw(ArgumentError("nwarmup must be nonnegative"))

    results = TLROutputResult[]
    for precision in run.precisions, (case_index, n) in enumerate(sizes)
        T = precision.storage_type
        n % b == 0 || throw(ArgumentError("size=$n must be divisible by tile_size=$b"))
        seed = run.seed + case_index + 1000 * shared_rank

        A, B = generate_tlr_operands(
            n, n, n, b, ranks, T; seed, shared_rank, backend=run.backend)
        tlr_flops = _tlr_tlr_flops(A, B, n, b)
        dense_flops = 2.0 * n^3
        C_template = generate_tlr_matrix(
            n, n, b, output_rank, T; seed=seed + 1, backend=run.backend)
        _fill_constant_tlr!(C_template, T, run.backend)
        C_direct = generate_tlr_matrix(
            n, n, b, output_rank, T; seed=seed + 2, backend=run.backend)

        direct_workspace = TLRGemmWorkspace(
            C_direct, A, B;
            bytes=run.workspace_factor * tlr_gemm_minimum_workspace_bytes(
                C_direct, A, B; block=run.block), block=run.block)
        tlr_tlr_ms = _time!(
            () -> gemm!(C_direct, A, B; workspace=direct_workspace,
                        alpha=one(T), beta=one(T), tol=run.tol, rel=run.rel,
                        block=run.block, compute=precision.compute),
            () -> _reset_tlr!(C_direct, C_template, run.backend), run)
        direct_workspace = nothing
        _collect!()

        C_dense = _uncompress(run.backend, T, C_template)
        C_compressed = generate_tlr_matrix(
            n, n, b, output_rank, T; seed=seed + 3, backend=run.backend)
        dense_workspace = DenseGemmWorkspace(
            A, B; bytes=run.workspace_factor * gemm_minimum_workspace_bytes(A, B))
        compress_workspace = alloc_workspace(C_compressed)
        dense_compress_ms = _time!(
            () -> begin
                gemm!(C_dense, A, B; workspace=dense_workspace,
                      alpha=one(T), beta=one(T), compute=precision.compute)
                compress!(C_compressed, C_dense, compress_workspace;
                          tol=run.tol, rel=run.rel)
            end,
            () -> _reset_dense!(C_dense, T), run)
        dense_workspace = nothing
        compress_workspace = nothing
        C_dense = nothing
        _collect!()

        # Recreate dense operands from the same deterministic seed for the
        # dense baseline and the common reference result.  This avoids holding
        # dense and TLR versions of the operands simultaneously for longer than
        # necessary.
        A, B = nothing, nothing
        _collect!()
        A, B = generate_tlr_operands(
            n, n, n, b, ranks, T; seed, shared_rank, backend=run.backend)
        A_dense = _uncompress(run.backend, T, A)
        B_dense = _uncompress(run.backend, T, B)
        C_reference = _uncompress(run.backend, T, C_template)
        A, B, C_template = nothing, nothing, nothing
        _collect!()

        dense_dense_ms = _time!(
            () -> precision_gemm!('N', 'N', one(T), A_dense, B_dense,
                                  one(T), C_reference, precision.compute),
            () -> _reset_dense!(C_reference, T), run)
        tlr_error = _relative_error(C_direct, C_reference, run.backend, T)
        compressed_error = _relative_error(C_compressed, C_reference, run.backend, T)
        dense_gflops = dense_flops / (dense_dense_ms * 1.0e6)
        tlr_gflops = tlr_flops / (tlr_tlr_ms * 1.0e6)
        dense_compress_gflops = dense_flops / (dense_compress_ms * 1.0e6)
        metrics = TLROutputMetrics(
            dense_gflops, tlr_gflops, dense_compress_gflops,
            dense_dense_ms / tlr_tlr_ms, dense_dense_ms / dense_compress_ms,
            tlr_gflops / dense_gflops, dense_compress_gflops / dense_gflops)

        push!(results, TLROutputResult(
            experiment, precision.name, T, n, n, n, b, rA, rB, output_rank, shared_rank,
            TLROutputTiming(tlr_tlr_ms, dense_compress_ms, dense_dense_ms,
                            tlr_error, compressed_error), metrics))

        A_dense, B_dense, C_reference = nothing, nothing, nothing
        C_direct, C_compressed = nothing, nothing
        _collect!()
    end
    return results
end

function _tlr_tlr_flops(A, B, n, b)
    mt, kt = grid_size(A); _, nt = grid_size(B)
    flops = 0.0
    for i in 1:mt, l in 1:kt, j in 1:nt
        tm, tk = tile_size(A, i, l)
        _, tn = tile_size(B, l, j)
        ra = size(get_factors(A, i, l)[1], 2)
        rb = size(get_factors(B, l, j)[1], 2)
        flops += 2.0 * (tk * ra * rb + tm * ra * rb + tm * rb * tn)
    end
    flops
end

function _time!(f, reset, run::TLROutputRunConfig{B}) where {B}
    for _ in 1:run.nwarmup
        reset()
        _synchronize(run.backend)
        f()
        _synchronize(run.backend)
    end
    best = Inf
    for _ in 1:run.nreps
        reset()
        _synchronize(run.backend)
        start = time_ns()
        f()
        _synchronize(run.backend)
        best = min(best, (time_ns() - start) / 1.0e6)
    end
    return best
end

function _reset_tlr!(C, template, backend)
    for field in (:int_U, :int_V, :right_U, :right_V,
                  :bottom_U, :bottom_V, :corner_U, :corner_V, :ranks, :resid)
        copyto!(getproperty(C, field), getproperty(template, field))
    end
    _synchronize(backend)
    return C
end

@inline _reset_dense!(C, ::Type{T}) where {T} = fill!(C, one(T))

function _fill_constant_tlr!(C, ::Type{T}, backend) where {T}
    C.ranks .= C.maxrank
    for field in (:int_U, :int_V, :right_U, :right_V,
                  :bottom_U, :bottom_V, :corner_U, :corner_V)
        fill!(getproperty(C, field), zero(T))
    end
    for i in 1:grid_size(C)[1], j in 1:grid_size(C)[2]
        U, V = get_factors(C, i, j)
        U[:, 1] .= one(T)
        V[:, 1] .= one(T)
    end
    _synchronize(backend)
    return C
end

function _relative_error(A_tlr, reference, backend, ::Type{T}) where {T}
    got = similar(reference)
    uncompress!(got, A_tlr)
    _synchronize(backend)
    return norm(got - reference) / max(norm(reference), eps(real(T)))
end

function _uncompress(backend, ::Type{T}, A) where {T}
    dense = if backend isa KernelAbstractions.CPU
        zeros(T, size(A)...)
    elseif _HAS_CUDA
        CUDA.zeros(T, size(A)...)
    else
        throw(ArgumentError("non-CPU backend requires CUDA"))
    end
    uncompress!(dense, A)
    _synchronize(backend)
    return dense
end

@inline _synchronize(::KernelAbstractions.CPU) = nothing
@inline _synchronize(backend) = KernelAbstractions.synchronize(backend)

function _collect!()
    GC.gc(true)
    _HAS_CUDA && CUDA.reclaim()
    return nothing
end

end
