"""Padded-FTLR-output GEMM experiments and output-quality measurements."""
module PaddedFTLROutputExperiment

using LinearAlgebra
using Printf
using KernelAbstractions
using NextLA: DenseGemmWorkspace, TLRGemmWorkspace, GEMMCompute,
              alloc_workspace, compress!, precision_gemm!, uncompress!,
              tlr_gemm_maximum_workspace_bytes
using NextLA.TLRmodule: gemm!, get_factors, grid_size, maxrank, tile_size
using Main.DenseGemmCommon: PrecisionConfig, _row_run_workspace_bytes,
                            _prepare_csv_append
using Main.DenseGemmCommon.ExperimentMatrixGeneration:
    allocate_tlr_matrix, generate_tlr_operands

const _HAS_CUDA = try
    @eval import CUDA
    CUDA.functional()
catch
    false
end

export PaddedFTLROutputRunConfig, PaddedFTLROutputStrongScalingConfig,
       PaddedFTLROutputOverlapConfig, PaddedFTLROutputTiming,
       PaddedFTLROutputMetrics, PaddedFTLROutputResult,
       padded_ftlr_output_strong_scaling,
       padded_ftlr_output_overlap_sweep, write_padded_ftlr_output_csv,
       append_padded_ftlr_output_csv

struct PaddedFTLROutputRunConfig{B}
    precisions::Vector{PrecisionConfig}
    rows_per_run::Int
    nreps::Int
    nwarmup::Int
    seed::Int
    backend::B
    block::Int
    tol::Float64
    rel::Bool
    show_progress::Bool
end

function PaddedFTLROutputRunConfig(precisions, rows_per_run, nreps, nwarmup,
                                   seed, backend; block=32, tol=0.0, rel=false,
                                   show_progress=true)
    modes = PrecisionConfig[
        p isa PrecisionConfig ? p :
        PrecisionConfig(Symbol(p), p, GEMMCompute{p}()) for p in precisions
    ]
    return PaddedFTLROutputRunConfig(
        modes, Int(rows_per_run), Int(nreps), Int(nwarmup), Int(seed), backend,
        Int(block), Float64(tol), Bool(rel), Bool(show_progress))
end

struct PaddedFTLROutputStrongScalingConfig{B}
    sizes::Vector{Int}
    tile_size::Int
    ranks::NTuple{2,Int}
    output_rank::Int
    run::PaddedFTLROutputRunConfig{B}
end

PaddedFTLROutputStrongScalingConfig(sizes, tile_size, ranks, output_rank,
                                    run::PaddedFTLROutputRunConfig) =
    PaddedFTLROutputStrongScalingConfig(
        Int.(sizes), Int(tile_size), Int.(ranks), Int(output_rank), run)

struct PaddedFTLROutputOverlapConfig{B}
    matrix_size::Int
    tile_size::Int
    ranks::NTuple{2,Int}
    output_rank::Int
    shared_ranks::Vector{Int}
    run::PaddedFTLROutputRunConfig{B}
end

PaddedFTLROutputOverlapConfig(matrix_size, tile_size, ranks, output_rank,
                              shared_ranks,
                              run::PaddedFTLROutputRunConfig) =
    PaddedFTLROutputOverlapConfig(
        Int(matrix_size), Int(tile_size), Int.(ranks), Int(output_rank),
        Int.(shared_ranks), run)

struct PaddedFTLROutputTiming
    tlr_tlr_ms::Float64
    dense_compress_ms::Float64
    dense_dense_ms::Float64
    tlr_tlr_rel_fro_error::Float64
    dense_compress_rel_fro_error::Float64
end

struct PaddedFTLROutputMetrics
    dense_gflops::Float64
    tlr_tlr_gflops::Float64
    dense_compress_gflops::Float64
    tlr_tlr_speedup::Float64
    dense_compress_speedup::Float64
    tlr_tlr_efficiency::Float64
    dense_compress_efficiency::Float64
end

struct PaddedFTLROutputResult
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
    timing::PaddedFTLROutputTiming
    metrics::PaddedFTLROutputMetrics
end

padded_ftlr_output_strong_scaling(
    config::PaddedFTLROutputStrongScalingConfig; output_path=nothing) =
    _run_cases(:padded_ftlr_output_strong_scaling, config.sizes,
               config.tile_size, config.ranks, config.output_rank, 0,
               config.run; output_path)

function padded_ftlr_output_overlap_sweep(
    config::PaddedFTLROutputOverlapConfig; output_path=nothing)
    results = PaddedFTLROutputResult[]
    for shared in config.shared_ranks
        0 <= shared <= min(config.ranks...) ||
            throw(ArgumentError("shared_rank must be in 0:min(ranks)"))
        append!(results, _run_cases(
            :padded_ftlr_output_overlap_sweep, [config.matrix_size],
            config.tile_size,
            config.ranks, config.output_rank, shared, config.run; output_path))
    end
    return results
end

function _run_cases(experiment, sizes, b, ranks, output_rank, shared_rank, run;
                    output_path=nothing)
    rA, rB = ranks
    b > 0 && 0 < rA < b && 0 < rB < b && 0 < output_rank < b ||
        throw(ArgumentError("tile size and ranks must satisfy 0 < rank < tile_size"))
    run.rows_per_run >= 1 || throw(ArgumentError("rows_per_run must be positive"))
    run.block >= 1 || throw(ArgumentError("block must be positive"))
    run.nreps >= 1 || throw(ArgumentError("nreps must be positive"))
    run.nwarmup >= 0 || throw(ArgumentError("nwarmup must be nonnegative"))

    results = PaddedFTLROutputResult[]
    completed = _padded_output_completed_keys(output_path)
    for precision in run.precisions, (case_index, n) in enumerate(sizes)
        n % b == 0 || throw(ArgumentError("size=$n must be divisible by tile_size=$b"))
        T = precision.storage_type
        key = string.((experiment, precision.name, T, n, n, n, b, rA, rB,
                       output_rank, shared_rank))
        if key in completed
            _announce(run, "  skipping completed precision=$(precision.name) size=$n")
            continue
        end
        seed = run.seed + case_index + 1000shared_rank
        _announce(run, "[$experiment] precision=$(precision.name) size=$n " *
                       "tile=$b ranks=$ranks output_rank=$output_rank shared=$shared_rank")
        _announce(run, "  generating orthonormal factors")
        A, B = generate_tlr_operands(
            n, n, n, b, ranks, T; seed, shared_rank, backend=run.backend)
        tlr_flops = _tlr_product_flops(A, B)
        dense_flops = 2.0n^3

        C0 = allocate_tlr_matrix(n, n, b, output_rank, T; backend=run.backend)
        _fill_initial_output!(C0, T, run.backend)
        C_tlr = allocate_tlr_matrix(n, n, b, output_rank, T; backend=run.backend)
        direct_workspace = TLRGemmWorkspace(C_tlr, A, B;
            bytes=tlr_gemm_maximum_workspace_bytes(
                C_tlr, A, B; block=run.block), block=run.block)
        _announce(run, "  TLR × TLR → TLR")
        direct_ms = _time!(
            () -> gemm!(C_tlr, A, B; workspace=direct_workspace,
                        alpha=one(T), beta=one(T), tol=run.tol, rel=run.rel,
                        block=run.block, compute=precision.compute),
            () -> _reset_tlr!(C_tlr, C0, run.backend), run)
        _report(run, "TLR × TLR → TLR", direct_ms, tlr_flops)
        direct_workspace = nothing
        _collect!(run.backend)

        C_dense = _uncompress(run.backend, T, C0)
        C_compressed = allocate_tlr_matrix(
            n, n, b, output_rank, T; backend=run.backend)
        dense_workspace = DenseGemmWorkspace(
            A, B; bytes=_row_run_workspace_bytes(A, B, run.rows_per_run))
        compress_workspace = alloc_workspace(C_compressed)
        _announce(run, "  TLR × TLR → Dense + compression")
        dense_compress_ms = _time!(
            () -> begin
                gemm!(C_dense, A, B; workspace=dense_workspace,
                      alpha=one(T), beta=one(T), compute=precision.compute)
                compress!(C_compressed, C_dense, compress_workspace;
                          tol=run.tol, rel=run.rel)
            end,
            () -> fill!(C_dense, one(T)), run)
        _report(run, "Dense + compression", dense_compress_ms, tlr_flops)
        dense_workspace = compress_workspace = C_dense = nothing
        _collect!(run.backend)

        _announce(run, "  uncompressing operands for Dense × Dense")
        A_dense = _uncompress(run.backend, T, A)
        B_dense = _uncompress(run.backend, T, B)
        C_reference = _uncompress(run.backend, T, C0)
        A = B = C0 = nothing
        _collect!(run.backend)
        _announce(run, "  Dense × Dense → Dense")
        dense_ms = _time!(
            () -> precision_gemm!('N', 'N', one(T), A_dense, B_dense,
                                  one(T), C_reference, precision.compute),
            () -> fill!(C_reference, one(T)), run)
        _report(run, "Dense × Dense → Dense", dense_ms, dense_flops)

        reference_norm = Float64(norm(C_reference))
        direct_error = _relative_error(
            C_tlr, C_reference, reference_norm, run.backend)
        compressed_error = _relative_error(
            C_compressed, C_reference, reference_norm, run.backend)
        dense_gflops = _gflops(dense_flops, dense_ms)
        direct_gflops = _gflops(tlr_flops, direct_ms)
        compressed_gflops = _gflops(tlr_flops, dense_compress_ms)
        metrics = PaddedFTLROutputMetrics(
            dense_gflops, direct_gflops, compressed_gflops,
            dense_ms / direct_ms, dense_ms / dense_compress_ms,
            direct_gflops / dense_gflops, compressed_gflops / dense_gflops)
        result = PaddedFTLROutputResult(
            experiment, precision.name, T, n, n, n, b, rA, rB, output_rank,
            shared_rank,
            PaddedFTLROutputTiming(direct_ms, dense_compress_ms, dense_ms,
                                   direct_error, compressed_error), metrics)
        push!(results, result)
        isnothing(output_path) || append_padded_ftlr_output_csv(output_path, result)
        push!(completed, key)

        A_dense = B_dense = C_reference = C_tlr = C_compressed = nothing
        _collect!(run.backend)
    end
    return results
end

function _tlr_product_flops(A, B)
    mt, kt = grid_size(A)
    nt = grid_size(B)[2]
    flops = 0.0
    for i in 1:mt, l in 1:kt, j in 1:nt
        tm, tk = tile_size(A, i, l)
        tn = tile_size(B, l, j)[2]
        ra = size(get_factors(A, i, l)[1], 2)
        rb = size(get_factors(B, l, j)[1], 2)
        flops += 2.0 * (tk * ra * rb + tm * ra * rb + tm * rb * tn)
    end
    return flops
end

@inline _gflops(flops, milliseconds) = flops / (milliseconds * 1.0e6)

function _time!(f, reset, run)
    for _ in 1:run.nwarmup
        reset(); _synchronize(run.backend); f(); _synchronize(run.backend)
    end
    best = Inf
    for _ in 1:run.nreps
        reset(); _synchronize(run.backend)
        start = time_ns()
        f(); _synchronize(run.backend)
        best = min(best, (time_ns() - start) / 1.0e6)
    end
    return best
end

const _TLR_FIELDS = (
    :int_U, :int_V, :right_U, :right_V, :bottom_U, :bottom_V,
    :corner_U, :corner_V, :ranks, :resid)

function _reset_tlr!(C, template, backend)
    foreach(field -> copyto!(getproperty(C, field), getproperty(template, field)),
            _TLR_FIELDS)
    _synchronize(backend)
    return C
end

function _fill_initial_output!(C, ::Type{T}, backend) where {T}
    C.ranks .= min(1, maxrank(C))
    foreach(field -> fill!(getproperty(C, field), zero(T)), _TLR_FIELDS[1:8])
    if maxrank(C) > 0
        for i in 1:grid_size(C)[1], j in 1:grid_size(C)[2]
            U, V = get_factors(C, i, j)
            U[:, 1] .= one(T)
            V[:, 1] .= one(T)
        end
    end
    _synchronize(backend)
    return C
end

function _relative_error(A_tlr, reference, reference_norm, backend)
    difference = similar(reference)
    uncompress!(difference, A_tlr)
    difference .-= reference
    _synchronize(backend)
    return Float64(norm(difference) /
                   max(reference_norm, eps(eltype(reference))))
end

function _uncompress(backend, ::Type{T}, A) where {T}
    dense = backend isa KernelAbstractions.CPU ?
        zeros(T, size(A)...) :
        (_HAS_CUDA ? CUDA.zeros(T, size(A)...) :
         throw(ArgumentError("non-CPU backend requires CUDA")))
    uncompress!(dense, A)
    _synchronize(backend)
    return dense
end

function _announce(run, message)
    run.show_progress || return nothing
    println(message)
    flush(stdout)
end

function _report(run, label, milliseconds, flops)
    run.show_progress || return nothing
    @printf("    %-28s %10.3f ms  %12.2f effective GFLOP/s\n",
            label, milliseconds, _gflops(flops, milliseconds))
    flush(stdout)
end

_synchronize(::KernelAbstractions.CPU) = nothing
_synchronize(backend) = KernelAbstractions.synchronize(backend)

function _collect!(backend)
    GC.gc(true)
    _HAS_CUDA && !(backend isa KernelAbstractions.CPU) && CUDA.reclaim()
    return nothing
end

function write_padded_ftlr_output_csv(path, results)
    header = fieldnames(PaddedFTLROutputResult)[1:11]
    timing = fieldnames(PaddedFTLROutputTiming)
    metrics = fieldnames(PaddedFTLROutputMetrics)
    open(path, "w") do io
        println(io, join((header..., timing..., metrics...), ','))
        for r in results
            row = (getfield.(Ref(r), header)...,
                   getfield.(Ref(r.timing), timing)...,
                   getfield.(Ref(r.metrics), metrics)...)
            println(io, join(row, ','))
        end
    end
    return path
end

function append_padded_ftlr_output_csv(path, result::PaddedFTLROutputResult)
    header = fieldnames(PaddedFTLROutputResult)[1:11]
    timing = fieldnames(PaddedFTLROutputTiming)
    metrics = fieldnames(PaddedFTLROutputMetrics)
    row = (getfield.(Ref(result), header)...,
           getfield.(Ref(result.timing), timing)...,
           getfield.(Ref(result.metrics), metrics)...)
    new_file = _prepare_csv_append(path)
    open(path, "a") do io
        new_file && println(io, join((header..., timing..., metrics...), ','))
        println(io, join(row, ','))
        flush(io)
    end
    return path
end

function _padded_output_completed_keys(path)
    keys = Set{NTuple{11,String}}()
    (isnothing(path) || !isfile(path)) && return keys
    expected = 11 + fieldcount(PaddedFTLROutputTiming) +
               fieldcount(PaddedFTLROutputMetrics)
    for line in Iterators.drop(eachline(path), 1)
        fields = split(line, ',')
        length(fields) == expected || continue
        push!(keys, Tuple(fields[1:11]))
    end
    return keys
end

end
