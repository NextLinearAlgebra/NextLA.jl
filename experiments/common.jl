"""Shared configuration, timing, and execution code for dense-output sweeps."""
module DenseGemmCommon

using LinearAlgebra
using Printf
using KernelAbstractions
using NextLA: DenseGemmWorkspace, GEMMCompute, TF32,
              gemm_minimum_workspace_bytes, gemm_maximum_workspace_bytes,
              precision_gemm!, uncompress!
using NextLA.TLRmodule: gemm!, get_factors, grid_size, tile_size, maxrank,
                        PaddedFTLRMatrix, CompressedFTLRMatrix,
                        logical_operand, logical_operands, choose_fold, FoldRight,
                        _compressed_ftlr_rank_plan, _compressed_ftlr_row_runs,
                        _compressed_ftlr_rank, _compressed_ftlr_execution_rank,
                        _compressed_ftlr_axis_range

const _HAS_CUDA = try
    @eval import CUDA
    CUDA.functional()
catch
    false
end

include(joinpath(@__DIR__, "matrix_generation.jl"))
using .ExperimentMatrixGeneration: generate_ftlr_operands

export PrecisionConfig, RunConfig, MatrixCase, GemmTiming, GemmMetrics,
       GemmResult, run_cases, write_dense_csv, append_dense_csv,
       GEMMCompute, TF32

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
    rows_per_run::Int
    nreps::Int
    nwarmup::Int
    seed::Int
    backend::B
    check_results::Bool
    show_progress::Bool
end

function RunConfig(precisions, rows_per_run::Integer, nreps::Integer,
                   nwarmup::Integer, seed::Integer, backend;
                   check_results::Bool=false, show_progress::Bool=true)
    modes = PrecisionConfig[
        p isa PrecisionConfig ? p : _precision_config(p)
        for p in precisions
    ]
    return RunConfig(modes, Int(rows_per_run), Int(nreps), Int(nwarmup),
                     Int(seed), backend, check_results, show_progress)
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
                   cases, run::RunConfig; square=false, output_path=nothing)
    b = tile_size
    b > 0 || throw(ArgumentError("tile_size must be positive"))
    run.rows_per_run >= 1 || throw(ArgumentError("rows_per_run must be positive"))
    run.nreps >= 1 || throw(ArgumentError("nreps must be positive"))
    run.nwarmup >= 0 || throw(ArgumentError("nwarmup must be nonnegative"))
    results = GemmResult[]
    completed = _dense_completed_keys(output_path)

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
        key = _dense_result_key(experiment, case, precision, m, k, n, b,
                                ranks, lo, hi)
        if key in completed
            _announce_measurement(run, "skipping completed case=$(case.name) " *
                "precision=$(precision.name) size=($m,$k,$n)")
            continue
        end
        if run.show_progress
            println("[$experiment] case=$(case.name) precision=$(precision.name) " *
                    "size=($m,$k,$n) tile=$b ranks=$(ranks)")
            println("  generating orthonormal factors on $(_backend_name(run.backend))")
            flush(stdout)
        end

        A_tlr, B_tlr = generate_ftlr_operands(
            m, k, n, b, ranks, T; seed=run.seed + shape_index,
            backend=run.backend, format=case.format,
            rank_distribution=case.distribution, min_rank=lo, max_rank=hi)
        C = _backend_zeros(run.backend, T, m, n)
        reference = run.check_results ?
            _reference_result(run.backend, T, A_tlr, B_tlr, m, n) : nothing
        reference_norm = reference === nothing ? NaN : Float64(norm(reference))
        workspace_bytes = _row_run_workspace_bytes(
            A_tlr, B_tlr, run.rows_per_run)
        dense_flops, tlr_dense_flops, dense_tlr_flops, tlr_tlr_flops =
            _flop_counts(A_tlr, B_tlr, m, k, n, workspace_bytes)
        workspace = DenseGemmWorkspace(A_tlr, B_tlr;
            bytes=workspace_bytes)
        _announce_measurement(run, "TLR × TLR → Dense")
        tlr_tlr_ms = _time_gemm!(C, T, run) do
            gemm!(C, A_tlr, B_tlr; workspace, alpha=one(T), beta=one(T),
                  compute=precision.compute)
        end
        _report_measurement(run, "TLR × TLR → Dense", tlr_tlr_ms, tlr_tlr_flops)
        tlr_tlr_error = _relative_error(C, reference, reference_norm)
        workspace = nothing
        _collect_large_temporaries!(run.backend)

        _announce_measurement(run, "uncompressing B for TLR × Dense")
        B_dense = _uncompress(run.backend, T, B_tlr)
        workspace = DenseGemmWorkspace(A_tlr,
            _single_tlr_workspace_bytes(A_tlr, n))
        _announce_measurement(run, "TLR × Dense → Dense")
        tlr_dense_ms = _time_gemm!(C, T, run) do
            gemm!(C, A_tlr, B_dense; workspace, alpha=one(T), beta=one(T),
                  compute=precision.compute)
        end
        _report_measurement(run, "TLR × Dense → Dense", tlr_dense_ms, tlr_dense_flops)
        tlr_dense_error = _relative_error(C, reference, reference_norm)
        workspace = nothing; B_dense = nothing
        _collect_large_temporaries!(run.backend)

        _announce_measurement(run, "uncompressing A for Dense × TLR")
        A_dense = _uncompress(run.backend, T, A_tlr)
        workspace = DenseGemmWorkspace(B_tlr,
            _single_tlr_workspace_bytes(B_tlr, m))
        _announce_measurement(run, "Dense × TLR → Dense")
        dense_tlr_ms = _time_gemm!(C, T, run) do
            gemm!(C, A_dense, B_tlr; workspace, alpha=one(T), beta=one(T),
                  compute=precision.compute)
        end
        _report_measurement(run, "Dense × TLR → Dense", dense_tlr_ms, dense_tlr_flops)
        dense_tlr_error = _relative_error(C, reference, reference_norm)
        workspace = nothing; A_dense = nothing
        _collect_large_temporaries!(run.backend)

        _announce_measurement(run, "uncompressing A and B for Dense × Dense")
        A_dense = _uncompress(run.backend, T, A_tlr); A_tlr = nothing
        _collect_large_temporaries!(run.backend)
        B_dense = _uncompress(run.backend, T, B_tlr); B_tlr = nothing
        _collect_large_temporaries!(run.backend)
        _announce_measurement(run, "Dense × Dense → Dense")
        dense_dense_ms = _time_gemm!(C, T, run) do
            precision_gemm!('N', 'N', one(T), A_dense, B_dense, one(T), C,
                            precision.compute)
        end
        _report_measurement(run, "Dense × Dense → Dense", dense_dense_ms, dense_flops)
        dense_dense_error = _relative_error(C, reference, reference_norm)

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

        result = GemmResult(experiment, case.name, precision.name, T, m, k, n, b,
            ranks[1], ranks[2], case.format, case.distribution, lo, hi,
            GemmTiming(tlr_dense_ms, dense_tlr_ms, tlr_tlr_ms, dense_dense_ms), metrics)
        push!(results, result)
        isnothing(output_path) || append_dense_csv(output_path, result)
        push!(completed, key)
        A_dense = nothing; B_dense = nothing; C = nothing; reference = nothing
        _collect_large_temporaries!(run.backend)
    end
    return results
end

function _reference_result(backend, ::Type{T}, A, B, m, n) where {T}
    R = T === Float64 ? Float64 : Float32
    A_ref = _reference_operand(backend, R, T, A)
    B_ref = _reference_operand(backend, R, T, B)
    C_ref = _backend_zeros(backend, R, m, n)
    fill!(C_ref, one(R))
    mul!(C_ref, A_ref, B_ref, one(R), one(R))
    _synchronize(backend)
    A_ref = B_ref = nothing
    _collect_large_temporaries!(backend)
    return C_ref
end

function _reference_operand(backend, ::Type{T}, ::Type{T}, A) where {T}
    return _uncompress(backend, T, A)
end

function _reference_operand(backend, ::Type{R}, ::Type{T}, A) where {R,T}
    storage = _uncompress(backend, T, A)
    reference = _backend_zeros(backend, R, size(A)...)
    copyto!(reference, storage)
    _synchronize(backend)
    storage = nothing
    _collect_large_temporaries!(backend)
    return reference
end

@inline _execution_rank(A::PaddedFTLRMatrix, i, j) = maxrank(A)
@inline _execution_rank(A::CompressedFTLRMatrix, i, j) =
    _compressed_ftlr_execution_rank(A, i, j)
@inline _execution_rank(A, i, j) = size(get_factors(A, i, j)[1], 2)

function _flop_counts(A, B, m, k, n, workspace_bytes)
    mt, kt = grid_size(A); _, nt = grid_size(B)
    tlr_dense = dense_tlr = tlr_tlr = 0.0
    for i in 1:mt, l in 1:kt
        tm = tile_size(A, i, l)[1]
        tk = tile_size(A, i, l)[2]
        ra = _execution_rank(A, i, l)
        tlr_dense += 2.0 * ra * (tk * n + tm * n)
    end
    for l in 1:kt, j in 1:nt
        tk = tile_size(A, 1, l)[2]
        tn = tile_size(B, l, j)[2]
        rb = _execution_rank(B, l, j)
        dense_tlr += 2.0 * rb * (m * tk + m * tn)
    end
    tlr_tlr = _tlr_tlr_executed_flops(A, B, workspace_bytes)
    return 2.0 * m * k * n, tlr_dense, dense_tlr, tlr_tlr
end

function _tlr_tlr_executed_flops(A::PaddedFTLRMatrix,
                                  B::PaddedFTLRMatrix, workspace_bytes)
    mt, kt = grid_size(A); _, nt = grid_size(B)
    fold = choose_fold(logical_operands(logical_operand(A), logical_operand(B)))
    right = fold isa FoldRight
    rA, rB = maxrank(A), maxrank(B)
    flops = 0.0
    for i in 1:mt, l in 1:kt, j in 1:nt
        tm = tile_size(A, i, l)[1]
        tk = tile_size(A, i, l)[2]
        tn = tile_size(B, l, j)[2]
        flops += right ?
            2.0 * (tk * rA * rB + tn * rA * rB + tm * rA * tn) :
            2.0 * (tk * rA * rB + tm * rA * rB + tm * rB * tn)
    end
    return flops
end

function _tlr_tlr_executed_flops(A::CompressedFTLRMatrix,
                                  B::CompressedFTLRMatrix, workspace_bytes)
    LA, LB = logical_operand(A), logical_operand(B)
    plan = _compressed_ftlr_rank_plan(LA, LB)
    budget = min(Int(workspace_bytes), plan.profile.maximum)
    flops = 0.0
    _, qk = grid_size(LA)
    _, qn = grid_size(LB)
    N = size(LB, 2)
    for run in _compressed_ftlr_row_runs(plan.profile, budget), i in run.rows
        plan.pair_ranks[i] == 0 && continue
        mi = length(_compressed_ftlr_axis_range(LA, i, 1))
        common = 0.0
        fold_specific = 0.0
        for l in 1:qk, j in 1:qn
            ra = _compressed_ftlr_execution_rank(LA, i, l)
            rb = _compressed_ftlr_execution_rank(LB, l, j)
            (ra == 0 || rb == 0) && continue
            tk = length(_compressed_ftlr_axis_range(LA, l, 2))
            nj = length(_compressed_ftlr_axis_range(LB, j, 2))
            common += tk * ra * rb
            fold_specific += run.fold === :right ? nj * ra * rb : mi * ra * rb
        end
        terminal = if run.fold === :right
            mi * N * plan.a_k_prefix[i, end]
        else
            sum(mi * plan.output_col_widths[j] * plan.b_col_ranks[j]
                for j in 1:qn)
        end
        flops += 2.0 * (common + fold_specific + terminal)
    end
    return flops
end

function _tlr_tlr_exact_flops(A::CompressedFTLRMatrix,
                               B::CompressedFTLRMatrix, workspace_bytes)
    LA, LB = logical_operand(A), logical_operand(B)
    plan = _compressed_ftlr_rank_plan(LA, LB)
    budget = min(Int(workspace_bytes), plan.profile.maximum)
    flops = 0.0
    _, qk = grid_size(LA); _, qn = grid_size(LB)
    N = size(LB, 2)
    for run in _compressed_ftlr_row_runs(plan.profile, budget), i in run.rows
        mi = length(_compressed_ftlr_axis_range(LA, i, 1))
        common = 0.0; fold_specific = 0.0
        exact_a_total = 0
        exact_b_cols = zeros(Int, qn)
        for l in 1:qk
            exact_a_total += _compressed_ftlr_rank(LA, i, l)
            for j in 1:qn
                ra = _compressed_ftlr_rank(LA, i, l)
                rb = _compressed_ftlr_rank(LB, l, j)
                exact_b_cols[j] += rb
                (ra == 0 || rb == 0) && continue
                tk = length(_compressed_ftlr_axis_range(LA, l, 2))
                nj = length(_compressed_ftlr_axis_range(LB, j, 2))
                common += tk * ra * rb
                fold_specific += run.fold === :right ? nj * ra * rb : mi * ra * rb
            end
        end
        terminal = run.fold === :right ? mi * N * exact_a_total :
            sum(mi * plan.output_col_widths[j] * exact_b_cols[j] for j in 1:qn)
        flops += 2.0 * (common + fold_specific + terminal)
    end
    return flops
end

@inline _gflops(flops, milliseconds) = flops / (milliseconds * 1.0e6)

# The one-TLR drivers use a `rank × batch_width` temporary.  A single element
# is technically sufficient, but it degenerates into one tiny GEMM per output
# column/row.  The experiment baseline is therefore one complete output panel;
# The full panel is the intended production baseline.
@inline function _single_tlr_workspace_bytes(A, panel_length::Int)
    maxrank(A) * panel_length * sizeof(eltype(A))
end

function _row_run_workspace_bytes(A::PaddedFTLRMatrix,
                                  B::PaddedFTLRMatrix, rows::Int)
    minimum = gemm_minimum_workspace_bytes(A, B)
    maximum = gemm_maximum_workspace_bytes(A, B)
    qm = grid_size(A)[1]
    target = cld(min(rows, qm) * maximum, qm)
    aligned = cld(target, sizeof(eltype(A))) * sizeof(eltype(A))
    return clamp(aligned, minimum, maximum)
end

function _row_run_workspace_bytes(A::CompressedFTLRMatrix,
                                  B::CompressedFTLRMatrix, rows::Int)
    profile = _compressed_ftlr_rank_plan(
        logical_operand(A), logical_operand(B)).profile
    width = min(rows, length(profile.row_bytes))
    best = 0
    for first in 1:(length(profile.row_bytes) - width + 1)
        last = first + width - 1
        right = profile.right_byte_prefix === nothing ? typemax(Int) :
            profile.right_byte_prefix[last + 1] - profile.right_byte_prefix[first]
        left = profile.left_byte_prefix === nothing ? typemax(Int) :
            profile.left_byte_prefix[last + 1] - profile.left_byte_prefix[first]
        best = max(best, min(right, left))
    end
    return clamp(best, profile.minimum, profile.maximum)
end

function _announce_measurement(run::RunConfig, label)
    run.show_progress || return nothing
    println("  $label")
    flush(stdout)
    return nothing
end

function _report_measurement(run::RunConfig, label, milliseconds, flops)
    run.show_progress || return nothing
    @printf("    %-24s %10.3f ms  %12.2f executed GFLOP/s\n",
            label, milliseconds, _gflops(flops, milliseconds))
    flush(stdout)
    return nothing
end

_relative_error(_, ::Nothing, _) = NaN

function _relative_error(C, reference, reference_norm)
    difference = similar(reference)
    copyto!(difference, C)
    difference .-= reference
    error = norm(difference) / max(reference_norm, eps(eltype(reference)))
    difference = nothing
    return Float64(error)
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
_backend_name(::KernelAbstractions.CPU) = "the CPU"
_backend_name(_) = "the GPU"

function _collect_large_temporaries!(backend)
    GC.gc(true)
    _HAS_CUDA && !(backend isa KernelAbstractions.CPU) && CUDA.reclaim()
    return nothing
end

const _DENSE_CSV_HEADER = [
    "experiment", "case", "precision", "dtype", "m", "k", "n", "tile_size",
    "rank_A", "rank_B", "format", "rank_distribution", "min_rank", "max_rank",
    "tlr_dense_ms", "dense_tlr_ms", "tlr_tlr_ms", "dense_dense_ms",
    "dense_gflops", "tlr_dense_gflops", "dense_tlr_gflops", "tlr_tlr_gflops",
    "tlr_dense_speedup", "dense_tlr_speedup", "tlr_tlr_speedup",
    "tlr_dense_efficiency", "dense_tlr_efficiency", "tlr_tlr_efficiency",
    "tlr_dense_rel_error", "dense_tlr_rel_error", "tlr_tlr_rel_error",
    "dense_dense_rel_error",
]

_csv_value(x) = begin
    s = string(x)
    occursin(',', s) || occursin('"', s) ?
        "\"" * replace(s, '"' => "\"\"") * "\"" : s
end

_dense_row(r::GemmResult) =
    (r.experiment, r.case, r.precision, r.dtype, r.m, r.k, r.n, r.tile_size,
     r.rank_A, r.rank_B, r.format, r.rank_distribution, r.min_rank, r.max_rank,
     r.timing.tlr_dense_ms, r.timing.dense_tlr_ms, r.timing.tlr_tlr_ms,
     r.timing.dense_dense_ms, r.metrics.dense_gflops,
     r.metrics.tlr_dense_gflops, r.metrics.dense_tlr_gflops,
     r.metrics.tlr_tlr_gflops, r.metrics.tlr_dense_speedup,
     r.metrics.dense_tlr_speedup, r.metrics.tlr_tlr_speedup,
     r.metrics.tlr_dense_efficiency, r.metrics.dense_tlr_efficiency,
     r.metrics.tlr_tlr_efficiency, r.metrics.tlr_dense_rel_error,
     r.metrics.dense_tlr_rel_error, r.metrics.tlr_tlr_rel_error,
     r.metrics.dense_dense_rel_error)

function write_dense_csv(path, results)
    open(path, "w") do io
        println(io, join(_DENSE_CSV_HEADER, ','))
        for r in results
            println(io, join(_csv_value.(_dense_row(r)), ','))
        end
    end
    path
end

function append_dense_csv(path, result::GemmResult)
    new_file = _prepare_csv_append(path)
    open(path, "a") do io
        new_file && println(io, join(_DENSE_CSV_HEADER, ','))
        println(io, join(_csv_value.(_dense_row(result)), ','))
        flush(io)
    end
    return path
end

function _prepare_csv_append(path)
    mkpath(dirname(path))
    new_file = !isfile(path) || filesize(path) == 0
    if !new_file
        open(path, "r+") do io
            seekend(io)
            seek(io, position(io) - 1)
            read(io, UInt8) == UInt8('\n') || (seekend(io); println(io))
        end
    end
    return new_file
end

_dense_result_key(experiment, case, precision, m, k, n, b, ranks, lo, hi) =
    string.((experiment, case.name, precision.name, precision.storage_type,
             m, k, n, b, ranks[1], ranks[2], case.format,
             case.distribution, lo, hi))

function _dense_completed_keys(path)
    keys = Set{NTuple{14,String}}()
    (isnothing(path) || !isfile(path)) && return keys
    for line in Iterators.drop(eachline(path), 1)
        fields = split(line, ',')
        length(fields) == length(_DENSE_CSV_HEADER) || continue
        push!(keys, Tuple(fields[1:14]))
    end
    return keys
end

end
