# Compare dense GEMM, TLR GEMM followed by compression, and direct TLR-output GEMM.
#
# All input tiles are populated with rank exactly maxrank.  C has capacity
# min(maxrank_A, maxrank_B).  The dense-plus-compression time includes both
# the dense-output TLR GEMM and compress!; reconstruction errors are measured
# after the timed section with uncompress!.

using LinearAlgebra
using Printf
using Random
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

const CONFIG = load_config(; default_benchmark=Symbol("tlr-output"))
const NREPS = CONFIG.reps
const WARMUP = CONFIG.warmup
const BLOCK = CONFIG.tile_size
const OUTPUT = output_path(CONFIG, Symbol("tlr-output"))
const SEED = CONFIG.seed

const CSV_COLUMNS = (
    "case_id", "m", "k", "n", "bm", "bk", "bn", "rank_A", "rank_B", "rank_C", "precision",
    "nreps", "dense_ms", "tlr_dense_compress_ms", "tlr_output_ms",
    "dense_plus_compress_speedup", "tlr_output_speedup",
    "error_dense_compress_abs", "error_dense_compress_rel",
    "error_tlr_output_abs", "error_tlr_output_rel",
)

@inline backend_sync(backend) =
    backend isa KernelAbstractions.CPU ? nothing : KernelAbstractions.synchronize(backend)

@inline backend_zeros(backend, ::Type{T}, dims...) where {T} =
    backend isa KernelAbstractions.CPU ? zeros(T, dims...) : CUDA.zeros(T, dims...)

function release_backend_memory!()
    GC.gc(true)
    HAS_CUDA && CUDA.reclaim()
    return nothing
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

function make_full_rank_tlr(backend, ::Type{T}, m, n, bm, bn, rank, seed) where {T}
    m % bm == 0 || error("m=$m must be divisible by tile size bm=$bm")
    n % bn == 0 || error("n=$n must be divisible by tile size bn=$bn")
    X = TLRM.TLRMatrix(
        backend, T, m, n, (bm, bn), rank; tile_order=TLRM.TileRowMajor)
    Random.seed!(seed)
    randn!(X.int_U)
    randn!(X.int_V)
    fill!(X.ranks, rank)
    backend_sync(backend)
    return X
end

function uncompressed(X, backend, ::Type{T}) where {T}
    dense = backend_zeros(backend, T, size(X)...)
    TLRM.uncompress!(dense, X)
    backend_sync(backend)
    return dense
end

function reconstruction_error(reference, X, backend, ::Type{T}) where {T}
    got = uncompressed(X, backend, T)
    reference_host = Array(reference)
    got_host = Array(got)
    absolute = norm(got_host - reference_host)
    relative = absolute / max(norm(reference_host), eps(Float64))
    return absolute, relative
end

function write_header_if_needed(path)
    expected = join(CSV_COLUMNS, ',')
    if isfile(path) && filesize(path) > 0
        open(path, "r") do io
            readline(io) == expected || error("CSV schema mismatch in $path")
        end
    else
        open(path, "w") do io
            println(io, expected)
        end
    end
end

function completed_cases(path)
    isfile(path) || return Set{String}()
    ids = Set{String}()
    open(path, "r") do io
        eof(io) && return ids
        readline(io)
        for line in eachline(io)
            isempty(line) || push!(ids, first(split(line, ',')))
        end
    end
    return ids
end

function benchmark_case(backend, ::Type{T}, m, k, n, bm, bk, bn,
                        rank_A, rank_B) where {T}
    rank_C = min(rank_A, rank_B)
    A = make_full_rank_tlr(backend, T, m, k, bm, bk, rank_A, SEED + 11)
    B = make_full_rank_tlr(backend, T, k, n, bk, bn, rank_B, SEED + 29)
    A_dense = uncompressed(A, backend, T)
    B_dense = uncompressed(B, backend, T)
    reference = backend_zeros(backend, T, m, n)
    oneT, zeroT = one(T), zero(T)
    dense_gemm = () -> NextLA._gemm_compute!(
        NextLA.GEMMCompute{T}(), 'N', 'N', oneT, A_dense, B_dense, zeroT, reference)
    dense_ms = best_time_ms(dense_gemm, backend)

    C_dense = backend_zeros(backend, T, m, n)
    C_compressed = TLRM.TLRMatrix(
        backend, T, m, n, (bm, bn), rank_C; tile_order=TLRM.TileRowMajor)
    dense_ws = TLRM.DenseGemmWorkspace(
        A, B; bytes=TLRM.gemm_maximum_workspace_bytes(A, B))
    compress_ws = TLRM.alloc_workspace(C_compressed)
    dense_plus_compress = () -> begin
        TLRM.gemm!(C_dense, A, B; alpha=oneT, beta=zeroT, workspace=dense_ws)
        TLRM.compress!(C_compressed, C_dense, compress_ws; tol=0.0, rel=false)
    end
    dense_compress_ms = best_time_ms(dense_plus_compress, backend)

    C_tlr = TLRM.TLRMatrix(
        backend, T, m, n, (bm, bn), rank_C; tile_order=TLRM.TileRowMajor)
    tlr_ws = TLRM.TLRGemmWorkspace(C_tlr, A, B; block=BLOCK)
    tlr_output = () -> TLRM.gemm!(
        C_tlr, A, B; alpha=oneT, beta=zeroT, tol=0.0, rel=false,
        eps_rel=1f-5, r_required=max(1, min(BLOCK, rank_C)), block=BLOCK,
        workspace=tlr_ws)
    tlr_output_ms = best_time_ms(tlr_output, backend)

    err_dense_abs, err_dense_rel = reconstruction_error(
        reference, C_compressed, backend, T)
    err_tlr_abs, err_tlr_rel = reconstruction_error(
        reference, C_tlr, backend, T)
    result = (; dense_ms, dense_compress_ms, tlr_output_ms,
              err_dense_abs, err_dense_rel, err_tlr_abs, err_tlr_rel)
    A = B = A_dense = B_dense = reference = C_dense = C_compressed = C_tlr = nothing
    dense_ws = compress_ws = tlr_ws = nothing
    release_backend_memory!()
    return result
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
    cases = NamedTuple[]
    for spec in CONFIG.cases, precision in CONFIG.precisions
        m, k, n = spec.m, spec.k, spec.n
        bm, bk, bn = spec.bm, spec.bk, spec.bn
        rank_A, rank_B = spec.maxrank_A, spec.maxrank_B
        max(rank_A, rank_B) <= min(bm, bk, bn) ||
            error("ranks must not exceed tile sizes (m=$m, k=$k, n=$n, " *
                  "bm=$bm, bk=$bk, bn=$bn, " *
                  "rank_A=$rank_A, rank_B=$rank_B)")
        precision_name = precision === :float16 ? "fp16" :
                         precision === :float32 ? "fp32" : "fp64"
        base_id = "m$(m)__k$(k)__n$(n)__bm$(bm)__bk$(bk)__bn$(bn)__ra$(rank_A)__rb$(rank_B)"
        id = precision === :float16 ? "fp16__$(base_id)" :
             precision === :float32 ? base_id : "fp64__$(base_id)"
        push!(cases, (; id, precision, precision_name, m, k, n, bm, bk, bn, rank_A, rank_B))
    end
    selected = [
        case for (index, case) in enumerate(cases)
        if mod1(index, CONFIG.shard_count) == CONFIG.shard_index &&
           occursin(CONFIG.case_regex, case.id) && !(case.id in done)
    ]
    @printf("NextLA TLR-output GEMM benchmark: backend=%s cases=%d/%d reps=%d warmup=%d shard=%d/%d output=%s\n",
            backend_name, length(selected), length(cases), NREPS, WARMUP,
            CONFIG.shard_index, CONFIG.shard_count, OUTPUT)
    for case in selected
        id, precision, precision_name = case.id, case.precision, case.precision_name
        m, k, n = case.m, case.k, case.n
        bm, bk, bn = case.bm, case.bk, case.bn
        rank_A, rank_B = case.rank_A, case.rank_B
        T = precision === :float16 ? Float16 :
            precision === :float32 ? Float32 : Float64
        result = benchmark_case(backend, T, m, k, n, bm, bk, bn, rank_A, rank_B)
        row = (id, m, k, n, bm, bk, bn, rank_A, rank_B, min(rank_A, rank_B), precision_name, NREPS,
               result.dense_ms, result.dense_compress_ms, result.tlr_output_ms,
               result.dense_ms / result.dense_compress_ms,
               result.dense_ms / result.tlr_output_ms,
               result.err_dense_abs, result.err_dense_rel,
               result.err_tlr_abs, result.err_tlr_rel)
        open(OUTPUT, "a") do io
            println(io, join(row, ','))
            flush(io)
        end
        @printf("%-28s dense=%8.3f ms  dense+compress=%8.3f ms  TLR=%8.3f ms  errors=(%.3e, %.3e)\n",
                id, result.dense_ms, result.dense_compress_ms, result.tlr_output_ms,
                result.err_dense_rel, result.err_tlr_rel)
    end
end

main()
