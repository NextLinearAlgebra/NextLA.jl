# Julia half of the executable-first NextLA-vs-KBLAS TLR accumulation benchmark.
# `run_benchmark_tlr_accum_kblas.sh` runs KBLAS first and writes the exact Float32
# factors/output here; this script then times NextLA using those same factor values.

using CUDA, LinearAlgebra, Printf, Statistics
using NextLA
const M = NextLA.TLRmodule
const T = Float32
const MAGIC = UInt32(0x4e4b544c)

struct Record
    b::Int; nt::Int; r::Int; rC::Int; profile::Symbol; beta::T; kblas_rank::Int; kblas_ms::Float64
    au::Array{T,3}; av::Array{T,3}; bu::Array{T,3}; bv::Array{T,3}
    cu0::Array{T,3}; cv0::Array{T,3}; cu::Array{T,3}; cv::Array{T,3}
end

profile_name(id::Int) = id == 0 ? :uniform : id == 1 ? :a_row_skew : :b_column_skew

function read_array(io, b, r, nt)
    x = Vector{T}(undef, b * r * nt * nt)
    read!(io, x)
    return reshape(x, b, r, nt * nt)
end

function read_record(path)
    open(path, "r") do io
        read(io, UInt32) == MAGIC || error("$path is not a KBLAS benchmark record")
        read(io, UInt32) == 1 || error("$path has an unsupported record version")
        b, nt, r, rC, profile, beta, kr = (Int(read(io, Int32)) for _ in 1:7)
        kms = read(io, Float64)
        au = read_array(io, b, r, nt); av = read_array(io, b, r, nt)
        bu = read_array(io, b, r, nt); bv = read_array(io, b, r, nt)
        cu0 = read_array(io, b, rC, nt); cv0 = read_array(io, b, rC, nt)
        cu = read_array(io, b, rC, nt); cv = read_array(io, b, rC, nt)
        return Record(b, nt, r, rC, profile_name(profile), T(beta), kr, kms, au, av, bu, bv, cu0, cv0, cu, cv)
    end
end

@inline function effective_rank(profile, operand, i, j, r, beta)
    operand === :C && return iszero(beta) ? 0 : r
    low = max(1, r ÷ 4)
    profile === :a_row_skew && operand === :A && return isodd(i) ? low : r
    profile === :b_column_skew && operand === :B && return isodd(j) ? low : r
    return r
end

function load_tlr(Ulogical, Vlogical, b, nt, capacity, order, profile, operand, r, beta)
    X = M.PaddedFTLRMatrix(CUDA.CUDABackend(), T, b * nt, b * nt, (b, b), capacity; tile_order=order)
    Uh = zeros(T, size(X.int_U)); Vh = zeros(T, size(X.int_V))
    for j in 1:nt, i in 1:nt
        logical = i + (j - 1) * nt
        slot = M.tile_linear_index(X.order, nt, nt, i, j)
        Uh[:, :, slot] .= view(Ulogical, :, :, logical)
        Vh[:, :, slot] .= view(Vlogical, :, :, logical)
        X.ranks[slot] = Int32(effective_rank(profile, operand, i, j, r, beta))
    end
    copyto!(X.int_U, Uh); copyto!(X.int_V, Vh)
    return X
end

function copy_factors!(dest, src)
    copyto!(dest.int_U, src.int_U); copyto!(dest.int_V, src.int_V)
    copyto!(dest.ranks, src.ranks); copyto!(dest.resid, src.resid)
    return dest
end

function reconstruct(Ulogical, Vlogical, b, nt, ranks)
    out = zeros(Float64, b * nt, b * nt)
    for j in 1:nt, i in 1:nt
        r = ranks(i, j); r == 0 && continue
        logical = i + (j - 1) * nt
        rows = (i - 1) * b + 1:i * b; cols = (j - 1) * b + 1:j * b
        out[rows, cols] .= Float64.(view(Ulogical, :, 1:r, logical)) * Float64.(view(Vlogical, :, 1:r, logical))'
    end
    return out
end

function reconstruct_nextla(X)
    Uh, Vh = Array(X.int_U), Array(X.int_V); b = size(Uh, 1); nt = size(X, 1) ÷ b
    return reconstruct(Uh, Vh, b, nt,
                       (i, j) -> Int(X.ranks[M.tile_linear_index(X.order, nt, nt, i, j)]))
end

relative_error(x, ref) = norm(x - ref) / max(norm(ref), eps(Float64))

function time_nextla!(C, C0, A, B, beta; warmup=3, reps=10)
    for _ in 1:warmup
        copy_factors!(C, C0); M.gemm!(C, A, B; alpha=one(T), beta, tol=0.0, rel=false)
    end
    CUDA.synchronize(); samples = Float64[]
    for _ in 1:reps
        copy_factors!(C, C0); CUDA.synchronize()
        push!(samples, CUDA.@elapsed M.gemm!(C, A, B; alpha=one(T), beta, tol=0.0, rel=false))
    end
    return median(samples) * 1e3
end

function run_record(rec; smoke=false)
    A = load_tlr(rec.au, rec.av, rec.b, rec.nt, rec.r, M.TileRowMajor(), rec.profile, :A, rec.r, rec.beta)
    B = load_tlr(rec.bu, rec.bv, rec.b, rec.nt, rec.r, M.TileColMajor(), rec.profile, :B, rec.r, rec.beta)
    C0 = load_tlr(rec.cu0, rec.cv0, rec.b, rec.nt, rec.rC, M.TileRowMajor(), rec.profile, :C, rec.r, rec.beta)
    C = M.PaddedFTLRMatrix(CUDA.CUDABackend(), T, rec.b * rec.nt, rec.b * rec.nt, (rec.b, rec.b), rec.rC; tile_order=M.TileRowMajor())
    Aref = reconstruct(rec.au, rec.av, rec.b, rec.nt, (i, j) -> effective_rank(rec.profile, :A, i, j, rec.r, rec.beta))
    Bref = reconstruct(rec.bu, rec.bv, rec.b, rec.nt, (i, j) -> effective_rank(rec.profile, :B, i, j, rec.r, rec.beta))
    C0ref = reconstruct(rec.cu0, rec.cv0, rec.b, rec.nt, (i, j) -> effective_rank(rec.profile, :C, i, j, rec.r, rec.beta))
    ref = Aref * Bref + Float64(rec.beta) * C0ref
    kblas_finite = all(isfinite, rec.cu) && all(isfinite, rec.cv)
    if smoke && !kblas_finite
        error("KBLAS returned non-finite factors for profile=$(rec.profile), beta=$(rec.beta). " *
              "Its scalar-rank LLL routine does not accept this exact zero-padded A effective-rank profile.")
    end
    nextla_ms = time_nextla!(C, C0, A, B, rec.beta)
    copy_factors!(C, C0); M.gemm!(C, A, B; alpha=one(T), beta=rec.beta, tol=0.0, rel=false); CUDA.synchronize()
    nextla_err = relative_error(reconstruct_nextla(C), ref)
    kblas_err = kblas_finite ?
        relative_error(reconstruct(rec.cu, rec.cv, rec.b, rec.nt, (i, j) -> rec.kblas_rank), ref) : Inf
    ranks = Int.(Array(C.ranks))
    smoke && (nextla_err <= 5e-4 && kblas_err <= 5e-4 || error("smoke reconstruction error exceeded 5e-4"))
    return (; rec, nextla_ms, nextla_err, kblas_err, kblas_finite, ranks,
            maxres=maximum(Array(C.resid)))
end

function main(args=ARGS)
    pos = findfirst(==("--results"), args)
    pos === nothing && error("usage: benchmark_tlr_accum_kblas.jl --results DIR [--smoke]")
    result_dir = args[pos + 1]; smoke = "--smoke" in args
    files = sort(filter(f -> endswith(f, ".bin"), readdir(result_dir; join=true)))
    isempty(files) && error("no KBLAS benchmark records under $result_dir")
    @printf("NextLA vs KBLAS TLR accumulation benchmark\nGPU: %s\nCUDA runtime: %s\nKBLAS: %s\n\n",
            CUDA.name(CUDA.device()), CUDA.runtime_version(),
            get(ENV, "KBLAS_ROOT", "(not provided)"))
    println(" b   nt   r  profile        beta | NextLA ms  KBLAS ms  speedup | NextLA relerr  KBLAS relerr | NextLA ranks(min/mean/max)  KBLAS rank  finite  max residual")
    println("-"^160)
    for file in files
        x = run_record(read_record(file); smoke)
        q = x.rec
        @printf("%3d %4d %3d  %-14s %4.0f | %9.3f  %8.3f  %7.2f | %13.3e  %12.3e | %4d/%5.1f/%4d           %4d    %-5s %11.3e\n",
                q.b, q.nt, q.r, String(q.profile), q.beta, x.nextla_ms, q.kblas_ms,
                x.nextla_ms / q.kblas_ms, x.nextla_err, x.kblas_err,
                minimum(x.ranks), mean(x.ranks), maximum(x.ranks), q.kblas_rank,
                string(x.kblas_finite), x.maxres)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
