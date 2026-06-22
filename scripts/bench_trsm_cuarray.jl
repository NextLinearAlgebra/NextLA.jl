 using CUDA
 using LinearAlgebra
 using Statistics
 using NextLA
 using CUDA.CUBLAS

const T = Float32
const SIDE = 'L'
const UPLO = 'L'
const TRANSA = 'N'
const DIAG = 'N'

_env_int(name, default) = parse(Int, get(ENV, name, string(default)))
_env_bool(name, default) = get(ENV, name, default ? "1" : "0") == "1"

trsm_gflops(n::Int, nrhs::Int, time_s::Real; batch::Int=1) = batch * Float64(n)^2 * nrhs / time_s / 1e9

function make_triangular_gpu(n::Int, ::Type{T}) where {T}
    M = CUDA.rand(T, n, n) .+ one(T)
    A = Matrix(LowerTriangular(Array(M)))
    A .+= Diagonal(fill(T(10), n))
    return CuArray(A)
end

function make_rhs_gpu(n::Int, nrhs::Int, ::Type{T}) where {T}
    return CUDA.rand(T, n, nrhs) .+ one(T)
end

function make_triangular_batch_host(n::Int, batch::Int, ::Type{T}) where {T}
    A = Array{T,3}(undef, n, n, batch)
    for b in 1:batch
        M = rand(T, n, n) .+ one(T)
        Ab = Matrix(LowerTriangular(M))
        Ab .+= Diagonal(fill(T(10), n))
        A[:, :, b] = Ab
    end
    return A
end

function make_rhs_batch_host(n::Int, nrhs::Int, batch::Int, ::Type{T}) where {T}
    B = Array{T,3}(undef, n, nrhs, batch)
    for b in 1:batch
        B[:, :, b] = rand(T, n, nrhs) .+ one(T)
    end
    return B
end

function bench_nextla_single!(Aref, Bref, Awork, Bwork; warmup::Int, iters::Int)
    for _ in 1:warmup
        copyto!(Awork, Aref)
        copyto!(Bwork, Bref)
        NextLA.trsm!(SIDE, UPLO, TRANSA, DIAG, Awork, Bwork)
        synchronize()
    end
    times = Float64[]
    for _ in 1:iters
        copyto!(Awork, Aref)
        copyto!(Bwork, Bref)
        t = @elapsed begin
            NextLA.trsm!(SIDE, UPLO, TRANSA, DIAG, Awork, Bwork)
            synchronize()
        end
        push!(times, t)
    end
    return times
end

function bench_cublas_single!(Aref, Bref, Awork, Bwork; warmup::Int, iters::Int)
    for _ in 1:warmup
        copyto!(Awork, Aref)
        copyto!(Bwork, Bref)
        CUBLAS.trsm!(SIDE, UPLO, TRANSA, DIAG, one(T), Awork, Bwork)
        synchronize()
    end
    times = Float64[]
    for _ in 1:iters
        copyto!(Awork, Aref)
        copyto!(Bwork, Bref)
        t = @elapsed begin
            CUBLAS.trsm!(SIDE, UPLO, TRANSA, DIAG, one(T), Awork, Bwork)
            synchronize()
        end
        push!(times, t)
    end
    return times
end

function bench_nextla_batched!(Aref, Bref, Awork, Bwork; warmup::Int, iters::Int)
    for _ in 1:warmup
        copyto!(Awork, Aref)
        copyto!(Bwork, Bref)
        NextLA.trsm_batched!(SIDE, UPLO, TRANSA, DIAG, Awork, Bwork)
        synchronize()
    end
    times = Float64[]
    for _ in 1:iters
        copyto!(Awork, Aref)
        copyto!(Bwork, Bref)
        t = @elapsed begin
            NextLA.trsm_batched!(SIDE, UPLO, TRANSA, DIAG, Awork, Bwork)
            synchronize()
        end
        push!(times, t)
    end
    return times
end

function report_case(label::String, n::Int, nrhs::Int, times; batch::Int=1)
    avg = mean(times)
    med = median(times)
    best = minimum(times)
    println(label)
    println("  n=$n nrhs=$nrhs batch=$batch")
    println("  times_s=$(times)")
    println("  avg_s=$(avg)")
    println("  median_s=$(med)")
    println("  best_s=$(best)")
    println("  avg_gflops=$(trsm_gflops(n, nrhs, avg; batch))")
    println("  median_gflops=$(trsm_gflops(n, nrhs, med; batch))")
    println("  best_gflops=$(trsm_gflops(n, nrhs, best; batch))")
end

function main()
    CUDA.functional() || error("CUDA is not functional on this machine")

    single_n = _env_int("NEXTLA_TRSM_SINGLE_N", 8192)
    single_nrhs = _env_int("NEXTLA_TRSM_SINGLE_NRHS", single_n)
    batch_n = _env_int("NEXTLA_TRSM_BATCH_N", 512)
    batch_nrhs = _env_int("NEXTLA_TRSM_BATCH_NRHS", batch_n)
    batch_count = _env_int("NEXTLA_TRSM_BATCH_COUNT", 100)
    warmup = _env_int("NEXTLA_TRSM_WARMUP", 1)
    iters = _env_int("NEXTLA_TRSM_ITERS", 3)
    run_cublas = _env_bool("NEXTLA_TRSM_RUN_CUBLAS", true)

    println("trsm benchmark")
    println("  date=2026-06-22")
    println("  backend=CUDA")
    println("  side=$SIDE uplo=$UPLO transa=$TRANSA diag=$DIAG")
    println("  warmup=$warmup iters=$iters")

    println("preparing single-matrix input n=$single_n nrhs=$single_nrhs")
    single_A_ref = make_triangular_gpu(single_n, T)
    single_B_ref = make_rhs_gpu(single_n, single_nrhs, T)
    single_A_work = similar(single_A_ref)
    single_B_work = similar(single_B_ref)

    nextla_single = bench_nextla_single!(single_A_ref, single_B_ref, single_A_work, single_B_work; warmup, iters)
    report_case("NextLA single", single_n, single_nrhs, nextla_single)

    if run_cublas
        cublas_single = bench_cublas_single!(single_A_ref, single_B_ref, single_A_work, single_B_work; warmup, iters)
        report_case("cuBLAS single", single_n, single_nrhs, cublas_single)
    end

    println("preparing batched input n=$batch_n nrhs=$batch_nrhs batch=$batch_count")
    batch_A_ref = CuArray(make_triangular_batch_host(batch_n, batch_count, T))
    batch_B_ref = CuArray(make_rhs_batch_host(batch_n, batch_nrhs, batch_count, T))
    batch_A_work = similar(batch_A_ref)
    batch_B_work = similar(batch_B_ref)

    nextla_batched = bench_nextla_batched!(batch_A_ref, batch_B_ref, batch_A_work, batch_B_work; warmup, iters)
    report_case("NextLA batched", batch_n, batch_nrhs, nextla_batched; batch=batch_count)
end

main()
