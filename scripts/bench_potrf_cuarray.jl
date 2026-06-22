using CUDA
using CUDA.CUSOLVER
using LinearAlgebra
using Statistics
using NextLA

const T = Float32

_env_int(name, default) = parse(Int, get(ENV, name, string(default)))
_env_bool(name, default) = get(ENV, name, default ? "1" : "0") == "1"
_env_int_list(name, default) = [parse(Int, x) for x in split(get(ENV, name, default), ',') if !isempty(x)]

potrf_gflops(n::Int, time_s::Real; batch::Int=1) = batch * (Float64(n)^3 / 3) / time_s / 1e9

function make_spd_gpu(n::Int, ::Type{T}) where {T}
    X = CUDA.randn(T, n, n)
    return X * X' + T(n) * I
end

function make_spd_batch_host(n::Int, batch::Int, ::Type{T}) where {T}
    A = Array{T,3}(undef, n, n, batch)
    for b in 1:batch
        X = randn(T, n, n)
        A[:, :, b] = X * X' + T(n) * I
    end
    return A
end

function bench_nextla_single!(Aref, Awork, config; warmup::Int, iters::Int, uplo::Char='L')
    ob, ib, rb, mr = config
    for _ in 1:warmup
        copyto!(Awork, Aref)
        NextLA.potrf!(uplo, Awork, Val(ob), Val(ib), Val(rb), Val(mr))
        synchronize()
    end
    times = Float64[]
    for _ in 1:iters
        copyto!(Awork, Aref)
        t = @elapsed begin
            NextLA.potrf!(uplo, Awork, Val(ob), Val(ib), Val(rb), Val(mr))
            synchronize()
        end
        push!(times, t)
    end
    return times
end

function bench_cusolver_single!(Aref, Awork; warmup::Int, iters::Int, uplo::Char='L')
    for _ in 1:warmup
        copyto!(Awork, Aref)
        CUSOLVER.Xpotrf!(uplo, Awork)
        synchronize()
    end
    times = Float64[]
    for _ in 1:iters
        copyto!(Awork, Aref)
        t = @elapsed begin
            CUSOLVER.Xpotrf!(uplo, Awork)
            synchronize()
        end
        push!(times, t)
    end
    return times
end

function bench_nextla_batched!(Aref, Awork, config; warmup::Int, iters::Int, uplo::Char='L')
    ob, ib, rb, mr = config
    for _ in 1:warmup
        copyto!(Awork, Aref)
        NextLA.potrf!(uplo, Awork, Val(ob), Val(ib), Val(rb), Val(mr))
        synchronize()
    end
    times = Float64[]
    for _ in 1:iters
        copyto!(Awork, Aref)
        t = @elapsed begin
            NextLA.potrf!(uplo, Awork, Val(ob), Val(ib), Val(rb), Val(mr))
            synchronize()
        end
        push!(times, t)
    end
    return times
end

_case_stats(times, n; batch::Int=1) = (
    avg = mean(times),
    median = median(times),
    best = minimum(times),
    avg_gflops = potrf_gflops(n, mean(times); batch),
    median_gflops = potrf_gflops(n, median(times); batch),
    best_gflops = potrf_gflops(n, minimum(times); batch),
)

function report_case(label::String, n::Int, times; batch::Int=1, config=nothing)
    stats = _case_stats(times, n; batch)
    println(label)
    println("  n=$n batch=$batch")
    config === nothing || println("  config=$(config)")
    println("  times_s=$(times)")
    println("  avg_s=$(stats.avg)")
    println("  median_s=$(stats.median)")
    println("  best_s=$(stats.best)")
    println("  avg_gflops=$(stats.avg_gflops)")
    println("  median_gflops=$(stats.median_gflops)")
    println("  best_gflops=$(stats.best_gflops)")
end

function potrf_configs()
    obs = _env_int_list("NEXTLA_POTRF_OBS", "64")
    ibs = _env_int_list("NEXTLA_POTRF_IBS", "32,64")
    rbs = _env_int_list("NEXTLA_POTRF_RBS", "4,8")
    mrs = _env_int_list("NEXTLA_POTRF_MRS", "8,16")
    configs = Tuple{Int,Int,Int,Int}[]
    for ob in obs, ib in ibs, rb in rbs, mr in mrs
        ib <= ob || continue
        push!(configs, (ob, ib, rb, mr))
    end
    return configs
end

function sweep_nextla(label, bench!, Aref, Awork, n; warmup::Int, iters::Int, batch::Int=1)
    results = NamedTuple[]
    for config in potrf_configs()
        print("testing $label config=$config ... ")
        flush(stdout)
        try
            times = bench!(Aref, Awork, config; warmup, iters)
            stats = _case_stats(times, n; batch)
            println("best_s=$(stats.best) best_gflops=$(stats.best_gflops)")
            push!(results, (; config, times, stats))
        catch err
            println("failed ($err)")
        end
    end
    isempty(results) && error("no valid $label configurations completed")
    sort!(results, by = x -> x.stats.best)
    best = first(results)
    println("$label best config=$(best.config)")
    report_case("$label best", n, best.times; batch, config=best.config)
    topk = min(length(results), 5)
    println("$label top $topk configs by best_s")
    for result in results[1:topk]
        println("  config=$(result.config) best_s=$(result.stats.best) best_gflops=$(result.stats.best_gflops)")
    end
    return results
end

function main()
    CUDA.functional() || error("CUDA is not functional on this machine")

    single_n = _env_int("NEXTLA_POTRF_SINGLE_N", 2048)
    batch_n = _env_int("NEXTLA_POTRF_BATCH_N", 2048)
    batch_count = _env_int("NEXTLA_POTRF_BATCH_COUNT", 100)
    warmup = _env_int("NEXTLA_POTRF_WARMUP", 1)
    iters = _env_int("NEXTLA_POTRF_ITERS", 3)
    run_cusolver = _env_bool("NEXTLA_POTRF_RUN_CUSOLVER", true)
    run_batched = _env_bool("NEXTLA_POTRF_RUN_BATCHED", true)

    println("potrf benchmark")
    println("  backend=CUDA")
    println("  warmup=$warmup iters=$iters")
    println("  configs=$(potrf_configs())")

    println("preparing single-matrix input n=$single_n")
    single_ref = CuArray(Matrix(make_spd_gpu(single_n, T)))
    single_work = similar(single_ref)
    sweep_nextla("NextLA single", bench_nextla_single!, single_ref, single_work, single_n; warmup, iters)

    if run_cusolver
        cusolver_single = bench_cusolver_single!(single_ref, single_work; warmup, iters)
        report_case("cuSOLVER single", single_n, cusolver_single)
    end

    if run_batched
        println("preparing batched input n=$batch_n batch=$batch_count")
        batch_ref = CuArray(make_spd_batch_host(batch_n, batch_count, T))
        batch_work = similar(batch_ref)
        sweep_nextla("NextLA batched", bench_nextla_batched!, batch_ref, batch_work, batch_n; warmup, iters, batch=batch_count)
    end
end

main()
