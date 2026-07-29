"""CUDA grouped-GEMM microbenchmark against batched and individual GEMMs.

Run from the repository root with:

    julia --project=experiments experiments/grouped_gemm.jl

The configuration block is intentionally at the top so the benchmark cases
can be changed without editing the measurement code.
"""

using CUDA
using NextLA
using Printf
using Statistics

# ── Configuration ────────────────────────────────────────────────────────────
const BATCH = 128
const FIXED_SIZE = (1024, 1024, 1024)       # m, k, n for the fixed-size case
const RAGGED_N_RANGE = (128, 1024)        # n range for fixed m and k
const HETEROGENEOUS_RANGES = ((256, 1024), (256, 1024), (128, 1024)) # m, k, n
const WARMUP = parse(Int, get(ENV, "NEXTLA_GEMM_GROUPED_WARMUP", "5"))
const REPS = parse(Int, get(ENV, "NEXTLA_GEMM_GROUPED_REPS", "20"))

CUDA.functional() || error("no CUDA device available")

function elapsed_ms!(f)
    CUDA.synchronize()
    start = time_ns()
    f()
    CUDA.synchronize()
    return (time_ns() - start) / 1.0e6
end

function summarize(label, f, flops)
    for _ in 1:WARMUP
        f()
    end
    samples = [elapsed_ms!(f) for _ in 1:REPS]
    elapsed = median(samples)
    @printf("%-34s %9.3f ms  %9.2f GFLOP/s\n",
            label, elapsed, flops / (elapsed * 1.0e6))
    return elapsed
end

function grouped_tasks(dims)
    tasks = NextLA.GroupedGemmTask[]
    for (m, k, n) in dims
        A = CUDA.rand(Float32, m, k)
        B = CUDA.rand(Float32, k, n)
        C = CUDA.zeros(Float32, m, n)
        push!(tasks, NextLA.GroupedGemmTask('N', 'N', 1.0f0, A, B, 0.0f0, C))
    end
    return tasks
end

function benchmark_case(label, dims)
    tasks = grouped_tasks(dims)
    A = [task.A for task in tasks]
    B = [task.B for task in tasks]
    Cbatched = [CUDA.zeros(Float32, size(task.C)...) for task in tasks]
    flops = sum(2.0 * m * k * n for (m, k, n) in dims)

    grouped_time = summarize(
        "precision_gemm_grouped!", 
        () -> NextLA.precision_gemm_grouped!(tasks, NextLA.GEMMCompute{Float32}()),
        flops)
    batched_time = summarize(
        "gemm_batched!",
        () -> NextLA.gemm_batched!('N', 'N', 1.0f0, A, B, 0.0f0, Cbatched),
        flops)
    @printf("%s grouped / batched: %.3fx\n", label, grouped_time / batched_time)
end

function benchmark_individual_case(label, dims)
    tasks = grouped_tasks(dims)
    flops = sum(2.0 * m * k * n for (m, k, n) in dims)
    individual = () -> foreach(tasks) do task
        NextLA.precision_gemm!(task.transA, task.transB, task.alpha, task.A,
                               task.B, task.beta, task.C,
                               NextLA.GEMMCompute{Float32}())
    end
    grouped_time = summarize(
        "precision_gemm_grouped!",
        () -> NextLA.precision_gemm_grouped!(
            tasks, NextLA.GEMMCompute{Float32}()),
        flops)
    individual_time = summarize("individual precision_gemm!", individual, flops)
    @printf("%s grouped / individual: %.3fx\n",
            label, grouped_time / individual_time)
end

release_gpu_memory!() = (GC.gc(true); CUDA.reclaim(); nothing)

function main()
    m, k, n = FIXED_SIZE
    println("CUDA device: ", CUDA.name(CUDA.device()))
    println("batch=$BATCH, warmup=$WARMUP, reps=$REPS, dtype=Float32")

    println("\n$BATCH × fixed $(m)×$(k)×$(n)")
    benchmark_case("fixed", fill(FIXED_SIZE, BATCH))
    release_gpu_memory!()

    nlo, nhi = RAGGED_N_RANGE
    ns = round.(Int, range(nlo, nhi; length=BATCH))
    println("\n$BATCH × $(m)×$(k)×n, n=$nlo:$nhi (grouped vs individual)")
    benchmark_individual_case("ragged", [(m, k, ni) for ni in ns])
    release_gpu_memory!()

    (mlo, mhi), (klo, khi), (nlo, nhi) = HETEROGENEOUS_RANGES
    dims = collect(zip(round.(Int, range(mlo, mhi; length=BATCH)),
                       reverse(round.(Int, range(klo, khi; length=BATCH))),
                       round.(Int, range(nlo, nhi; length=BATCH))))
    println("\n$BATCH × heterogeneous m,k,n (grouped vs individual)")
    benchmark_individual_case("heterogeneous", dims)
    release_gpu_memory!()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
