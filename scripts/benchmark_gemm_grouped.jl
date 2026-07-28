"""Benchmark CUDA grouped GEMM against conventional batched/individual GEMMs.

Run with, for example:

    julia --project=. scripts/benchmark_gemm_grouped.jl

The benchmark uses Float32 and reports median wall-clock time for the device
operation, including a synchronization but excluding allocation.
"""

using CUDA
using NextLA
using Printf
using Statistics

CUDA.functional() || error("no CUDA device available")

const BATCH = 128
const M = 512
const K = 512
const WARMUP = parse(Int, get(ENV, "NEXTLA_GEMM_GROUPED_WARMUP", "5"))
const REPS = parse(Int, get(ENV, "NEXTLA_GEMM_GROUPED_REPS", "20"))
const MODE = NextLA.GEMMCompute{Float32}()

function elapsed_ms!(f)
    CUDA.synchronize()
    t = time_ns()
    f()
    CUDA.synchronize()
    return (time_ns() - t) / 1.0e6
end

function summarize(label, f, flops)
    for _ in 1:WARMUP
        f()
    end
    CUDA.synchronize()
    samples = [elapsed_ms!(f) for _ in 1:REPS]
    t = median(samples)
    @printf("%-34s %9.3f ms  %9.2f GFLOP/s\n", label, t, flops / (t * 1.0e6))
    return t
end

function grouped_tasks(ns)
    tasks = NextLA.GroupedGemmTask[]
    for i in 1:BATCH
        n = ns[i]
        A = CUDA.rand(Float32, M, K)
        B = CUDA.rand(Float32, K, n)
        C = CUDA.zeros(Float32, M, n)
        push!(tasks, NextLA.GroupedGemmTask('N', 'N', 1.0f0, A, B, 0.0f0, C))
    end
    return tasks
end

function main()
    println("CUDA device: ", CUDA.name(CUDA.device()))
    println("batch=$BATCH, m=k=$M, Float32, warmup=$WARMUP, reps=$REPS")

    println("128 × 512×512×256 (grouped vs pointer-batched)")
    grouped = grouped_tasks(fill(512, BATCH))
    A = [task.A for task in grouped]
    B = [task.B for task in grouped]
    Cbatched = [CUDA.zeros(Float32, M, 512) for _ in 1:BATCH]
    fixed_flops = 2.0 * BATCH * M * K * 512
    tg = summarize("precision_gemm_grouped!", () -> NextLA.precision_gemm_grouped!(grouped, MODE), fixed_flops)
    tb = summarize("gemm_batched!", () -> NextLA.gemm_batched!('N', 'N', 1.0f0, A, B, 0.0f0, Cbatched), fixed_flops)
    @printf("grouped / batched: %.3fx\n", tg / tb)

    println("\n128 × 512×512×n, n=128:512 (grouped vs individual GEMMs)")
    ns = collect(round.(Int, range(128, 512; length=BATCH)))
    ragged = grouped_tasks(ns)
    individual = () -> foreach(ragged) do task
        NextLA.precision_gemm!(task.transA, task.transB, task.alpha, task.A,
                               task.B, task.beta, task.C, MODE)
    end
    ragged_flops = sum(2.0 * M * K .* ns)
    tg_ragged = summarize("precision_gemm_grouped!", () -> NextLA.precision_gemm_grouped!(ragged, MODE), ragged_flops)
    ti = summarize("128 individual precision_gemm!", individual, ragged_flops)
    @printf("grouped / individual: %.3fx\n", tg_ragged / ti)
end

main()
