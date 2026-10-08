# Run from NextLA.jl with crankseg_2.mtx and nasasrb.mtx beside this script.
# Two independent runs use identical seeded dense matrices:
#   METHODS=gpu    julia --project=. cholesky_sparse_baselines.jl
#   METHODS=stiles julia --project=. cholesky_sparse_baselines.jl
# Set CASES=dense|real to select a subset. By default the dense shift is n;
# for real matrices, the shift is at least n and large enough to make every
# row strictly diagonally dominant. Use DIAGONAL_SHIFT=0 and
# REAL_SHIFT_POLICY=fixed together to test unmodified catalog matrices.
# sTiles' Julia wrapper is run on CPU and needs a CPU allocation with enough
# memory and cores for a fully populated 65536-by-65536 CSC input.
# This is a timing harness. Check factorization residuals separately before
# interpreting speedups, especially for the mixed-precision configurations.

using Test, CUDA, LinearAlgebra, Printf, KernelAbstractions, sTiles
using MatrixMarket, Random, SparseArrays, StochasticRounding

include("benchmark.jl")
include("flops.jl")

const MATRIX_DIR = get(ENV, "MATRIX_DIR", @__DIR__)
const REPEATS = parse(Int, get(ENV, "BENCH_REPEATS", "5"))
const STILES_CORES = parse(Int, get(ENV, "STILES_CORES", get(ENV, "SLURM_CPUS_PER_TASK", "1")))
const SHIFT_SETTING = get(ENV, "DIAGONAL_SHIFT", "n")
const CASES = get(ENV, "CASES", "all")
const METHODS = get(ENV, "METHODS", "all")
# Synthetic dense matrices still use DIAGONAL_SHIFT, defaulting to n.
# Real matrices use a separate setting, defaulting to no modification.
const REAL_SHIFT_SETTING = get(ENV, "REAL_DIAGONAL_SHIFT", "0")
const REAL_SHIFT_POLICY = get(ENV, "REAL_SHIFT_POLICY", "fixed")

const REAL_MATRICES = filter(
    name -> !isempty(name),
    strip.(split(get(
        ENV, "REAL_MATRICES",
        "nasasrb,crankseg_2,bodyy4,bodyy5,bodyy6,inla_graph_animal2"
    ), ','))
)
CASES in ("all", "dense", "real") || error("CASES must be all, dense, or real")
METHODS in ("all", "gpu", "stiles") || error("METHODS must be all, gpu, or stiles")
REAL_SHIFT_POLICY in ("dominant", "fixed") || error("REAL_SHIFT_POLICY must be dominant or fixed")
REPEATS > 0 || error("BENCH_REPEATS must be positive")

if METHODS != "stiles"
    # These are the source files included by the original benchmark. Run this
    # script from the same project directory as that benchmark.
    for file in ("symmmixedprec.jl", "recmixedprectri.jl", "trsm.jl", "trmm.jl",
                 "matmul.jl", "rectrxm.jl", "recsyrk.jl", "potrf.jl", "wrappers.jl")
        path = joinpath(@__DIR__, file)
        isfile(path) || error("Missing $path: copy this benchmark next to your solver source files")
        include(path)
    end

    # The two recursive entry points from the implementation you provided.
    # If these are already in potrf.jl, use those definitions instead.
    if !isdefined(Main, :potrf_recursive!)
        function potrf_recursive!(A, block_size)
            n = size(A, 1)
            if n <= block_size
                potrf!(A)
                return
            end
            n1 = 2^floor(Int, log2(n)) ÷ 2
            A11 = @view A[1:n1, 1:n1]
            A21 = @view A[n1+1:end, 1:n1]
            A22 = @view A[n1+1:end, n1+1:end]
            potrf_recursive!(A11, block_size)
            if eltype(A11) == Float16
                unified_rectrxm!('R', 'L', 'T', 'N', 1.0, 'S', A11, A21)
            else
                trsm!('R', 'L', 'T', 'N', 1.0, A11, A21)
            end
            if eltype(A21) == Float16
                recsyrk!(-1.0, A21, 1.0, A22)
            else
                syrk!('L', 'N', -1.0, A21, 1.0, A22)
            end
            potrf_recursive!(A22, block_size)
            return
        end
        function potrf_recursive!(A::SymmMixedPrec)
            if A.BaseCase !== nothing
                potrf_recursive!(A.BaseCase, 4096)
                return
            end
            potrf_recursive!(A.A11)
            unified_rectrxm!('R', 'L', 'T', 'N', 1.0, 'S', TriMixedPrec(A.A11), A.OffDiag)
            recsyrk!(-1.0, A.OffDiag, 1.0, A.A22)
            potrf_recursive!(A.A22)
            return
        end
    end
end

function dense_case(n)
    rng = MersenneTwister(20260924 + n)
    A = rand(rng, Float64, n, n)
    # Symmetrize in place to avoid another huge dense allocation at 64k.
    @inbounds for j in 1:n, i in j+1:n
        a = (A[i, j] + A[j, i]) / 2
        A[i, j] = a
        A[j, i] = a
    end
    shift = SHIFT_SETTING == "n" ? Float64(n) : parse(Float64, SHIFT_SETTING)
    shift >= 0 || error("DIAGONAL_SHIFT must be nonnegative")
    @inbounds for i in 1:n
        A[i, i] += shift
    end
    return A, shift
end

function benchmark_op(op, reset_op, backend)
    reset_op()
    op()  # warm-up and compilation
    KernelAbstractions.synchronize(backend)

    min_time_ns = Inf
    for _ in 1:REPEATS
        reset_op()
        KernelAbstractions.synchronize(backend)  # exclude resetting inputs
        min_time_ns = min(min_time_ns, run_single_benchmark(op, backend))
    end
    return min_time_ns
end

function get_runtime_pure(A_spd_fp64, n::Int, T_prec::DataType)
    # Unlike the old pure-F16 benchmark, F32 and F64 factor the SAME input.
    A_clean = T_prec.(A_spd_fp64)
    A_perf = copy(A_clean)
    backend = KernelAbstractions.get_backend(A_perf)
    op = () -> potrf_recursive!(A_perf, 4096)
    reset_op = () -> copyto!(A_perf, A_clean)

    min_time_ns = benchmark_op(op, reset_op, backend)
    runtime_ms = min_time_ns / 1_000_000
    gflops = calculate_gflops(flops_potrf(T_prec, n), min_time_ns)
    return runtime_ms, gflops
end

function get_runtime_mixed(A_spd_fp64, n::Int, precisions::Vector)
    backend = KernelAbstractions.get_backend(A_spd_fp64)
    op = () -> begin
        A_to_factor = SymmMixedPrec(A_spd_fp64, 'L'; precisions=precisions)
        potrf_recursive!(A_to_factor)
    end
    reset_op = () -> nothing
    # As in your previous benchmark, this includes constructing SymmMixedPrec.
    min_time_ns = benchmark_op(op, reset_op, backend)
    runtime_ms = min_time_ns / 1_000_000
    gflops = calculate_gflops(flops_potrf(precisions[1], n), min_time_ns)
    return runtime_ms, gflops
end

function get_runtime_cusolver(A_spd_fp64, n::Int, T_prec::DataType)
    A_clean = T_prec == Float64 ? A_spd_fp64 : T_prec.(A_spd_fp64)
    A_perf = copy(A_clean)
    backend = KernelAbstractions.get_backend(A_perf)
    op = () -> CUDA.CUSOLVER.potrf!('L', A_perf)
    reset_op = () -> copyto!(A_perf, A_clean)

    min_time_ns = benchmark_op(op, reset_op, backend)
    runtime_ms = min_time_ns / 1_000_000
    gflops = calculate_gflops(flops_potrf(T_prec, n), min_time_ns)
    return runtime_ms, gflops
end

function stiles_once(A_cpu; cores=STILES_CORES)
    F = nothing
    t0 = time_ns()
    try
        F = sTiles.analyze(A_cpu; cores=cores)
        analyze_ms = (time_ns() - t0) / 1e6
        t0 = time_ns()
        sTiles.factorize!(F)
        factor_ms = (time_ns() - t0) / 1e6
        return analyze_ms, factor_ms
    finally
        F === nothing || close(F)
    end
end

function get_runtime_stiles(A_cpu)
    # Warm the Julia wrapper on a separate tiny matrix; only one live Factor at a time.
    Q = spdiagm(-1 => fill(-1.0, 5), 0 => fill(4.0, 6), 1 => fill(-1.0, 5))
    stiles_once(Q)
    times = [stiles_once(A_cpu) for _ in 1:REPEATS]
    # Choose a single repetition by total time, keeping its phases paired.
    return times[argmin(first.(times) .+ last.(times))]
end

function load_case(name)
    path = joinpath(MATRIX_DIR, name * ".mtx")
    isfile(path) || error("Matrix missing: $path")
    A = sparse(MatrixMarket.mmread(path))
    n, m = size(A)
    n == m && issymmetric(A) || error("$name must be square and symmetric")
    d = diag(A)
    off = vec(sum(abs, A; dims=2)) .- abs.(d)
    requested_shift = REAL_SHIFT_SETTING == "n" ?
                    Float64(n) : parse(Float64, REAL_SHIFT_SETTING)

    isfinite(requested_shift) && requested_shift >= 0 ||
        error("REAL_DIAGONAL_SHIFT must be finite and nonnegative")
    # For a symmetric matrix, d[i] + shift > sum(abs, offdiagonal row i))
    # guarantees positive definiteness and strong diagonal dominance.
    needed = max(0.0, maximum(off .- d))
    safe_shift = needed + max(1.0, needed) * 1e-8
    shift = REAL_SHIFT_POLICY == "dominant" ? max(requested_shift, safe_shift) : requested_shift
    if shift != 0
        A = A + spdiagm(0 => fill(shift, n))
    end
    d .+= shift
    margin = d .- off
    fraction = count(>(0), margin) / n
    @info "Input diagnostics" matrix=name shift=shift dominant_row_fraction=fraction worst_margin=minimum(margin)
    return A, shift
end

function run_one_matrix(name, A_cpu, shift, pure_scenarios, mixed_scenarios)
    n = size(A_cpu, 1)
    @printf("\n%s | n=%d | nnz(input)=%d | diagonal shift=%.8g\n",
            name, n, A_cpu isa SparseMatrixCSC ? nnz(A_cpu) : n*n, shift)

    if METHODS != "stiles"
        # Both GPU algorithms use a dense representation, including for real
        # sparse matrices. Conversion and transfer happen outside the timings.
        A_spd_fp64 = CuArray(A_cpu isa SparseMatrixCSC ? Matrix(A_cpu) : A_cpu)
        CUDA.synchronize()

        println("\n--- Pure Precision Recursive Cholesky (GPU) ---")
        for (label, T_prec) in pure_scenarios
            runtime_ms, gflops = get_runtime_pure(A_spd_fp64, n, T_prec)
            @printf("    %-28s | Runtime: %9.3f ms | Dense-equivalent GFLOPS: %9.2f\n",
                    label, runtime_ms, gflops)
            GC.gc(true); CUDA.reclaim()
        end

        println("\n--- Mixed Precision Recursive Cholesky (GPU) ---")
        for (label, precisions) in mixed_scenarios
            runtime_ms, gflops = get_runtime_mixed(A_spd_fp64, n, precisions)
            @printf("    %-28s | Runtime: %9.3f ms | Dense-equivalent GFLOPS: %9.2f\n",
                    label, runtime_ms, gflops)
            GC.gc(true); CUDA.reclaim()
        end

        println("\n--- cuSOLVER Dense Cholesky (GPU) ---")
        for (label, T_prec) in (("cuSOLVER F32", Float32), ("cuSOLVER F64", Float64))
            runtime_ms, gflops = get_runtime_cusolver(A_spd_fp64, n, T_prec)
            @printf("    %-28s | Runtime: %9.3f ms | Dense-equivalent GFLOPS: %9.2f\n",
                    label, runtime_ms, gflops)
            GC.gc(true); CUDA.reclaim()
        end

        A_spd_fp64 = nothing
        GC.gc(true); CUDA.reclaim()
    end

    if METHODS != "gpu"
        println("\n--- sTiles Sparse Cholesky (CPU, $(STILES_CORES) cores) ---")
        # sTiles takes CSC input; the dense synthetic matrices are stored as
        # fully populated CSC. This is very memory-intensive at 65536.
        A_stiles = A_cpu isa SparseMatrixCSC ? A_cpu : sparse(A_cpu)
        analysis_ms, factor_ms = get_runtime_stiles(A_stiles)
        @printf("    %-28s | Analyze: %9.3f ms | Factor: %9.3f ms\n",
                "sTiles F64", analysis_ms, factor_ms)
        A_stiles = nothing
        GC.gc(true)
    end
    flush(stdout)
end

function run_cholesky_benchmarks()
    n_values = [4096, 8192, 16384, 32768, 65536]
    # Pure F16 in the old script adds 100*I after rescaling, which changes the
    # matrix. Keep F32/F64 here so every printed result uses the same matrix.
    pure_scenarios = [("Pure F32", Float32), ("Pure F64", Float64)]
    mixed_scenarios = [
        ("[F16, F16, F32]", [Float16, Float16, Float32]),
        ("[F16, F16, F16, F16, F32]", [Float16, Float16, Float16, Float16, Float32]),
        ("[F16, F16, F16, F16, F16, F16, F32]", [Float16, Float16, Float16, Float16, Float16, Float16, Float32]),
        ("[F16, F16, F16, F16, F16, F16, F16, F32]", [Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float32]),
        ("[F16, F32]", [Float16, Float32]),
    ]

    println("Starting Cholesky Benchmark...")
    if CASES != "real"
        for n in n_values
            A_cpu, shift = dense_case(n)
            run_one_matrix("Dense synthetic $n × $n", A_cpu, shift, pure_scenarios, mixed_scenarios)
            A_cpu = nothing
            GC.gc(true)
        end
    end
    if CASES != "dense"
        for name in REAL_MATRICES
            A_cpu, shift = load_case(name)
            run_one_matrix("Real sparse $name", A_cpu, shift, pure_scenarios, mixed_scenarios)
            A_cpu = nothing
            GC.gc(true)
        end
    end
    println("\nBenchmark complete.")
end

run_cholesky_benchmarks()
