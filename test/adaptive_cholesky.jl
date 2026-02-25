using Test, CUDA, LinearAlgebra, Printf, KernelAbstractions
using MatrixDepot    
using SparseArrays  
using BFloat16s
include("benchmark.jl")

const PRECISION_MAP = Dict(
    "q52"  => Nothing,
    "bf16" => BFloat16,
    "f16"  => Float16,
    "f32"  => Float32,
    "f64"  => Float64
)


function benchmark_op(op, reset_op, backend)
    # 1. Warm-up
    reset_op()
    op()
    KernelAbstractions.synchronize(backend)

    min_time_ns = Inf
    for _ in 1:5
        reset_op()
        time = run_single_benchmark(op, backend)
        min_time_ns = min(min_time_ns, time)
    end
    
    return min_time_ns
end

function reconstruct_matrix(A::SymmMixedPrec{T_Base}) where {T_Base}
    if A.BaseCase !== nothing
        if A.base_scale !== nothing
            return copy(A.BaseCase) .* A.base_scale
        else
            return copy(A.BaseCase)
        end
    end
    
    C11 = reconstruct_matrix(A.A11)
    C22 = reconstruct_matrix(A.A22)
    
    local C21
    if A.offDiag_scale !== nothing
        C21 = A.OffDiag .* A.offDiag_scale
    else
        C21 = A.OffDiag
    end

    n1, m1 = size(C11)
    n2, m2 = size(C22)
    n = n1 + n2

    T_Recon = promote_type(eltype(C11), eltype(C22), eltype(C21))
    C_full = CuArray{T_Recon}(undef, n, n)
    
    C_full[1:n1, 1:m1] .= C11
    C_full[n1+1:n, 1:m1] .= C21
    C_full[n1+1:n, m1+1:n] .= C22
    C_full[1:n1, m1+1:n] .= transpose(C21)

    return C_full
end

function get_runtime_pure(A_spd_fp64, n::Int, T_prec::DataType)
    local A_clean
    
    if T_prec == Float16
        scale_factor = maximum(abs, A_spd_fp64)
        A_clean = Float16.(A_spd_fp64 ./ scale_factor)
    else
        A_clean = T_prec.(A_spd_fp64)
    end

    A_perf = copy(A_clean)
    backend = KernelAbstractions.get_backend(A_perf)

    op = () -> potrf_recursive!(A_perf, 4096)
    reset_op = () -> copyto!(A_perf, A_clean)

    min_time_ns = benchmark_op(op, reset_op, backend)
    runtime_ms = min_time_ns / 1_000_000
    gflops = ( (1/3) * n^3 ) / min_time_ns 

    return runtime_ms, gflops
end

function get_runtime_mixed(A_spd_fp64, n::Int, precisions::Vector)
    backend = KernelAbstractions.get_backend(A_spd_fp64)

    op = () -> begin
        A_to_factor = SymmMixedPrec(A_spd_fp64, 'L'; precisions=precisions)
        potrf_recursive!(A_to_factor)
    end
    
    reset_op = () -> () 

    min_time_ns = benchmark_op(op, reset_op, backend)
    runtime_ms = min_time_ns / 1_000_000
    gflops = ( (1/3) * n^3 ) / min_time_ns

    return runtime_ms, gflops
end

function get_runtime_cusolver(A_spd_fp64, n::Int, T_prec::DataType)
    A_clean = (T_prec == Float64) ? A_spd_fp64 : T_prec.(A_spd_fp64)
    A_perf = copy(A_clean)
    backend = KernelAbstractions.get_backend(A_perf)

    op = () -> CUSOLVER.potrf!('L', A_perf)
    reset_op = () -> copyto!(A_perf, A_clean)
    
    min_time_ns = benchmark_op(op, reset_op, backend)
    runtime_ms = min_time_ns / 1_000_000
    gflops = ( (1/3) * n^3 ) / min_time_ns

    return runtime_ms, gflops
end

function get_accuracy_pure(A_spd_fp64::CuMatrix, T_prec::DataType)
    # No more scaling! Just a direct cast.
    A_to_factor = T_prec.(A_spd_fp64)
    
    potrf_recursive!(A_to_factor, 4096)
    A_tri = tril(A_to_factor)
    A_reconstructed = Float64.(A_tri * Transpose(A_tri))
    
    A_to_factor = nothing
    A_tri = nothing
    GC.gc(true); CUDA.reclaim()
    
    # Pure, unified math for the relative error:
    orig_norm = norm(A_spd_fp64)
    error_norm = norm(A_reconstructed .- A_spd_fp64)
    
    return max(error_norm / orig_norm, 1e-20)
end

function get_accuracy_mixed(A_spd_fp64::CuMatrix, precisions::Vector)
    A_mixed_input = SymmMixedPrec(copy(A_spd_fp64), 'L'; precisions=precisions)
    
    try
        potrf_recursive!(A_mixed_input)
    catch e
        if isa(e, LinearAlgebra.PosDefException)
            return NaN
        else
            rethrow(e)
        end
    end

    A_reconstructed = Float64.(tril(reconstruct_matrix(A_mixed_input)) * tril(reconstruct_matrix(A_mixed_input))')
    error_norm = norm(A_reconstructed - A_spd_fp64)
    orig_norm = norm(A_spd_fp64)
    return max(error_norm / orig_norm, 1e-20)
end

function get_accuracy_cusolver(A_spd_fp64::CuMatrix, T_prec::DataType)
    A_to_factor = T_prec.(A_spd_fp64)
    CUSOLVER.potrf!('L', A_to_factor)
    A_reconstructed = Float64.(tril(A_to_factor) * tril(A_to_factor)')
    error_norm = norm(A_reconstructed - A_spd_fp64)
    orig_norm = norm(A_spd_fp64)
    return max(error_norm / orig_norm, 1e-20)
end

function get_runtime_adaptive(A_spd_fp64, n::Int, epsilon_target::Float64, base_case_size::Int)
    available_precisions = [3, 4, 5]
    precision_strings = adaptive_precision_LT(Array(A_spd_fp64), available_precisions, base_case_size, epsilon_target)
    prec_list = [PRECISION_MAP[s] for s in precision_strings]
    push!(prec_list, Float64)
    @printf("      -> Adaptively selected: %s\n", prec_list)
    return get_runtime_mixed(A_spd_fp64, n, prec_list)
end

function get_accuracy_adaptive(A_spd_fp64, epsilon_target::Float64, base_case_size::Int)
    available_precisions = [3, 4, 5]
    precision_strings = adaptive_precision_LT(Array(A_spd_fp64), available_precisions, base_case_size, epsilon_target)
    prec_list = [PRECISION_MAP[s] for s in precision_strings]
    push!(prec_list, Float64)
    @printf("      -> Adaptively selected: %s\n", prec_list)
    return get_accuracy_mixed(A_spd_fp64, prec_list)
end

function run_all_cholesky_tests()
    base_case_size = 256

    matrices_to_test = [  
        # "Zhao/zhao2",
        # "HB/saylr3",
        # "DRIVCAV/cavity18",        
        # "HB/psmigr_1",            
        # "LeGresley/LeGresley_2508",
        # "FIDAP/ex37",
        # "HB/1138_bus",              
        # "HB/bcsstk08",
        "ACUSIM/Pres_Poisson",
        "Boeing/msc01440",
        # "Cannizzo/sts4098",
        "Cylshell/s1rmq4m1",
        "Cylshell/s3rmq4m1",
        "Cylshell/s3rmt3m3",
        "HB/bcsstk15",
        "HB/bcsstk26",
        "HB/bcsstk27",
        "HB/bcsstm26",
        "Nasa/nasa2146",
        "Pothen/bodyy4",
        "Pothen/bodyy5",
        "Simon/raefsky4"
    ]

    pure_scenarios = Dict(
        "Pure F32" => [Float32],
        "Pure F64" => [Float64],
        "Pure F16" => [Float16],
    )
    mixed_scenarios = Dict(
        "[F16, F32]" => [Float16, Float32],
        "[F16, F16, F32]" => [Float16, Float16, Float32],
        "[F16, F16, F16, F32]" => [Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float32],
        "[F32, F32, F64]" => [Float32, Float32, Float64],
        "[F32, F32, F32, F64]" => [Float32, Float32, Float32, Float64],
        "[F16, F32, F64]" => [Float16, Float32, Float64],
        "[F32, F64]" => [Float32, Float64],
    )
    cusolver_scenarios = Dict(
        "CUSOLVER F32" => Float32,
        "CUSOLVER F64" => Float64,
    )
    adaptive_scenarios = Dict(
        # "Adaptive: eps=1e-4"  => 1e-4,
        # "Adaptive: eps=1e-6"  => 1e-6,
        # "Adaptive: eps=1e-8"  => 1e-8,
        # "Adaptive: eps=1e-10" => 1e-10,
    )

    println("🚀 Starting Cholesky Benchmark and Accuracy Suite...")

    for matrix_name in matrices_to_test
        println("\n" * "="^80)
        println("Testing Matrix: $matrix_name")

        A_raw = matrixdepot(matrix_name)
        A_cpu = Matrix{Float64}(A_raw)
        n = size(A_cpu, 1)
        println("Matrix Size (n x n) = $n x $n")

        max_val = maximum(abs, A_cpu)
        println("Max Absolute Value: $max_val")

        A_gpu = CuArray(A_cpu)
        # A_spd_fp64 = A_gpu' * A_gpu + (n * 0.01) * I
        # A_spd_fp64 = (A_gpu + transpose(A_gpu)) ./ 2.0
        A_spd_fp64 = A_gpu
        
        A_raw = nothing
        A_cpu = nothing
        A_gpu = nothing
        GC.gc(true); CUDA.reclaim()

        println("\n--- 🏃 Performance Benchmarks ---")
        
        println("\n  --- Pure Precision ---")
        for (name, precisions) in pure_scenarios
            runtime_ms, gflops = get_runtime_pure(A_spd_fp64, n, precisions[1])
            @printf("    %-25s | Runtime: %8.3f ms | GFLOPS: %8.2f\n", name, runtime_ms, gflops)
        end

        println("\n  --- Manual Mixed Precision ---")
        for (name, precisions) in mixed_scenarios
            runtime_ms, gflops = get_runtime_mixed(A_spd_fp64, n, precisions)
            @printf("    %-25s | Runtime: %8.3f ms | GFLOPS: %8.2f\n", name, runtime_ms, gflops)
        end

        println("\n  --- Adaptive Mixed Precision ---")
        for (name, epsilon) in adaptive_scenarios
            runtime_ms, gflops = get_runtime_adaptive(A_spd_fp64, n, epsilon, base_case_size)
            @printf("    %-25s | Runtime: %8.3f ms | GFLOPS: %8.2f\n", name, runtime_ms, gflops)
        end
        
        println("\n  --- Standard CUSOLVER ---")
        for (name, T_prec) in cusolver_scenarios
            runtime_ms, gflops = get_runtime_cusolver(A_spd_fp64, n, T_prec)
            @printf("    %-25s | Runtime: %8.3f ms | GFLOPS: %8.2f\n", name, runtime_ms, gflops)
        end

        println("\n--- ✅ Accuracy Checks ---")

        println("\n  --- Pure Precision ---")
        for (name, precisions) in pure_scenarios
            rel_err = get_accuracy_pure(A_spd_fp64, precisions[1])
            @printf("    %-25s | Rel. Error: %9.2e\n", name, rel_err)
        end

        println("\n  --- Manual Mixed Precision ---")
        for (name, precisions) in mixed_scenarios
            rel_err = get_accuracy_mixed(A_spd_fp64, precisions)
            @printf("    %-25s | Rel. Error: %9.2e\n", name, rel_err)
        end

        println("\n  --- Adaptive Mixed Precision ---")
        for (name, epsilon) in adaptive_scenarios
            rel_err = get_accuracy_adaptive(A_spd_fp64, epsilon, base_case_size)
            @printf("    %-25s | Rel. Error: %9.2e\n", name, rel_err)
        end

        println("\n  --- Standard CUSOLVER ---")
        for (name, T_prec) in cusolver_scenarios
            rel_err = get_accuracy_cusolver(A_spd_fp64, T_prec)
            @printf("    %-25s | Rel. Error: %9.2e\n", name, rel_err)
        end
    end
    
    println("\n" * "="^80)
    println("✅ All tests complete.")
    println("="^80)
end

run_all_cholesky_tests()


