using Test, AMDGPU, rocBLAS, LinearAlgebra, Printf, Plots, KernelAbstractions

include("benchmark.jl")

function reconstruct_matrix(A::SymmMixedPrec{T_Base}, backend) where {T_Base}
    if A.BaseCase !== nothing
        return copy(A.BaseCase)
    end
    
    C11 = reconstruct_matrix(A.A11, backend)
    C22 = reconstruct_matrix(A.A22, backend)
    C21 = A.OffDiag
    n1, m1 = size(C11)
    n2, m2 = size(C22)
    n = n1 + n2

    C_full = KernelAbstractions.allocate(backend, T_Base, n, n)
    C_full[1:n1, 1:m1] .= C11
    C_full[n1+1:n, 1:m1] .= C21
    C_full[n1+1:n, m1+1:n] .= C22
    C_full[1:n1, m1+1:n] .= transpose(C21)

    return C_full
end


function run_recsyrk_accuracy_benchmark()
    n_values = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    backend = AMDGPU.rocbackend()

    test_scenarios = Dict(
        "Pure F16"                  => [Float16, Float16, Float16],
        "Pure F32"                  => [Float32, Float32, Float32],
        "Pure F64"                  => [Float64, Float64, Float64],
        "[F32, F32, F64]"           => [Float32, Float32, Float64],
        "[F32, F32, F64, F64]"      => [Float32, Float32, Float64, Float64],
        "[F64, F64, F32, F32]"      => [Float64, Float64, Float32, Float32],
        "[F32, F64, F64]"           => [Float32, Float64, Float64],
        "[F16, F16, F32]"           => [Float16, Float16, Float32],
        "[F16, F16, F16, F32]"      => [Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float32, Float64],
        "[F16, F32, F32]"           => [Float16, Float32, Float32]
    )
    
    accuracy_results = Dict(name => Float64[] for name in keys(test_scenarios))
    
    println("🚀 Starting recsyrk! Accuracy Benchmark...")

    for n in n_values
        println("\n" * "-"^50)
        println("Benchmarking Accuracy for C(n x n)=$n, A(n x n)=$n")
        
        for (name, precisions) in test_scenarios
            T_out = precisions[end]
            alpha, beta = -1.0, 1.0

            A_cpu = randn(T_out, n, n) .* 0.1f0
            C_cpu = zeros(T_out, n, n)
            d_A = KernelAbstractions.allocate(backend, T_out, n, n)
            copyto!(d_A, A_cpu)
            d_C_orig = KernelAbstractions.allocate(backend, T_out, n, n)
            copyto!(d_C_orig, C_cpu)

            h_A_fp64 = Array{Float64}(d_A)
            h_C_ground_truth = zeros(Float64, n, n)
            
            LinearAlgebra.BLAS.syrk!('L', 'N', Float64(alpha), h_A_fp64, Float64(beta), h_C_ground_truth)
            h_A_fp64 = nothing

            C_for_custom = copy(d_C_orig)
            C_custom_result_gpu = if name in ["Pure F16", "Pure F32", "Pure F64"]
                recsyrk!(T_out(alpha), d_A, T_out(beta), C_for_custom, 256)
                C_for_custom
            else
                C_mixed = SymmMixedPrec(C_for_custom, 'L'; precisions=precisions)
                recsyrk!(T_out(alpha), d_A, T_out(beta), C_mixed)
                reconstruct_matrix(C_mixed, backend)
            end

            h_C_custom_result = Array(C_custom_result_gpu)

            error_norm = norm(tril(h_C_custom_result) - tril(h_C_ground_truth))
            solution_norm = norm(tril(h_C_ground_truth))
            relative_error = max(error_norm / solution_norm, 1e-20)
            
            push!(accuracy_results[name], -log10(max(relative_error, 1e-18)))

            @printf("           %-28s | Rel. Error: %9.2e\n", name, relative_error)

            if name in ["Pure F16", "Pure F32", "Pure F64"]
                local C_for_blas_result
                if T_out == Float16
                    C_for_blas = KernelAbstractions.zeros(backend, Float32, size(d_C_orig)...)
                    gemm!('N', 'T', T_out(alpha), d_A, d_A, T_out(beta), C_for_blas)
                    C_for_blas_result = T_out.(C_for_blas)
                else
                    C_for_blas = copy(d_C_orig)
                    syrk!('L', 'N', T_out(alpha), d_A, T_out(beta), C_for_blas)
                    C_for_blas_result = C_for_blas
                end

                h_C_blas_result = Array(C_for_blas_result)
                blas_error_norm = norm(tril(h_C_blas_result) - tril(h_C_ground_truth))
                blas_relative_error = max(blas_error_norm / solution_norm, 1e-20)
                
                @printf("           %-28s | Rel. Error: %9.2e\n", "Vendor BLAS " * name, blas_relative_error)
            end

            d_A = nothing
            d_C_orig = nothing
            C_for_custom = nothing
            C_custom_result_gpu = nothing
            h_C_custom_result = nothing
            h_C_ground_truth = nothing
            GC.gc(true)
        end
    end

    println("\n" * "="^60)
    println("📊 Generating and saving accuracy plot...")

    acc_plot = plot(title="Accuracy vs. Matrix Size", xlabel="Matrix Size (n)", ylabel="-log10(Relative Error)", legend=:outertopright, xaxis=:log2)
    
    for (name, acc_values) in accuracy_results
        if name != "Pure Float64" 
            plot!(acc_plot, n_values[1:length(acc_values)], acc_values, label=name, marker=:auto)
        end
    end
    
    savefig(acc_plot, "recsyrk_accuracy.png")

    println("✅ Accuracy benchmark complete. Plot saved to recsyrk_accuracy.png")
    println("="^60)
end

run_recsyrk_accuracy_benchmark()