using Test, CUDA, LinearAlgebra, Printf, Plots, KernelAbstractions

include("benchmark.jl") 


function reconstruct_matrix(A::SymmMixedPrec{T_Base}) where {T_Base}
    if A.BaseCase !== nothing
        return copy(A.BaseCase)
    end
    
    C11 = reconstruct_matrix(A.A11)
    C22 = reconstruct_matrix(A.A22)
    C21 = A.OffDiag
    n1, m1 = size(C11)
    n2, m2 = size(C22)
    n = n1 + n2

    C_full = CuArray{T_Base}(undef, n, n)
    C_full[1:n1, 1:m1] .= C11
    C_full[n1+1:n, 1:m1] .= C21
    C_full[n1+1:n, m1+1:n] .= C22
    C_full[1:n1, m1+1:n] .= transpose(C21)

    return C_full
end


function run_recsyrk_accuracy_benchmark()
    n_values = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

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

            d_A = CuArray(randn(T_out, n, n) .* 0.1f0)
            d_C_orig = CuArray(zeros(T_out, n, n))

            d_A_fp64 = CuArray{Float64}(d_A)
            d_C_ground_truth = CuArray(zeros(Float64, n, n))
            CUBLAS.syrk!('L', 'N', Float64(alpha), d_A_fp64, Float64(beta), d_C_ground_truth)

            C_for_custom = copy(d_C_orig)
            C_custom_result = if name in ["Pure F16", "Pure F32", "Pure F64"]
                recsyrk!(T_out(alpha), d_A, T_out(beta), C_for_custom, 256)
                C_for_custom
            else
                C_mixed = SymmMixedPrec(C_for_custom, 'L'; precisions=precisions)
                recsyrk!(T_out(alpha), d_A, T_out(beta), C_mixed)
                reconstruct_matrix(C_mixed)
            end

            error_norm = norm(tril(CuArray{Float64}(C_custom_result)) - tril(d_C_ground_truth))
            solution_norm = norm(tril(d_C_ground_truth))
            relative_error = max(error_norm / solution_norm, 1e-20)
            
            push!(accuracy_results[name], -log10(max(relative_error, 1e-18)))

            @printf("    %-28s | Rel. Error: %9.2e\n", name, relative_error)
        end
    end

    println("\n" * "="^60)
    println("📊 Generating and saving accuracy plot...")

    acc_plot = plot(title="Accuracy vs. Matrix Size", xlabel="Matrix Size (n)", ylabel="-log10(Relative Error)", legend=:outertopright, xaxis=:log2)
    
    for (name, acc_values) in accuracy_results
        if name != "Pure Float64" 
            plot!(acc_plot, n_values, acc_values, label=name, marker=:auto)
        end
    end
    
    savefig(acc_plot, "recsyrk_accuracy.png")

    println("✅ Accuracy benchmark complete. Plot saved to recsyrk_accuracy.png")
    println("="^60)
end

run_recsyrk_accuracy_benchmark()
