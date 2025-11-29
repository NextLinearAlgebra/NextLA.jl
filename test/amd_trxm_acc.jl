using Test, AMDGPU, LinearAlgebra, Printf, KernelAbstractions

function check_accuracy()
    sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    uplo = 'U'
    side = 'L'
    alpha = 1.0f0
    trans = 'N'
    backend = AMDGPU.ROCBackend()

    test_scenarios = Dict(
        "Recursive Float64" => [Float64],
        "Recursive Float32" => [Float32],
        "Recursive Float16" => [Float16],
        "TriMixed: [F16, F64]" => [Float16, Float64],
        "TriMixed: [F64, F16]" => [Float64, Float16],
        "TriMixed: [F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float32],
        "TriMixed: [F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float32, Float64],
        "TriMixed: [F16, F16, F32]" => [Float16, Float16, Float32],
        "TriMixed: [F32, F64]" => [Float32, Float64],
        "TriMixed: [F16, F32]" => [Float16, Float32],
        "TriMixed: [F16, F64]" => [Float16, Float64],
        "TriMixed: [F32, F32, F64]" => [Float32, Float32, Float64],
        "TriMixed: [F32, F32, F32, F64]" => [Float32, Float32, Float32, Float64],
        "TriMixed: [F32, F32, F32, F32, F64]" => [Float32, Float32, Float32, Float32, Float64],
    )

    for func in ['S']
        op_name = func == 'S' ? "TRSM" : "TRMM"
        println("\n" * "="^70)
        println("🔬 Starting Accuracy Check for $op_name (uplo='$uplo')...")
        println("="^70)

        for n in sizes
            println("\n--- Testing Matrix Size: $n x $n ---")

            A_cpu = Matrix(UpperTriangular(rand(Float64, n, n)))
            A_cpu .+= Diagonal(fill(Float64(n), n))
            B_cpu = rand(Float64, n, n)

            A_sol_gpu = KernelAbstractions.allocate(backend, Float64, n, n)
            copyto!(A_sol_gpu, A_cpu)
            B_sol_gpu = KernelAbstractions.allocate(backend, Float64, n, n)
            copyto!(B_sol_gpu, B_cpu)

            if func == 'S'
                trsm!(side, uplo, trans, 'N', alpha, A_sol_gpu, B_sol_gpu)
            else
                trmm!(side, uplo, trans, 'N', alpha, A_sol_gpu, B_sol_gpu, B_sol_gpu)
            end

            for (name, prec_list) in test_scenarios
                T_Base = prec_list[1]
                
                A_test_gpu = KernelAbstractions.allocate(backend, Float64, n, n)
                copyto!(A_test_gpu, A_cpu)
                B_test_gpu = KernelAbstractions.allocate(backend, T_Base, n, n)
                copyto!(B_test_gpu, B_cpu)

                if startswith(name, "Recursive")
                    unified_rectrxm!(side, uplo, trans, alpha, func, T_Base.(A_test_gpu), B_test_gpu)
                else
                    A_mixed = TriMixedPrec(A_test_gpu, uplo; precisions=prec_list)
                    unified_rectrxm!(side, uplo, trans, alpha, func, A_mixed, B_test_gpu)
                end

                error_norm = norm(Float64.(B_test_gpu) .- B_sol_gpu)
                solution_norm = norm(B_sol_gpu)
                relative_error = iszero(solution_norm) ? 0.0 : error_norm / solution_norm
                
                @printf "   %-40s | Relative Error: %.3e\n" "'$name'" relative_error
            end
        end
    end
end

println("Running Accuracy Checks...")
check_accuracy()
println("\n✅ Accuracy checks complete.")