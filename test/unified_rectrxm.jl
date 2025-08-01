# @testset "Accuracy Test for unified_rectrxm!" begin
#     # Matrix sizes to test
#     sizes = [16, 32, 128, 256, 512, 1024, 2048, 4096] #, 4096] #, 250, 275, 300, 325, 350, 750] #512, 1024, 2048, 64, 8192, 

#     # Number of columns/rows in B to test
#     m_sizes = [256] #[1, 8, 64, 256, 350]  #2, 4, 16, 32, 128, 256
    
#     # Tolerance for accuracy check
#     tolerance = 1e-2

#     for n in sizes
#         for m in m_sizes
#             for side in ['L', 'R']
#                 for uplo in ['L', 'U']
#                     for trans in ['N', 'T'] #, 'C']
#                         for func in ['S'] #, 'M']
#                             for alpha in [1.0]
#                                 # Skip testing 'M' if the side is not 'L'
#                                 # if func == 'M' && side == 'R'
#                                 #     continue
#                                 # end

#                                 # Log the test configuration
#                                 println("Testing FUNC: $func ; side: $side, uplo: $uplo, trans: $trans, alpha: $alpha, n: $n, m: $m")

#                                 # Generate the triangular matrix A based on `uplo`
#                                 if uplo == 'L'
#                                     # Lower triangular matrix
#                                     A = Matrix(LowerTriangular(rand(n, n) .+ 1))
#                                 else
#                                     # Upper triangular matrix
#                                     A = Matrix(UpperTriangular(rand(n, n) .+ 1))
#                                 end

#                                 # Add a diagonal to ensure the matrix is well-conditioned
#                                 A += Diagonal(10 * ones(n, n))

#                                 # Convert A to a CuArray for GPU computation
#                                 A_gpu = CuArray(A)

#                                 # Generate the B matrix based on the `side`
#                                 if side == 'L'
#                                     B = Matrix(rand(n, m) .+ 1)  # B has n rows
#                                 else
#                                     B = Matrix(rand(m, n) .+ 1)  # B has n columns
#                                 end

#                                 # Create copies of A and B for baseline and comparison
#                                 Ac = copy(A)
#                                 Bc = copy(B)
#                                 B_gpu = CuArray(B)
#                                 A_gpu_before = copy(A_gpu)

#                                 # Perform the GPU operation using `unified_rectrxm!`
#                                 unified_rectrxm!(side, uplo, trans, alpha, func, A_gpu, B_gpu)

#                                 # Perform the baseline operation using BLAS `trsm!` or `trmm!`
#                                 if func == 'S'
#                                     # Solve triangular system: A * X = B or X * A = B
#                                     CUBLAS.BLAS.trsm!(side, uplo, trans, 'N', alpha, Ac, Bc)
#                                 elseif func == 'M'
#                                     # Matrix multiply with triangular matrix: B = alpha * A * B
#                                     CUBLAS.BLAS.trmm!(side, uplo, trans, 'N', alpha, Ac, Bc)
#                                 end

#                                 # Compute the Frobenius norm difference (relative error)
#                                 result_diff = norm(Matrix(B_gpu) - Bc) / norm(Bc)

#                                 # Log the result difference
#                                 println("Size: $n x $n, B size: $(size(B)) | Result Diff (Relative Error): $result_diff")

#                                 # Handle NaN results (indicating an error in the computation)
#                                 if isnan(result_diff)
#                                     println("GOT NAN..... SKIPPING FOR NOW")
#                                 end

#                                 # Check if the relative error exceeds the tolerance
#                                 if result_diff >= tolerance
#                                     println("Test failed for matrix size $n x $n, B size: $(size(B)), trans: $trans")
#                                     println("Relative error: $result_diff")
#                                 end

#                                 # Assert that the relative error is within the tolerance
#                                 @test result_diff < tolerance
#                             end
#                         end
#                     end
#                 end
#             end
#         end
#     end
# end


function verify_precision_layers!(A::TriMixedPrec, precisions::Vector{DataType})
    
    if A.BaseCase !== nothing
        @test length(precisions) == 1
        
        @test eltype(A.BaseCase) == precisions[1]
        return
    end

    
    if A.OffDiag !== nothing
        @test eltype(A.OffDiag) == precisions[1]
    end

    
    if A.A11 !== nothing
        verify_precision_layers!(A.A11, precisions[2:end])
    end
    if A.A22 !== nothing
        verify_precision_layers!(A.A22, precisions[2:end])
    end
end


@testset "TriLayeredPrec Accuracy Test for unified_rectrxm!" begin
    sizes = [256, 512, 1024]
    m_sizes = [1, 8, 64]
    tolerance = 1e-2

    precision_lists = [
        # --- Simple (2-level) cases ---
        [Float64, Float64],
        [Float32, Float64],
        # [Float16, Float64],
        [Float32, Float32],
        # [Float16, Float32],
        # [Float16, Float16],
        # [Float64, Float16],

        # --- Useful (3-level) cases ---
        [Float64, Float32, Float64],
        # [Float16, Float16, Float32],
        # [Float32, Float16, Float64],
        # [Float32, Float16, Float32],
        [Float64, Float64, Float32],
        # [Float64, Float64, Float16],

        # --- More Complicated (4-level) cases ---
        [Float64, Float64, Float32, Float64],
        # [Float64, Float32, Float16, Float64],
        # [Float32, Float32, Float16, Float32],
        # [Float64, Float32, Float32, Float16, Float64]
    ]

    @info "Starting TriLayeredPrec TRSM Accuracy Test Suite..."

    for prec_list in precision_lists
        T_Base = prec_list[end]

        @testset "Precisions: $prec_list" begin
            for n in sizes
                for m in m_sizes
                    for side in ['L', 'R']
                        for uplo in ['L', 'U']
                            for trans in ['N', 'T']
                                for func in ['S'] #, 'M']
                                    for alpha in [1.0]

                                        println("Testing DATA STRUC ; FUNC: $func ; side: $side, trans: $trans, uplo: $uplo, n: $n, m: $m, Precisions=$prec_list")

                                        local A_cpu
                                        if uplo == 'L'
                                            A_cpu = Matrix(LowerTriangular(rand(T_Base, n, n)))
                                        else
                                            A_cpu = Matrix(UpperTriangular(rand(T_Base, n, n)))
                                        end

                                        A_cpu += Diagonal(10 * ones(T_Base, n))

                                        local B_cpu
                                        if side == 'L'
                                            B_cpu = rand(T_Base, n, m)
                                        else
                                            B_cpu = rand(T_Base, m, n)
                                        end

                                        A_blas_gpu = CuArray(A_cpu)
                                        B_solution_gpu = CuArray(copy(B_cpu))
                                        A_gpu = CuArray(A_cpu)
                                        B_gpu = CuArray(copy(B_cpu))

                                        if func == 'S'
                                            if eltype(A_blas_gpu) == Float16
                                                A_f32 = CuArray{Float32}(A_blas_gpu)
                                                B_f32 = CuArray{Float32}(B_solution_gpu)
                            
                                                CUBLAS.trsm!(side, uplo, trans, 'N', Float32(alpha), A_f32, B_f32)
                            
                                                B_solution_gpu .= B_f32
                                            else
                                                # For Float32 and Float64, we can run trsm! directly.
                                                # We just need to ensure `alpha` has the same type as the matrices.
                                                T = eltype(A_blas_gpu)
                                                CUBLAS.trsm!(side, uplo, trans, 'N', T(alpha), A_blas_gpu, B_solution_gpu)
                                            end
                                        elseif func == 'M'
                                            # Matrix multiply with triangular matrix: B = alpha * A * B
                                            if eltype(A_blas_gpu) == Float16
                                                A_f32 = CuArray{Float32}(A_blas_gpu)
                                                B_f32 = CuArray{Float32}(B_solution_gpu)
                            
                                                CUBLAS.trmm!(side, uplo, trans, 'N', Float32(alpha), A_f32, B_f32, B_f32)
                            
                                                B_solution_gpu .= B_f32
                                            else
                                                T = eltype(A_blas_gpu)
                                                CUBLAS.trmm!(side, uplo, trans, 'N', T(alpha), A_blas_gpu, B_solution_gpu, B_solution_gpu)
                                            end
                                        end

                                        A_mixed = TriMixedPrec(A_gpu, uplo; precisions=prec_list)

                                        # verify_precision_layers!(A_mixed, prec_list) # make sure the layers have the right precision

                                        unified_rectrxm!(side, uplo, trans, alpha, func, A_mixed, B_gpu)

                                        error_norm = norm(B_gpu .- B_solution_gpu)
                                        solution_norm = norm(B_solution_gpu)
                                        relative_error = error_norm / solution_norm

                                        println("Size: $n x $n, B size: $(size(B_cpu)) | Relative Error: $relative_error")

                                        # if isnan(relative_error)
                                        #     println("GOT NAN..... SKIPPING FOR NOW")
                                        # end

                                        if relative_error >= tolerance
                                            println("Test failed for matrix size $n x $n, precisions: $prec_list")
                                            println("Relative error: $relative_error")
                                        end

                                        @test relative_error < tolerance
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end


