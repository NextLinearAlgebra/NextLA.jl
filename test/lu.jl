using Test
using LinearAlgebra
using CUDA

# @testset "lu! with varing pivot" begin
#     for T in [Float32, Float64, ComplexF32, ComplexF64]
#         for pivot in [NoPivot(), RowNonZero(), CompletePivoting()]
#             for m in [10, 100, 1000]
#                 for n in [m, div(m,10)*9, div(m,10)*11]
#                     A = rand(T, m, n)
#                     B = copy(A)
#                     DLA_A = DLAMatrix{T}(A)
#                     F =LinearAlgebra.lu!(DLA_A, pivot)
#                     m, n = size(F.factors)
#                     L = tril(F.factors[1:m, 1:min(m,n)])
#                     for i in 1:min(m,n); L[i,i] = 1 end
#                     U = triu(F.factors[1:min(m,n), 1:n])
#                     p = LinearAlgebra.ipiv2perm(F.ipiv,m)
#                     q = LinearAlgebra.ipiv2perm(F.jpiv, n)
#                     L * U ≈ B[p, q]
#                     norm(L * U) ≈ norm(B[p, q])
#                 end
#             end
#         end
#     end
# end


# @testset "lu generic_lufact!" begin
#     for T in [Float32, Float64, ComplexF32, ComplexF64]
#         for pivot in [NoPivot(), RowNonZero(),  RowMaximum(), CompletePivoting()]
#             for m in [10, 100, 1000]
#                 for n in [m, div(m,10)*9, div(m,10)*11]
#                     A = rand(T, m, n)
#                     B = copy(A)
#                     DLA_A = DLAMatrix{T}(A)
#                     F =LinearAlgebra.generic_lufact!(DLA_A, pivot)
#                     m, n = size(F.factors)
#                     L = tril(F.factors[1:m, 1:min(m,n)])
#                     for i in 1:min(m,n); L[i,i] = 1 end
#                     U = triu(F.factors[1:min(m,n), 1:n])
#                     p = LinearAlgebra.ipiv2perm(F.ipiv,m)
#                     q = LinearAlgebra.ipiv2perm(F.jpiv, n)
#                     L * U ≈ B[p, q]
#                     norm(L * U) ≈ norm(B[p, q])
#                 end
#             end
#         end
#     end
# end


# @testset "lu! default RowMaximum()" begin
#     for T in [Float32, Float64, ComplexF32, ComplexF64]
#         for m in [10, 100, 1000]
#             for n in [m, div(m,10)*9, div(m,10)*11]
#                 A = rand(T, m, n)
#                 B = copy(A)
#                 DLA_A = DLAMatrix{T}(A)
#                 F =LinearAlgebra.lu!(DLA_A)
#                 m, n = size(F.factors)
#                 L = tril(F.factors[1:m, 1:min(m,n)])
#                 for i in 1:min(m,n); L[i,i] = 1 end
#                 U = triu(F.factors[1:min(m,n), 1:n])
#                 p = LinearAlgebra.ipiv2perm(F.ipiv,m)
#                 L * U ≈ B[p, :]
#                 norm(L * U) ≈ norm(B[p, :])
#             end
#         end
#     end
# end

# @testset "zlauum test" begin
#     for T in [Float32, Float64, ComplexF32, ComplexF64]
#         for uplo in ['U', 'L']
#             # Test different matrix sizes including edge cases
#             for n in [16, 32, 64, 128, 256]
#                 # Test a variety of block sizes including edge cases
#                 for block_size in [2, 3, 4, 5, 8]
#                     if uplo == 'U'
#                         # Create an upper triangular matrix with values centered around 0.5
#                         A = Matrix(UpperTriangular(0.5 .+ rand(T, n, n)))
#                     else
#                         # Create a lower triangular matrix with values centered around -0.5
#                         A = Matrix(LowerTriangular(-0.5 .+ rand(T, n, n)))
#                     end
#                     Ac = copy(A)
                
#                     info = zlauum(uplo, n, A, n, block_size)
                    
#                     @test info == 0  # Ensure no error from zlauum

#                     # Set tolerance based on type
#                     tolerance = T <: Union{Float64, ComplexF64} ? 1e-12 : 1e-6

#                     if uplo == 'U'
#                         expected_result = Matrix(UpperTriangular(Ac * Ac'))
#                         result_diff = norm(Matrix(A) - expected_result) / n

#                         @test result_diff < tolerance  # Use adjusted tolerance
#                         if result_diff >= tolerance
#                             println("Failure in zlauum test for T: $T, uplo: $uplo, n: $n, block_size: $block_size")
#                             println("Difference norm: $result_diff")
#                         end
#                     else
#                         expected_result = Matrix(LowerTriangular(Ac' * Ac))
#                         result_diff = norm(Matrix(A) - expected_result) / n

#                         @test result_diff < tolerance  # Use adjusted tolerance
#                         if result_diff >= tolerance
#                             println("Failure in zlauum test for T: $T, uplo: $uplo, n: $n, block_size: $block_size")
#                             println("Difference norm: $result_diff")
#                         end
#                     end
#                 end
#             end
#         end
#     end
# end

using CUDA
using LinearAlgebra
using Test

@testset "Accuracy Test for unified_rectrxm!" begin
    # Matrix sizes to test
    sizes = [16, 32, 128, 256, 2048, 4096, 250, 275, 300, 325, 350, 750] #512, 1024, 2048, 64, 8192, 

    # Number of columns/rows in B to test
    m_sizes = [1, 8, 64, 256]  #2, 4, 16, 32, 128, 256
    
    # Tolerance for accuracy check
    tolerance = 1e-14

    for n in sizes
        for m in m_sizes
            for side in ['L', 'R']
                for uplo in ['L', 'U']
                    for trans in ['N', 'T', 'C']
                        for func in ['S', 'M']
                            for alpha in [1.0]
                                # Skip testing 'M' if the side is not 'L'
                                if func == 'M' && side == 'R'
                                    continue
                                end

                                # Log the test configuration
                                println("Testing FUNC: $func ; side: $side, uplo: $uplo, trans: $trans, alpha: $alpha, n: $n, m: $m")

                                # Generate the triangular matrix A based on `uplo`
                                if uplo == 'L'
                                    # Lower triangular matrix
                                    A = Matrix(LowerTriangular(rand(n, n) .+ 1))
                                else
                                    # Upper triangular matrix
                                    A = Matrix(UpperTriangular(rand(n, n) .+ 1))
                                end

                                # Add a diagonal to ensure the matrix is well-conditioned
                                A += Diagonal(10 * ones(n, n))

                                # Convert A to a CuArray for GPU computation
                                A_gpu = CuArray(A)

                                # Generate the B matrix based on the `side`
                                if side == 'L'
                                    B = Matrix(rand(n, m) .+ 1)  # B has n rows
                                else
                                    B = Matrix(rand(m, n) .+ 1)  # B has n columns
                                end

                                # Create copies of A and B for baseline and comparison
                                Ac = copy(A)
                                Bc = copy(B)
                                B_gpu = CuArray(B)
                                A_gpu_before = copy(A_gpu)

                                # Perform the GPU operation using `unified_rectrxm!`
                                unified_rectrxm!(side, uplo, trans, alpha, func, A_gpu, B_gpu)

                                # Perform the baseline operation using BLAS `trsm!` or `trmm!`
                                if func == 'S'
                                    # Solve triangular system: A * X = B or X * A = B
                                    CUBLAS.BLAS.trsm!(side, uplo, trans, 'N', alpha, Ac, Bc)
                                elseif func == 'M'
                                    # Matrix multiply with triangular matrix: B = alpha * A * B
                                    CUBLAS.BLAS.trmm!(side, uplo, trans, 'N', alpha, Ac, Bc)
                                end

                                # Compute the Frobenius norm difference (relative error)
                                result_diff = norm(Matrix(B_gpu) - Bc) / norm(Bc)

                                # Log the result difference
                                println("Size: $n x $n, B size: $(size(B)) | Result Diff (Relative Error): $result_diff")

                                # Handle NaN results (indicating an error in the computation)
                                if isnan(result_diff)
                                    println("GOT NAN..... SKIPPING FOR NOW")
                                end

                                # Check if the relative error exceeds the tolerance
                                if result_diff >= tolerance
                                    println("Test failed for matrix size $n x $n, B size: $(size(B)), trans: $trans")
                                    println("Relative error: $result_diff")
                                end

                                # Assert that the relative error is within the tolerance
                                @test result_diff < tolerance
                            end
                        end
                    end
                end
            end
        end
    end
end


# @testset "Equivalence Test for TRSM: All Cases" begin
#     # Matrix sizes to test
#     sizes = [16, 32, 128, 256, 2048]
    
#     # Number of columns in B to test
#     m_sizes = [1, 8, 64]
    
#     # Tolerance for accuracy check
#     tolerance = 1e-12

#     cases = [
#         ("Left Upper", left_upper_no_transpose, left_upper_transpose),
#         ("Left Lower", left_lower_no_transpose, left_lower_transpose),
#         ("Right Upper", right_upper_no_transpose, right_upper_transpose),
#         ("Right Lower", right_lower_no_transpose, right_lower_transpose)
#     ]

#     for (case_name, no_transpose_func, transpose_func) in cases
#         @testset "$case_name" begin
#             for n in sizes
#                 for m in m_sizes
#                     # Generate appropriate triangular matrix A
#                     A = if startswith(case_name, "Left")
#                         if contains(case_name, "Upper")
#                             Matrix(UpperTriangular(rand(n, n) .+ 1))
#                         else
#                             Matrix(LowerTriangular(rand(n, n) .+ 1))
#                         end
#                     else
#                         if contains(case_name, "Upper")
#                             Matrix(UpperTriangular(rand(m, m) .+ 1))
#                         else
#                             Matrix(LowerTriangular(rand(m, m) .+ 1))
#                         end
#                     end
#                     A += Diagonal(10 * ones(size(A, 1)))  # Ensure well-conditioned
                    
#                     # Generate B matrix
#                     B = if startswith(case_name, "Left")
#                         rand(n, m) .+ 1
#                     else
#                         rand(n, m) .+ 1
#                     end
                    
#                     # Create copies for the two cases
#                     A_no_transpose = CuArray(A)
#                     B_no_transpose = CuArray(copy(B))
                    
#                     A_transpose = CuArray(A)
#                     B_transpose = CuArray(copy(B))
                    
#                     # Apply no_transpose function
#                     no_transpose_func(A_no_transpose, B_no_transpose)
                    
#                     # Apply transpose function
#                     transpose_func(A_transpose, B_transpose)
                    
#                     # Compare results
#                     result_diff = norm(Matrix(B_no_transpose) - Matrix(B_transpose)) / norm(Matrix(B_no_transpose))
                    
#                     @test result_diff < tolerance
                    
#                     if result_diff >= tolerance
#                         println("Test failed for $case_name, matrix size $(size(A)), B size: $(size(B))")
#                         println("Relative error: $result_diff")
#                     else
#                         println("Test passed for $case_name, matrix size $(size(A)), B size: $(size(B))")
#                         println("Relative error: $result_diff")
#                     end
#                 end
#             end
#         end
#     end
# end