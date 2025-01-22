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
    sizes = [30, 32, 45, 64, 102, 128, 250, 256, 512, 1024, 2048, 4096, 8192] #350, 750

    # Number of columns/rows in B to test
    m_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]  
    
    # Tolerance for accuracy check
    tolerance = 1e-14

    for n in sizes
        for m in m_sizes
            for side in ['L', 'R']
                for uplo in ['L', 'U']
                    for trans in ['N', 'T', 'C']
                        for alpha in [1.0]
                            println("Testing side: $side, uplo: $uplo, trans: $trans, alpha: $alpha, n: $n, m: $m")
                            
                            # Generate triangular matrix A based on `uplo`
                            if uplo == 'L'
                                A = Matrix(LowerTriangular(rand(n, n) .+ 1))
                            else
                                A = Matrix(UpperTriangular(rand(n, n) .+ 1))
                            end
                            A += Diagonal(10 * ones(n, n))  # Ensure well-conditioned matrix

                            # Convert to CuArray for GPU computation
                            A_gpu = CuArray(A)

                            # Create B matrix based on side
                            if side == 'L'
                                B = Matrix(rand(n, m) .+ 1)
                            else
                                B = Matrix(rand(m, n) .+ 1)
                            end
                            Bc = copy(B)
                            B_gpu = CuArray(B)

                            Ac = copy(A)
                            A_gpu_before = copy(A_gpu)

                            # Perform GPU operation with unified_rectrxm! for TRSM 'S'
                            unified_rectrxm!(side, uplo, trans, alpha, 'S', A_gpu, B_gpu)

                            # Check if A_gpu was mutated
                            A_diff = norm(A_gpu - A_gpu_before)
                            @test A_diff < tolerance

                            # Perform baseline operation with BLAS trsm!
                            CUBLAS.BLAS.trsm!(side, uplo, trans, 'N', alpha, Ac, Bc)

                            # Compute the Frobenius norm difference (relative error)
                            result_diff = norm(Matrix(B_gpu) - Bc) / norm(Bc)

                            println("Size: $n x $n, B size: $(size(B)) | Result Diff (Relative Error): $result_diff")

                            # Fail if result_diff is NaN
                            if isnan(result_diff)
                                println("GOT NAN..... SKIPPING FOR NOW")
                                # error("NaN encountered! Matrix size: $n x $n, B size: $(size(B)), trans: $trans")
                            end

                            # Check if the relative error exceeds tolerance
                            if result_diff >= tolerance
                                println("Test failed for matrix size $n x $n, B size: $(size(B)), trans: $trans")
                                println("Relative error: $result_diff")
                            end

                            # Assert relative error is within tolerance
                            @test result_diff < tolerance

                            # Check if we're testing 'L', 'L', 'N' (Lower, Lower, Non-Transpose)
                            if uplo == 'L' && side == 'L' && trans == 'N'
                                # Test TRMM: Update B as alpha*A*B
                                # Create another pair of matrices for TRMM 'M'
                                A_trmm = copy(A)
                                B_trmm = copy(B)

                                Ac_trmm = copy(A_trmm)
                                Bc_trmm = copy(B_trmm)

                                # Perform the TRMM operation on the GPU
                                unified_rectrxm!(side, uplo, trans, alpha, 'M', A_trmm, B_trmm)

                                # Perform the baseline TRMM operation with CUBLAS
                                CUBLAS.BLAS.trmm!(side, uplo, trans, 'N', alpha, Ac_trmm, Bc_trmm)

                                # Compute the Frobenius norm difference for TRMM
                                trmm_diff = norm(Matrix(B_trmm) - Bc_trmm) / norm(Bc_trmm)

                                println("TRMM Test | Size: $n x $n, B size: $(size(B)) | Result Diff (Relative Error): $trmm_diff")

                                # Fail if result_diff for TRMM is NaN
                                if isnan(trmm_diff)
                                    error("NaN encountered in TRMM test! Matrix size: $n x $n, B size: $(size(B))")
                                end

                                # Check if the relative error exceeds tolerance for TRMM
                                if trmm_diff >= tolerance
                                    println("TRMM Test failed for matrix size $n x $n, B size: $(size(B))")
                                    println("Relative error: $trmm_diff")
                                end

                                # Assert relative error is within tolerance for TRMM
                                @test trmm_diff < tolerance
                            end
                        end
                    end
                end
            end
        end
    end
end
