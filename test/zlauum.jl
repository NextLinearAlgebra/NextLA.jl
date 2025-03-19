@testset "zlauum test" begin
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        for uplo in ['U', 'L']
            # Test different matrix sizes including edge cases
            for n in [16, 32, 64, 128, 256]
                # Test a variety of block sizes including edge cases
                for block_size in [2, 3, 4, 5, 8]
                    if uplo == 'U'
                        # Create an upper triangular matrix with values centered around 0.5
                        A = Matrix(UpperTriangular(0.5 .+ rand(T, n, n)))
                    else
                        # Create a lower triangular matrix with values centered around -0.5
                        A = Matrix(LowerTriangular(-0.5 .+ rand(T, n, n)))
                    end
                    Ac = copy(A)             
                    info = zlauum(uplo, n, A, n, block_size)                  
                    @test info == 0  # Ensure no error from zlauum
                    # Set tolerance based on type
                    tolerance = T <: Union{Float64, ComplexF64} ? 1e-12 : 1e-6
                    if uplo == 'U'
                        expected_result = Matrix(UpperTriangular(Ac * Ac'))
                        result_diff = norm(Matrix(A) - expected_result) / n
                        @test result_diff < tolerance  # Use adjusted tolerance
                        if result_diff >= tolerance
                            println("Failure in zlauum test for T: $T, uplo: $uplo, n: $n, block_size: $block_size")
                            println("Difference norm: $result_diff")
                        end
                    else
                        expected_result = Matrix(LowerTriangular(Ac' * Ac))
                        result_diff = norm(Matrix(A) - expected_result) / n
                        @test result_diff < tolerance  # Use adjusted tolerance
                        if result_diff >= tolerance
                            println("Failure in zlauum test for T: $T, uplo: $uplo, n: $n, block_size: $block_size")
                            println("Difference norm: $result_diff")
                        end
                    end
                end
            end
        end
    end
end