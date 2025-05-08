@testset "Accuracy Test for unified_rectrxm!" begin
    # Matrix sizes to test
    sizes = [16, 32, 128, 256, 512, 1024, 2048] #, 4096] #, 250, 275, 300, 325, 350, 750] #512, 1024, 2048, 64, 8192, 

    # Number of columns/rows in B to test
    m_sizes = [1, 8, 64, 256, 350]  #2, 4, 16, 32, 128, 256
    
    # Tolerance for accuracy check
    tolerance = 1e-2

    for n in sizes
        for m in m_sizes
            for side in ['L', 'R']
                for uplo in ['L', 'U']
                    for trans in ['N'] #, 'T', 'C']
                        for func in ['S'] #, 'M']
                            for alpha in [1.0]
                                # Skip testing 'M' if the side is not 'L'
                                # if func == 'M' && side == 'R'
                                #     continue
                                # end

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
