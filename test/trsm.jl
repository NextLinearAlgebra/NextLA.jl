
@testset "Equivalence Test for TRSM: All Cases" begin
    # Matrix sizes to test
    sizes = [16, 32, 128, 256, 2048]
    # Number of columns in B to test
    m_sizes = [1, 8, 64]
    # Tolerance for accuracy check
    tolerance = 1e-12
    cases = [
        ("Left Upper", left_upper_no_transpose, left_upper_transpose),
        ("Left Lower", left_lower_no_transpose, left_lower_transpose),
        ("Right Upper", right_upper_no_transpose, right_upper_transpose),
        ("Right Lower", right_lower_no_transpose, right_lower_transpose)
    ]
    for (case_name, no_transpose_func, transpose_func) in cases
        @testset "$case_name" begin
            for n in sizes
                for m in m_sizes
                    # Generate appropriate triangular matrix A
                    A = if startswith(case_name, "Left")
                        if contains(case_name, "Upper")
                            Matrix(UpperTriangular(rand(n, n) .+ 1))
                        else
                            Matrix(LowerTriangular(rand(n, n) .+ 1))
                        end
                    else
                        if contains(case_name, "Upper")
                            Matrix(UpperTriangular(rand(m, m) .+ 1))
                        else
                            Matrix(LowerTriangular(rand(m, m) .+ 1))
                        end
                    end
                    A += Diagonal(10 * ones(size(A, 1)))  # Ensure well-conditioned
                    # Generate B matrix
                    B = if startswith(case_name, "Left")
                        rand(n, m) .+ 1
                    else
                        rand(n, m) .+ 1
                    end
                    # Create copies for the two cases
                    A_no_transpose = CuArray(A)
                    B_no_transpose = CuArray(copy(B))
                    A_transpose = CuArray(A)
                    B_transpose = CuArray(copy(B))
                    # Apply no_transpose function
                    no_transpose_func(A_no_transpose, B_no_transpose)
                    # Apply transpose function
                    transpose_func(A_transpose, B_transpose)
                    # Compare results
                    result_diff = norm(Matrix(B_no_transpose) - Matrix(B_transpose)) / norm(Matrix(B_no_transpose))
                    @test result_diff < tolerance
                    if result_diff >= tolerance
                        println("Test failed for $case_name, matrix size $(size(A)), B size: $(size(B))")
                        println("Relative error: $result_diff")
                    else
                        println("Test passed for $case_name, matrix size $(size(A)), B size: $(size(B))")
                        println("Relative error: $result_diff")
                    end
                end
            end
        end
    end
end