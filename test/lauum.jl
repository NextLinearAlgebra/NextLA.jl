@testset "LAUUM" begin
    @testset "$T" for T in TEST_TYPES
        rtol = test_rtol(T)

        @testset "uplo=$uplo, n=$n, ib=$ib" for uplo in ['U', 'L'],
                                                 n  in [16, 32, 64, 128],
                                                 ib in [2, 4, 8]
            A = if uplo == 'U'
                Matrix(UpperTriangular(0.5 .+ rand(T, n, n)))
            else
                Matrix(LowerTriangular(-0.5 .+ rand(T, n, n)))
            end
            A_orig = copy(A)

            NextLA.lauum!(uplo, n, A, ib)

            # Reference: U*Uᴴ (upper) or Lᴴ*L (lower), keep only the relevant triangle
            if uplo == 'U'
                expected = Matrix(UpperTriangular(A_orig * A_orig'))
            else
                expected = Matrix(LowerTriangular(A_orig' * A_orig))
            end

            @test norm(A - expected) / n < rtol
        end
    end

    @testset "Error handling" begin
        @test_throws ArgumentError NextLA.lauum!('X', 4, zeros(4, 4), 2)
        @test_throws ArgumentError NextLA.lauum!('U', -1, zeros(0, 0), 2)
    end
end