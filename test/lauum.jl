for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "LAUUM [$backend_name]" begin
        @testset "$T" for T in TEST_TYPES
                rtol = test_rtol(T)

                @testset "uplo=$uplo, n=$n, ib=$ib" for uplo in ['U', 'L'],
                                                     n  in [16, 32, 64, 128],
                                                     ib in [2, 4, 8]
                    A_init = if uplo == 'U'
                        Matrix(UpperTriangular(0.5 .+ rand(T, n, n)))
                    else
                        Matrix(LowerTriangular(-0.5 .+ rand(T, n, n)))
                    end
                    A_orig = copy(A_init)
                    A = ArrayType(copy(A_init))

                    NextLA.lauum!(uplo, n, A, ib)
                    synchronize(A)

                    if uplo == 'U'
                        expected = Matrix(UpperTriangular(A_orig * A_orig'))
                    else
                        expected = Matrix(LowerTriangular(A_orig' * A_orig))
                    end
                    @test norm(Array(A) - expected) / n < rtol
                end
        end
    end
end

@testset "LAUUM Error handling" begin
    @test_throws ArgumentError NextLA.lauum!('X', 4, zeros(4, 4), 2)
    @test_throws ArgumentError NextLA.lauum!('U', -1, zeros(0, 0), 2)
end
