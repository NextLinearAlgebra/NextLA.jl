# Single-matrix Cholesky. potrf! returns (A, info) and overwrites only the
# triangle named by uplo, so the tests assert both: the factor reconstructs the
# input, and the untouched triangle still holds what it did on entry.
#
# LinearAlgebra.cholesky is the oracle. It raises PosDefException on an
# indefinite input where potrf! reports info > 0 instead, so that case is
# checked directly rather than against it.

@testset "POTRF [$backend_name]" for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "$T" for T in TEST_TYPES
        rtol = test_rtol(T)

        @testset "uplo=$uplo n=$n" for uplo in ('L', 'U'), n in (1, 2, 8, 32, 64)
            R = rand(T, n, n)
            A0 = R * R' + T(n) * I          # symmetric/Hermitian positive definite
            Ad = ArrayType(copy(A0))

            F, info = NextLA.potrf!(uplo, Ad)
            synchronize(F)
            @test info == 0

            A = Array(F)
            if uplo == 'L'
                L = LowerTriangular(A)
                @test norm(L * L' - A0) / norm(A0) < rtol
            else
                U = UpperTriangular(A)
                @test norm(U' * U - A0) / norm(A0) < rtol
            end
        end

        # The opposite triangle is neither read nor written by LAPACK, so it
        # must survive untouched. Easy to break by "helpfully" zeroing it.
        @testset "leaves the opposite triangle alone (uplo=$uplo)" for uplo in ('L', 'U')
            n = 16
            R = rand(T, n, n)
            A0 = R * R' + T(n) * I
            Ad = ArrayType(copy(A0))
            NextLA.potrf!(uplo, Ad)
            A = Array(Ad)
            if uplo == 'L'
                @test triu(A, 1) == triu(A0, 1)
            else
                @test tril(A, -1) == tril(A0, -1)
            end
        end
    end

    @testset "defaults to lower" begin
        n = 16
        R = rand(Float64, n, n)
        A0 = R * R' + n * I
        a = ArrayType(copy(A0)); b = ArrayType(copy(A0))
        NextLA.potrf!(a)
        NextLA.potrf!('L', b)
        @test Array(a) == Array(b)
    end

    @testset "argument errors" begin
        A = ArrayType(rand(Float64, 4, 4) + 4I)
        @test_throws ArgumentError NextLA.potrf!('X', A)
        @test_throws DimensionMismatch NextLA.potrf!('L', ArrayType(rand(Float64, 4, 5)))
    end
end

@testset "POTRF non-positive-definite reports info" begin
    # Indefinite: LinearAlgebra.cholesky would raise, potrf! returns info > 0
    # naming the order of the failing leading minor.
    A = Matrix{Float64}([1.0 2.0; 2.0 1.0])
    _, info = NextLA.potrf!('L', copy(A))
    @test info > 0
    @test_throws PosDefException cholesky(A)
end

@testset "POTRF dispatches to the right implementation" begin
    # The generic method is CPU-only on purpose -- no silent host round-trip --
    # so each backend must reach its own wrapper rather than falling through.
    for (name, AT, _) in available_backends()
        A = AT(rand(Float64, 8, 8) + 8I)
        @test occursin(_expected_potrf_file(name), _method_file(NextLA.potrf!, 'L', A))
    end
end

# Float16 has no LAPACK, cuSOLVER or rocSOLVER routine, so potrf! factors a
# Float32 copy and rounds back once. On the CPU the claim is exact: the result
# is the Float32 factor rounded to Float16. On a GPU the vendor factor can
# differ from LAPACK in the last Float32 bit, so there the factor is held to
# reconstructing the input at Float16 accuracy instead. Both keep the opposite
# triangle untouched and return (A, info) in place.
@testset "POTRF Float16 [$backend_name]" for (backend_name, ArrayType, synchronize) in available_backends()
    backend_name in ("CPU", "CUDA", "AMDGPU") || continue
    @testset "uplo=$uplo n=$n" for uplo in ('L', 'U'), n in (1, 8, 32)
        R = rand(Float32, n, n)
        A0 = Float16.(R * R' + Float32(n) * I)
        Ad = ArrayType(copy(A0))

        F, info = NextLA.potrf!(uplo, Ad)
        synchronize(F)
        @test info == 0
        @test F === Ad && eltype(F) == Float16

        A = Array(F)
        if backend_name == "CPU"
            want, _ = NextLA.potrf!(uplo, Float32.(A0))
            @test A == Float16.(want)
        end
        A32, A032 = Float32.(A), Float32.(A0)
        if uplo == 'L'
            L = LowerTriangular(A32)
            @test norm(L * L' - A032) / norm(A032) < 1e-2
            @test triu(A, 1) == triu(A0, 1)
        else
            U = UpperTriangular(A32)
            @test norm(U' * U - A032) / norm(A032) < 1e-2
            @test tril(A, -1) == tril(A0, -1)
        end
    end

    @testset "reports info on an indefinite Float16 matrix" begin
        A = ArrayType(Float16[1 2; 2 1])
        _, info = NextLA.potrf!('L', A)
        @test info > 0
    end
end
