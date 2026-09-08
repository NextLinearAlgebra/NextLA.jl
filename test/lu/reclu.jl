# Recursive LU, splitting into a 2x2 block scheme and driving the updates
# through unified_rectrxm! and recgemm!.
#
# Diagonally dominant input for the sweeps. The recursion is unpivoted, leaves
# included, so a general random matrix can fail for mathematical reasons rather
# than a defect. The leaf used to be a pivoted lu! whose pivots were dropped; a
# matrix that has an unpivoted LU without being diagonally dominant is kept
# below so that regression is named.
#
# CPU only for the mixed-precision method: the containers index scalar-wise on
# the host, the limitation recorded for src/mxp/. Float32-only containers, since
# a Float16 block carries a scale factor getindex never applies.

@testset "RECLU [CPU]" begin
    @testset "$T" for T in (Float32, Float64)
        rtol = T === Float32 ? 1e-4 : 1e-10

        @testset "lu_recursive! n=$n block=$bs" for
                n  in [32, 64, 128],
                bs in [16, 32]

            A0 = rand(T, n, n) + T(2n) * I
            A = copy(A0)
            NextLA.lu_recursive!(A, bs)
            @test norm(UnitLowerTriangular(A) * UpperTriangular(A) - A0) / norm(A0) < rtol
        end

        # The rectangular sibling. Square shapes are here so the two can be
        # compared directly; the rest are the shapes lu_recursive! cannot take.
        @testset "lu_recursive_nopiv! $(m)x$(n) block=$bs" for
                (m, n) in [(64, 64), (128, 64), (64, 128), (100, 60), (48, 200)],
                bs in [16, 32]

            k = min(m, n)
            A0 = rand(T, m, n)
            A0[1:k, 1:k] += T(2 * max(m, n)) * I        # needs no pivoting
            A = copy(A0)
            NextLA.lu_recursive_nopiv!(A, bs)

            # L is unit lower trapezoidal, U upper trapezoidal, both packed in A.
            L = m >= n ? [UnitLowerTriangular(A[1:k, 1:k]); A[(k + 1):m, 1:k]] :
                         Matrix(UnitLowerTriangular(A[1:k, 1:k]))
            U = m >= n ? Matrix(UpperTriangular(A[1:k, 1:k])) :
                         [UpperTriangular(A[1:k, 1:k]) A[1:k, (k + 1):n]]
            @test norm(L * U - A0) / norm(A0) < rtol
        end

        # A wide matrix is where the branch's own split (n ÷ 2) walked off the
        # top of the matrix. Kept as its own case so the regression is named.
        @testset "wide matrix does not overrun its rows" begin
            A = rand(T, 48, 200)
            A[1:48, 1:48] += T(400) * I
            @test NextLA.lu_recursive_nopiv!(copy(A), 32) isa AbstractMatrix
        end

        @testset "rejects a block_size below one" begin
            @test_throws ArgumentError NextLA.lu_recursive_nopiv!(rand(T, 8, 8), 0)
            @test_throws ArgumentError NextLA.lu_recursive!(rand(T, 8, 8), 0)
        end

        @testset "lu_recursive! rejects a non-square matrix" begin
            @test_throws DimensionMismatch NextLA.lu_recursive!(rand(T, 8, 6), 4)
        end

        # Leading minors all nonzero, so the unpivoted LU exists, but the first
        # column's largest entry is off the diagonal: a pivoted leaf swaps rows.
        @testset "the leaf does not pivot, block=$bs" for bs in (1, 2, 4)
            A0 = T[1 2 3 4; 4 1 2 3; 3 4 1 2; 2 3 4 1]
            A = copy(A0)
            @test NextLA.lu_recursive!(A, bs) === A
            @test norm(UnitLowerTriangular(A) * UpperTriangular(A) - A0) / norm(A0) < rtol
            B = copy(A0)
            NextLA.lu_recursive_nopiv!(B, bs)
            @test norm(UnitLowerTriangular(B) * UpperTriangular(B) - A0) / norm(A0) < rtol
        end

        @testset "a zero pivot throws ZeroPivotException" begin
            @test_throws ZeroPivotException NextLA.lu_recursive!(T[0 1; 1 0], 2)
        end
    end
end

# The leaf on the GPU backends that serve it: cuSOLVER through lu! on CUDA, and
# rocSOLVER through ext/amdgpu/lu.jl on AMDGPU, where lu! never reaches it. The
# recursion hands the leaf views, which the AMDGPU method factors in a copy.
@testset "RECLU leaf on the GPU [$backend_name]" for (backend_name, ArrayType, synchronize) in
        filter(b -> b[1] in ("CUDA", "AMDGPU"), available_backends())
    @testset "$T n=$n block=$bs" for T in (Float32, Float64), n in [64, 128], bs in [16, 32]
        rtol = T === Float32 ? 1e-4 : 1e-10
        A0 = rand(T, n, n) + T(2n) * I
        A = ArrayType(A0)
        NextLA.lu_recursive!(A, bs)
        synchronize(A)
        F = Array(A)
        @test norm(UnitLowerTriangular(F) * UpperTriangular(F) - A0) / norm(A0) < rtol
    end

    # getrf! used to return quietly here and let Inf/NaN spread; the leaf raises.
    @testset "a zero pivot raises" begin
        A = ArrayType(zeros(Float32, 16, 16))
        @test_throws ZeroPivotException NextLA.lu_recursive!(A, 32)
    end

    @testset "the leaf does not pivot" begin
        A0 = [1.0 2 3 4; 4 1 2 3; 3 4 1 2; 2 3 4 1]
        A = ArrayType(copy(A0))
        NextLA.lu_recursive!(A, 4)
        synchronize(A)
        F = Array(A)
        @test norm(UnitLowerTriangular(F) * UpperTriangular(F) - A0) / norm(A0) < 1e-12
    end
end
