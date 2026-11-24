# potrf_recursive! over a SymmMixedPrec container,
# src/mxp/lapack/cholesky/recpotrf_mixedprec.jl. Checked by the backward error
# ‖A - LLᵀ‖/‖A‖ (Higham 2002, Thm 10.3) against the bound of test/setup/mxp.jl for
# the narrowest precision.

const MXP_POTRF_PRECISIONS = (DataType[Float32], DataType[Float32, Float32], DataType[Float16, Float32],
                              DataType[NextLA.BFloat16, Float32], DataType[Float16, Float32, Float64],
                              DataType[NextLA.BFloat16, Float16, Float32, Float64])

_mxp_factor(A) = reconstruct_matrix(TriMixedPrec(A))

@testset "POTRF_RECURSIVE mixed precision [CPU]" begin
    Random.seed!(2057)

    @testset "precisions=$P, uplo=$uplo, n=$n, block_size=$bs" for P in MXP_POTRF_PRECISIONS, uplo in ('L', 'U'),
            n in (13, 32, 40, 64), bs in (8, 4096)
        T = P[end]
        A0 = mxp_spd(T, n)
        A = SymmMixedPrec(copy(A0), uplo; precisions=P)
        @test potrf_recursive!(A, bs) === A
        F = _mxp_factor(A)
        @test uplo == 'L' ? istril(F) : istriu(F)
        @test mxp_chol_backward_error(F, Float64.(A0), uplo) <= mxp_tol(mxp_narrowest(P), n)
    end

    # A leaf no larger than block_size is factored by potrf!, whose info turns into
    # PosDefException. A larger leaf goes through the dense potrf_recursive!.
    @testset "a matrix that is not positive definite throws, uplo=$uplo, block_size=$bs" for uplo in ('L', 'U'),
            bs in (2, 4096)
        A = SymmMixedPrec(-Matrix{Float64}(I, 8, 8), uplo; precisions=DataType[Float32, Float64])
        @test_throws PosDefException potrf_recursive!(A, bs)
    end

    @testset "a leaf stored with a scale factor throws" begin
        A = SymmMixedPrec(1.0e39 * ones(4, 4) + 1.0e40I, 'L'; precisions=DataType[Float32])
        @test A.Base_scale != 1
        @test_throws ArgumentError potrf_recursive!(A)
    end

    # reconstruct_matrix of the factored container mirrors the factor into both
    # triangles; TriMixedPrec(A) reads the factor alone.
    @testset "the factor is read through TriMixedPrec, uplo=$uplo" for uplo in ('L', 'U')
        A = SymmMixedPrec(mxp_spd(Float64, 16), uplo; precisions=DataType[Float32, Float64])
        potrf_recursive!(A)
        F, R = _mxp_factor(A), reconstruct_matrix(A)
        @test mxp_tri(uplo, R) == F
        @test R == transpose(R)
    end
end

# On a GPU: the factorization runs on the device; the factor is read through a
# host copy of the container and checked on the CPU.
@testset "POTRF_RECURSIVE mixed precision [$name]" for (name, AT, _) in filter(b -> b[1] != "CPU", MXP_BACKENDS)
    Random.seed!(2058)
    @testset "precisions=$P, uplo=$uplo" for P in (DataType[Float32, Float32], DataType[Float16, Float32, Float64]),
            uplo in ('L', 'U')
        n = 64
        T = P[end]
        A0 = mxp_spd(T, n)
        A = SymmMixedPrec(AT(A0), uplo; precisions=P)
        potrf_recursive!(A, 8)
        @test mxp_chol_backward_error(_mxp_factor(mxp_host(A)), Float64.(A0), uplo) <= mxp_tol(mxp_narrowest(P), n)
    end
end
