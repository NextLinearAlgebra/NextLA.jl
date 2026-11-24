# lu_recursive! over a FullMixedPrec matrix, src/mxp/lapack/lu/reclu_mixedprec.jl.
# Checked by the backward error ‖A - LU‖/‖A‖ (Higham 2002, Thm 9.3) on diagonally
# dominant matrices, which need no pivoting and keep ‖|L||U|‖ ≈ ‖A‖ (§9.5), against
# the bound of test/setup/mxp.jl for the narrowest precision. The entry points of
# unified_rectrxm! it uses are tested in test/mxp/blas/trsm/rectrxm_mixed.jl.

const MXP_LU_PRECISIONS = (DataType[Float32], DataType[Float32, Float32], DataType[Float16, Float32],
                           DataType[NextLA.BFloat16, Float32], DataType[Float16, Float32, Float64],
                           DataType[NextLA.BFloat16, Float16, Float32, Float64])

@testset "RECLU mixed precision [$name]" for (name, AT, _) in MXP_BACKENDS
    Random.seed!(2053)

    @testset "precisions=$P, n=$n, block_size=$bs" for P in MXP_LU_PRECISIONS, n in (13, 32, 50, 100), bs in (8, 64)
        T = P[end]
        A0 = mxp_diagdom(T, n)
        A = FullMixedPrec(mxp_on(AT, copy(A0)); precisions=P)
        @test lu_recursive!(A, bs) === A
        @test mxp_lu_backward_error(reconstruct_matrix(A), Float64.(A0)) <= mxp_tol(mxp_narrowest(P), n)
    end

    @testset "the default block size" begin
        A0 = mxp_diagdom(Float32, 64)
        A = FullMixedPrec(mxp_on(AT, copy(A0)); precisions=DataType[Float16, Float32])
        lu_recursive!(A)
        @test mxp_lu_backward_error(reconstruct_matrix(A), Float64.(A0)) <= mxp_tol(Float16, 64)
    end

    # The two off-diagonal solves of a level write disjoint blocks, so the threaded
    # run does the same arithmetic as the sequential one and gives the same bits.
    @testset "parallel=false and parallel=true give the same factors" begin
        A0 = mxp_diagdom(Float64, 64)
        a = FullMixedPrec(mxp_on(AT, copy(A0)); precisions=DataType[Float16, Float32, Float64])
        b = FullMixedPrec(mxp_on(AT, copy(A0)); precisions=DataType[Float16, Float32, Float64])
        lu_recursive!(a, 8; parallel=true)
        lu_recursive!(b, 8; parallel=false)
        @test Array(reconstruct_matrix(a)) == Array(reconstruct_matrix(b))
    end

    @testset "two large Float32 scales do not overflow the trailing update" begin
        n = 8
        A0 = rand(n, n) .* 2.0e24 + 1.0e27I
        A = FullMixedPrec(mxp_on(AT, copy(A0)); precisions=DataType[Float16, Float64])
        @test isinf(A.A21_scale * A.A12_scale)
        lu_recursive!(A, 4)
        R = reconstruct_matrix(A)
        @test all(isfinite, Array(R))
        @test mxp_lu_backward_error(R, A0) <= mxp_tol(Float16, n)
    end

    @testset "a leaf stored with a scale factor throws" begin
        A = FullMixedPrec(mxp_on(AT, fill(1.0e39, 4, 4) + 1.0e40I); precisions=DataType[Float32])
        @test A.Base_scale != 1
        @test_throws ArgumentError lu_recursive!(A)
    end
end

@testset "RECLU mixed precision, other entry points [CPU]" begin
    Random.seed!(2054)

    @testset "the dense method still takes a dense matrix" begin
        A0 = mxp_diagdom(Float64, 16)
        A = copy(A0)
        lu_recursive!(A, 4)
        @test mxp_lu_backward_error(A, A0) <= mxp_tol(Float64, 16)
    end

    # Above RECLU_PARALLEL_THRESHOLD (512) the default is parallel.
    @testset "default parallel above the threshold" begin
        A0 = mxp_diagdom(Float32, 520)
        A = FullMixedPrec(copy(A0); precisions=DataType[Float16, Float32])
        lu_recursive!(A)
        @test mxp_lu_backward_error(reconstruct_matrix(A), Float64.(A0)) <= mxp_tol(Float16, 520)
    end
end
