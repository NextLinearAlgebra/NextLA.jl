# recgemm! into a FullMixedPrec destination, src/mxp/blas/gemm/recgemm_mixedprec.jl.
# Accuracy is checked against Float64 BLAS with the bound of test/setup/mxp.jl for
# the narrowest precision involved (the container's or the operands').

const MXP_GEMM_PRECISIONS = (DataType[Float32], DataType[Float32, Float32], DataType[Float16, Float32],
                             DataType[NextLA.BFloat16, Float32], DataType[Float16, Float32, Float64],
                             DataType[NextLA.BFloat16, Float16, Float32, Float64])

@testset "RECGEMM mixed precision [$name]" for (name, AT, _) in MXP_BACKENDS
    Random.seed!(2049)

    @testset "precisions=$P, n=$n, k=$k" for P in MXP_GEMM_PRECISIONS, n in (5, 13, 16), k in (1, 7, 16)
        T = P[end]
        A, B, C0 = mxp_rand(T, n, k), mxp_rand(T, k, n), mxp_rand(T, n, n)
        alpha, beta = T(1.5), T(0.5)
        C = FullMixedPrec(mxp_on(AT, copy(C0)); precisions=P)
        @test recgemm!(alpha, mxp_on(AT, A), mxp_on(AT, B), beta, C) === C
        ref = 1.5 * Float64.(A) * Float64.(B) + 0.5 * Float64.(C0)
        @test mxp_relerr(reconstruct_matrix(C), ref) <= mxp_tol(mxp_narrowest(P), k + 1)
    end

    @testset "operands of another type than the container: A, B $TA into $P" for (TA, P) in
            ((Float64, DataType[Float16, Float32]), (Float32, DataType[Float32, Float64]))
        n, k = 16, 8
        A, B, C0 = mxp_rand(TA, n, k), mxp_rand(TA, k, n), mxp_rand(Float64, n, n)
        C = FullMixedPrec(mxp_on(AT, copy(C0)); precisions=P)
        recgemm!(1.0, mxp_on(AT, A), mxp_on(AT, B), 1.0, C)
        ref = Float64.(A) * Float64.(B) + C0
        @test mxp_relerr(reconstruct_matrix(C), ref) <= mxp_tol(mxp_narrowest(vcat(P, TA)), k + 1)
    end

    @testset "beta = 0 overwrites C" begin
        n = 8
        A, B = mxp_rand(Float64, n, n), mxp_rand(Float64, n, n)
        C = FullMixedPrec(mxp_on(AT, mxp_rand(Float64, n, n)); precisions=DataType[Float32, Float64])
        recgemm!(1.0, mxp_on(AT, A), mxp_on(AT, B), 0.0, C)
        @test mxp_relerr(reconstruct_matrix(C), A * B) <= mxp_tol(Float32, n)
    end

    @testset "a scaled block takes alpha divided by its scale" begin
        n = 8
        C0 = ones(n, n)
        C0[5:8, 1:4] .= 1.0e5
        C = FullMixedPrec(mxp_on(AT, copy(C0)); precisions=DataType[Float16, Float64])
        @test C.A21_scale != 1
        A, B = mxp_rand(Float64, n, n), mxp_rand(Float64, n, n)
        recgemm!(-100.0, mxp_on(AT, A), mxp_on(AT, B), 1.0, C)
        @test mxp_relerr(reconstruct_matrix(C), -100 * A * B + C0) <= mxp_tol(Float16, n + 1)
    end

    # An update past floatmax of a Float16 off-diagonal block raises the block's
    # scale, so the result stays finite where a fixed scale left Inf.
    @testset "an update that overflows a Float16 block raises its scale" begin
        C = FullMixedPrec(mxp_on(AT, zeros(4, 4)); precisions=DataType[Float16, Float32])
        A, B = fill(1000.0, 4, 4), fill(1000.0, 4, 4)
        recgemm!(1.0, mxp_on(AT, A), mxp_on(AT, B), 0.0, C)
        @test C.A12_scale > 1 && C.A21_scale > 1
        R = reconstruct_matrix(C)
        @test all(isfinite, Array(R))
        @test mxp_relerr(R, A * B) <= mxp_tol(Float16, 4)
    end

    # The four quadrant updates touch disjoint blocks, so the threaded run does the
    # same arithmetic as the sequential one and gives the same bits.
    @testset "parallel and sequential give the same result" begin
        n = 16
        A, B, C0 = mxp_rand(Float32, n, n), mxp_rand(Float32, n, n), mxp_rand(Float32, n, n)
        seq = FullMixedPrec(mxp_on(AT, copy(C0)); precisions=DataType[Float16, Float32])
        par = FullMixedPrec(mxp_on(AT, copy(C0)); precisions=DataType[Float16, Float32])
        recgemm!(1f0, mxp_on(AT, A), mxp_on(AT, B), 1f0, seq; parallel=false)
        recgemm!(1f0, mxp_on(AT, A), mxp_on(AT, B), 1f0, par; parallel=true)
        @test Array(reconstruct_matrix(seq)) == Array(reconstruct_matrix(par))
    end

    @testset "mismatched shapes throw DimensionMismatch" begin
        C = FullMixedPrec(mxp_on(AT, rand(8, 8)); precisions=DataType[Float64])
        @test_throws DimensionMismatch recgemm!(1.0, mxp_on(AT, rand(7, 4)), mxp_on(AT, rand(4, 8)), 1.0, C)
        @test_throws DimensionMismatch recgemm!(1.0, mxp_on(AT, rand(8, 4)), mxp_on(AT, rand(4, 7)), 1.0, C)
        @test_throws DimensionMismatch recgemm!(1.0, mxp_on(AT, rand(8, 4)), mxp_on(AT, rand(5, 8)), 1.0, C)
    end
end

# Above RECGEMM_PARALLEL_THRESHOLD (512) the default is parallel. CPU only: the
# size is for the default, not for the device.
@testset "RECGEMM mixed precision, default parallel above the threshold [CPU]" begin
    Random.seed!(2050)
    n, k = 520, 16
    A, B, C0 = mxp_rand(Float32, n, k), mxp_rand(Float32, k, n), mxp_rand(Float32, n, n)
    C = FullMixedPrec(copy(C0); precisions=DataType[Float16, Float32])
    recgemm!(1f0, A, B, 1f0, C)
    @test mxp_relerr(reconstruct_matrix(C), Float64.(A) * Float64.(B) + C0) <= mxp_tol(Float16, k + 1)
end
