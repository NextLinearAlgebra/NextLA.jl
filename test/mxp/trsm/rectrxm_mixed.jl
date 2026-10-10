using NextLA.MixedPrecision
# Triangular solve and multiply over the mixed-precision containers, checked
# against BLAS on the equivalent dense matrix. CPU only: the containers index
# scalar-wise on the host, the same limitation recorded for src/mxp/.

# BLAS on the container's own values, in Float64, so a comparison measures the
# solve and not how the container rounded A.
function _mxp_reference(func, side, uplo, trans, diag, A, B)
    ref = Float64.(B)
    Ad = Float64.(A)
    if func == 'S'
        LinearAlgebra.BLAS.trsm!(side, uplo, trans, diag, 1.0, Ad, ref)
    else
        LinearAlgebra.BLAS.trmm!(side, uplo, trans, diag, 1.0, Ad, ref)
    end
    return ref
end

_mxp_relerr(X, ref) = norm(Float64.(X) - ref) / norm(ref)

@testset "RECTRXM mixed [CPU]" begin
    Random.seed!(2051)

    rtol = 1e-4   # Float32 containers, accumulated over a recursion

    @testset "func=$func side=$side uplo=$uplo n=$n" for
            func in ['S', 'M'],
            side in ['L', 'R'],
            uplo in ['L', 'U'],
            n    in [16, 37, 64]

        m = 8
        A = uplo == 'L' ? tril(rand(Float32, n, n)) : triu(rand(Float32, n, n))
        A += Diagonal(10f0 * ones(Float32, n))
        B = side == 'L' ? rand(Float32, n, m) : rand(Float32, m, n)

        Amp = TriMixedPrec(A, uplo; precisions = DataType[Float32, Float32])
        Bm = copy(B)
        @test MixedPrecision.unified_rec_mixed(func, side, uplo, 'N', Amp, Bm) === Bm
        @test _mxp_relerr(Bm, _mxp_reference(func, side, uplo, 'N', 'N', A, B)) < rtol
    end

    @testset "unit diagonal, func=$func" for func in ['S', 'M']
        n, m = 32, 4
        A = tril(rand(Float32, n, n)) / n
        B = rand(Float32, n, m)
        Amp = TriMixedPrec(A, 'L'; precisions = DataType[Float32, Float32])
        Bm = copy(B)
        MixedPrecision.unified_rec_mixed(func, 'L', 'L', 'U', Amp, Bm)
        @test _mxp_relerr(Bm, _mxp_reference(func, 'L', 'L', 'N', 'U', A, B)) < rtol
    end

    @testset "transposed container, func=$func side=$side" for func in ['S', 'M'], side in ['L', 'R']
        n, m = 32, 4
        A = tril(rand(Float32, n, n)) + Diagonal(10f0 * ones(Float32, n))
        B = side == 'L' ? rand(Float32, n, m) : rand(Float32, m, n)
        Amp = TriMixedPrec(A, 'L'; precisions = DataType[Float32, Float32])
        Bm = copy(B)
        MixedPrecision.unified_rec_mixed(func, side, 'U', 'N', transpose(Amp), Bm)
        @test _mxp_relerr(Bm, _mxp_reference(func, side, 'L', 'T', 'N', A, B)) < rtol
    end

    @testset "FullMixedPrec, uplo=$uplo, transposed=$t" for uplo in ['L', 'U'], t in (false, true)
        n, m = 32, 4
        A = rand(Float32, n, n) + Diagonal(10f0 * ones(Float32, n))
        B = rand(Float32, n, m)
        Amp = FullMixedPrec(A; precisions = DataType[Float32, Float32])
        Bm = copy(B)
        if t
            MixedPrecision.unified_rec_mixed('S', 'L', uplo, 'N', transpose(Amp), Bm)
            ref = _mxp_reference('S', 'L', uplo == 'L' ? 'U' : 'L', 'T', 'N', A, B)
        else
            MixedPrecision.unified_rec_mixed('S', 'L', uplo, 'N', Amp, Bm)
            ref = _mxp_reference('S', 'L', uplo, 'N', 'N', A, B)
        end
        @test _mxp_relerr(Bm, ref) < rtol
    end

    @testset "Float16 blocks, func=$func" for func in ['S', 'M']
        n, m = 32, 4
        A = tril(rand(Float32, n, n)) + Diagonal(10f0 * ones(Float32, n))
        B = rand(Float32, n, m)
        Amp = TriMixedPrec(A, 'L'; precisions = DataType[Float16, Float16])
        ref = _mxp_reference(func, 'L', 'L', 'N', 'N', reconstruct_matrix(Amp), B)
        Bm = copy(B)
        MixedPrecision.unified_rec_mixed(func, 'L', 'L', 'N', Amp, Bm)
        @test eltype(Bm) === Float32
        @test _mxp_relerr(Bm, ref) < 1e-2
    end

    @testset "scaled off-diagonal blocks, func=$func" for func in ['S', 'M']
        n, m = 16, 4
        A = tril(rand(Float32, n, n)) * 1f5 + Diagonal(1f6 * ones(Float32, n))
        B = rand(Float32, n, m)
        Amp = TriMixedPrec(A, 'L'; precisions = DataType[Float16, Float32])
        @test Amp.OffDiag_scale != 1
        ref = _mxp_reference(func, 'L', 'L', 'N', 'N', reconstruct_matrix(Amp), B)
        Bm = copy(B)
        MixedPrecision.unified_rec_mixed(func, 'L', 'L', 'N', Amp, Bm)
        @test _mxp_relerr(Bm, ref) < 1e-2
    end

    @testset "a scaled leaf, func=$func" for func in ['S', 'M']
        n, m = 16, 4
        A = tril(rand(n, n)) * 1.0e5 + 2.0e5I
        Amp = TriMixedPrec(A, 'L'; precisions = DataType[Float16])
        @test Amp.Base_scale != 1
        B = rand(Float32, n, m)
        Bm = copy(B)
        MixedPrecision.unified_rec_mixed(func, 'L', 'L', 'N', Amp, Bm)
        @test _mxp_relerr(Bm, _mxp_reference(func, 'L', 'L', 'N', 'N', A, B)) < 1e-2
    end

    @testset "a multiply that overflows gives Inf, not floatmax" begin
        Amp = TriMixedPrec(Matrix(1.0e5I, 4, 4), 'L'; precisions = DataType[Float16])
        @test Amp.Base_scale != 1
        B = fill(1f34, 4, 2)
        MixedPrecision.unified_rec_mixed('M', 'L', 'L', 'N', Amp, B)
        @test all(isinf, B)
    end

    # The off-diagonal product rounds B's part to the block's
    # Float16 through a scaled copy; a plain conversion made values past
    # floatmax(Float16) Inf.
    @testset "an off-diagonal product keeps B values past floatmax(Float16)" begin
        n = 16
        A = tril(rand(Float32, n, n)) + n * I
        Amp = TriMixedPrec(A, 'L'; precisions = DataType[Float16, Float32])
        B = 1.0f6 .* rand(Float32, n, 3)
        ref = Float64.(A) \ Float64.(B)
        Bm = copy(B)
        MixedPrecision.unified_rec_mixed('S', 'L', 'L', 'N', Amp, Bm)
        @test all(isfinite, Bm)
        @test _mxp_relerr(Bm, ref) < 1e-2
    end

    @testset "the form without diag takes 'N'" begin
        n = 16
        A = tril(rand(Float32, n, n)) + Diagonal(10f0 * ones(Float32, n))
        B = rand(Float32, n, 3)
        Amp = TriMixedPrec(A, 'L'; precisions = DataType[Float32, Float32])
        B1, B2 = copy(B), copy(B)
        MixedPrecision.unified_rec_mixed('S', 'L', 'L', Amp, B1)
        MixedPrecision.unified_rec_mixed('S', 'L', 'L', 'N', Amp, B2)
        @test B1 == B2
    end

    # The dense driver's TRMM base case is only valid up to TILE_DIM = 16;
    # a larger threshold used to give silently wrong answers, so unified_rec
    # clamps it. Guard the clamp from the mixed path's side.
    @testset "multiply respects the dense TRMM tile limit" begin
        n, m = 64, 8
        A = tril(rand(Float32, n, n)) + Diagonal(10f0 * ones(Float32, n))
        B = rand(Float32, n, m)

        Amp = TriMixedPrec(A, 'L'; precisions = DataType[Float32, Float32])
        # a deliberately oversized threshold must not reach src/trsm/trmm.jl's kernels
        Bm = copy(B)
        MixedPrecision.unified_rec_mixed('M', 'L', 'L', 'N', Amp, Bm, 256)
        @test _mxp_relerr(Bm, _mxp_reference('M', 'L', 'L', 'N', 'N', A, B)) < rtol
    end

    @testset "quantize/dequantize" begin
        A = rand(Float32, 32, 32) .* 1000
        q, s = MixedPrecision.quantize(A)
        @test eltype(q) === Float16
        @test s isa Float32
        back = MixedPrecision.dequantize(q, s, Float32)
        @test norm(back - A) / norm(A) < 1e-2   # Float16 storage

        # a matrix past the Float16 range must come back scaled, not saturated
        big = fill(1f5, 8, 8)
        qb, sb = MixedPrecision.quantize(big)
        @test sb > 1f0
        @test !any(isinf, Float32.(qb))
        @test norm(MixedPrecision.dequantize(qb, sb, Float32) - big) / norm(big) < 1e-2

        z, sz = MixedPrecision.quantize(zeros(Float32, 4, 4))
        @test sz == 1f0
        @test all(iszero, z)

        @testset "any storage and scale type" begin
            q2, s2 = MixedPrecision.quantize(rand(Float64, 4, 4), Float32, Float64)
            @test eltype(q2) === Float32
            @test s2 isa Float64
        end

        @testset "an empty matrix" begin
            qe, se = MixedPrecision.quantize(zeros(Float64, 0, 3), Float16, Float64)
            @test size(qe) == (0, 3) && eltype(qe) === Float16
            @test se === 1.0
        end

        @testset "a non-finite matrix throws" begin
            @test_throws ArgumentError MixedPrecision.quantize([1.0 NaN])
        end

        @testset "dequantize converts before multiplying" begin
            q3 = Float16[3]
            @test MixedPrecision.dequantize(reshape(q3, 1, 1), 1.1f0, Float64)[1] == 3.0 * Float64(1.1f0)
            @test (@inferred MixedPrecision.dequantize(reshape(q3, 1, 1), 1.1f0, Float64)) isa Matrix{Float64}
        end
    end
end
