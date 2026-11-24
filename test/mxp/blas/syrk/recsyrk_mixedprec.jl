# recsyrk! over a SymmMixedPrec container, src/mxp/blas/syrk/recsyrk_mixedprec.jl.
# Checked on the stored triangle against Float64 BLAS, with the bound of
# test/setup/mxp.jl for the narrowest precision.

const MXP_SYRK_PRECISIONS = (DataType[Float32], DataType[Float32, Float32], DataType[Float16, Float32],
                             DataType[NextLA.BFloat16, Float32], DataType[Float16, Float32, Float64],
                             DataType[NextLA.BFloat16, Float16, Float32, Float64])

_mxp_syrk_err(C, ref, uplo) = mxp_relerr(mxp_tri(uplo, Float64.(Array(reconstruct_matrix(C)))), mxp_tri(uplo, ref))

@testset "RECSYRK mixed precision [CPU]" begin
    Random.seed!(2055)

    @testset "precisions=$P, uplo=$uplo, n=$n, k=$k" for P in MXP_SYRK_PRECISIONS, uplo in ('L', 'U'),
            n in (13, 32, 45, 64), k in (1, 16)
        T = P[end]
        A, C0 = mxp_rand(T, n, k), mxp_spd(T, n)
        C = SymmMixedPrec(copy(C0), uplo; precisions=P)
        @test recsyrk!(T(1.5), A, T(0.5), C) === C
        ref = 1.5 * Float64.(A) * transpose(Float64.(A)) + 0.5 * Float64.(C0)
        @test _mxp_syrk_err(C, ref, uplo) <= mxp_tol(mxp_narrowest(P), k + 1)
        R = reconstruct_matrix(C)
        @test R == transpose(R)
    end

    @testset "A of another type than the container: $TA into $P" for (TA, P) in
            ((Float64, DataType[Float16, Float32]), (Float32, DataType[Float32, Float64]))
        n, k = 32, 8
        A, C0 = mxp_rand(TA, n, k), mxp_spd(Float64, n)
        C = SymmMixedPrec(copy(C0), 'L'; precisions=P)
        recsyrk!(1.0, A, 1.0, C)
        ref = Float64.(A) * transpose(Float64.(A)) + C0
        @test _mxp_syrk_err(C, ref, 'L') <= mxp_tol(mxp_narrowest(vcat(P, TA)), k + 1)
    end

    @testset "a scaled block takes alpha divided by its scale, uplo=$uplo" for uplo in ('L', 'U')
        n = 8
        C0 = ones(n, n)
        C0[5:8, 1:4] .= 1.0e5
        C0[1:4, 5:8] .= 1.0e5
        C = SymmMixedPrec(copy(C0), uplo; precisions=DataType[Float16, Float64])
        @test C.OffDiag_scale != 1
        A = mxp_rand(Float64, n, 4)
        recsyrk!(-100.0, A, 1.0, C)
        @test _mxp_syrk_err(C, -100 * A * transpose(A) + C0, uplo) <= mxp_tol(Float16, 5)
    end

    # An update past floatmax of the Float16 off-diagonal block raises its scale.
    @testset "an update that overflows a Float16 block raises its scale, uplo=$uplo" for uplo in ('L', 'U')
        C = SymmMixedPrec(zeros(4, 4), uplo; precisions=DataType[Float16, Float32])
        A = fill(1000.0, 4, 2)
        recsyrk!(1.0, A, 0.0, C)
        @test C.OffDiag_scale > 1
        @test _mxp_syrk_err(C, A * A', uplo) <= mxp_tol(Float16, 2)
    end

    # The threaded path runs only above RECSYRK_PARALLEL_THRESHOLD (4096) rows in
    # A11, too large for a test, so it is reached through the internal function.
    # The two halves write disjoint blocks: same arithmetic, same bits.
    @testset "parallel and sequential give the same result, uplo=$uplo" for uplo in ('L', 'U')
        n = 32
        A, C0 = mxp_rand(Float64, n, 8), mxp_spd(Float64, n)
        seq = SymmMixedPrec(copy(C0), uplo; precisions=DataType[Float16, Float32, Float64])
        par = SymmMixedPrec(copy(C0), uplo; precisions=DataType[Float16, Float32, Float64])
        MixedPrecision._recsyrk_mixed!(1.0, A, 1.0, seq; parallel=false)
        MixedPrecision._recsyrk_mixed!(1.0, A, 1.0, par; parallel=true)
        @test reconstruct_matrix(seq) == reconstruct_matrix(par)
    end

    @testset "A with the wrong number of rows throws" begin
        C = SymmMixedPrec(rand(8, 8), 'L'; precisions=DataType[Float64])
        @test_throws DimensionMismatch recsyrk!(1.0, rand(7, 3), 1.0, C)
    end
end

# On a GPU: the update runs on the device; the result is read through a host copy
# of the container and checked on the CPU.
@testset "RECSYRK mixed precision [$name]" for (name, AT, _) in filter(b -> b[1] != "CPU", MXP_BACKENDS)
    Random.seed!(2056)
    @testset "precisions=$P, uplo=$uplo" for P in (DataType[Float32, Float32], DataType[Float16, Float32, Float64]),
            uplo in ('L', 'U')
        n, k = 45, 16
        T = P[end]
        A, C0 = mxp_rand(T, n, k), mxp_spd(T, n)
        C = SymmMixedPrec(AT(C0), uplo; precisions=P)
        recsyrk!(T(1.5), AT(A), T(0.5), C)
        ref = 1.5 * Float64.(A) * transpose(Float64.(A)) + 0.5 * Float64.(C0)
        @test _mxp_syrk_err(mxp_host(C), ref, uplo) <= mxp_tol(mxp_narrowest(P), k + 1)
    end
end
