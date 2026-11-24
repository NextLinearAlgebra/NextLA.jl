# The block helpers of src/mxp/core/blocks.jl: the argument checks every
# container constructor shares, and storing an updated or solved off-diagonal
# block back with its scale.
# Helpers and error bounds: test/setup/mxp.jl.

@testset "block helpers [CPU]" begin
    Random.seed!(2047)

    @testset "leaves narrower than Float32 are refused" begin
        for T in MXP_NARROW
            @test_throws ArgumentError FullMixedPrec(rand(4, 4); precisions=DataType[T])
            @test_throws ArgumentError SymmMixedPrec(rand(4, 4), 'L'; precisions=DataType[Float16, T])
            @test_throws ArgumentError TriMixedPrec(rand(4, 4), 'L'; precisions=DataType[Float16, T])
            @test_throws ArgumentError TiledTriMixedPrec(rand(4, 4), 'L'; precisions=DataType[Float16, T], threshold=2)
        end
        @test FullMixedPrec(rand(4, 4); precisions=DataType[Float16, Float32]) isa FullMixedPrec
    end

    # The checks are shared (_check_mixedprec_args, and _quantize_mixedprec_block
    # for the values), so each one is checked on every constructor.
    @testset "arguments are checked, $(nameof(C))" for C in (FullMixedPrec, SymmMixedPrec, TriMixedPrec, TiledTriMixedPrec)
        make(A, P) = C === FullMixedPrec ? C(A; precisions=P) :
                     C === TiledTriMixedPrec ? C(A, 'L'; precisions=P, threshold=2) : C(A, 'L'; precisions=P)
        @test_throws ArgumentError make(zeros(0, 0), DataType[Float64])
        @test_throws DimensionMismatch make(rand(4, 8), DataType[Float64])
        @test_throws ArgumentError make(rand(4, 4), DataType[])
        @test_throws ArgumentError make(rand(4, 4), DataType[Int])
        @test_throws ArgumentError make(rand(4, 4), DataType[Float16, ComplexF64])
        @test_throws ArgumentError make([1.0 0.0; NaN 1.0], DataType[Float64])
        @test_throws ArgumentError make([1.0 0.0; Inf 1.0], DataType[Float64])
        @test_throws ArgumentError make(fill(1.0e300, 4, 4), DataType[Float16, Float32])
    end

    @testset "a solve that outgrows a Float16 block raises its scale" begin
        blk = fill(Float16(100), 2, 2)
        s = MixedPrecision._solve_mixedprec_offdiag_block!(W -> (W .*= 1000), blk, 1f0, Float32)
        @test s > 1 && all(isfinite, blk)
        @test Float32.(blk) .* s ≈ fill(1f5, 2, 2) rtol=2mxp_u(Float16)
    end

    @testset "a block of the base type is solved in place, scale unchanged" begin
        same = fill(1f0, 2, 2)
        @test MixedPrecision._solve_mixedprec_offdiag_block!(W -> (W .*= 2), same, 1f0, Float32) == 1f0
        @test same == fill(2f0, 2, 2)
    end

    @testset "a solve past what the scale can carry throws" begin
        blk = fill(1f0, 2, 2)
        @test_throws ArgumentError MixedPrecision._solve_mixedprec_offdiag_block!(W -> (W .*= 1e80), blk, 1f0, Float64)
    end

    @testset "an update of a Float32 block runs in place" begin
        blk = fill(1f0, 2, 2)
        @test MixedPrecision._update_mixedprec_offdiag_block!(P -> (P .+= 1), blk, 1f0) == 1f0
        @test blk == fill(2f0, 2, 2)
    end

    @testset "an update past floatmax of a Float16 block raises its scale" begin
        blk = fill(Float16(1), 2, 2)
        s = MixedPrecision._update_mixedprec_offdiag_block!(P -> (P .= 1f6), blk, 1f0)
        @test s > 1 && all(isfinite, blk)
        @test Float32.(blk) .* s ≈ fill(1f6, 2, 2) rtol=2mxp_u(Float16)
    end

    # floatmax(BFloat16) is just below floatmax(Float32), so a BFloat16 block is
    # updated on a Float32 copy and solved on a copy of the leaf type, as Float16.
    @testset "BFloat16 blocks raise their scale" begin
        BF = NextLA.BFloat16
        blk = fill(BF(1), 2, 2)
        s = MixedPrecision._update_mixedprec_offdiag_block!(P -> (P .= floatmax(Float32)), blk, 1f0)
        @test s > 1 && all(isfinite, blk)
        @test Float64.(blk) .* s ≈ fill(Float64(floatmax(Float32)), 2, 2) rtol=2mxp_u(BF)
        blk = fill(BF(1), 2, 2)
        s = MixedPrecision._solve_mixedprec_offdiag_block!(W -> (W .*= 1e39), blk, 1f0, Float64)
        @test s > 1 && all(isfinite, blk)
        @test Float64.(blk) .* s ≈ fill(1e39, 2, 2) rtol=2mxp_u(BF)
    end

    # A result with NaN or Inf is stored as it is and the scale stays (blocks.jl:30),
    # so a breakdown in a routine shows in its result; the constructors, in
    # contrast, refuse non-finite input.
    @testset "a non-finite result is stored as it is, $T block, $x" for T in (Float16, Float32), x in (NaN32, Inf32, -Inf32)
        blk = fill(T(1), 2, 2)
        @test MixedPrecision._update_mixedprec_offdiag_block!(P -> (P[1, 1] = x), blk, 1f0) == 1f0
        @test isequal(blk[1, 1], T(x)) && blk[2, 2] == 1
        blk = fill(T(1), 2, 2)
        @test MixedPrecision._solve_mixedprec_offdiag_block!(W -> (W[1, 1] = x), blk, 2f0, Float64) == 2f0
        @test isequal(blk[1, 1], T(x)) && blk[2, 2] == 1
    end
end

# On each GPU backend: the same helpers on device blocks (routines call them on
# the device inside recgemm!, recsyrk!, lu_recursive! and potrf_recursive!).
@testset "block helpers [$name]" for (name, AT, _) in filter(b -> b[1] != "CPU", MXP_BACKENDS)
    @testset "a solve that outgrows a Float16 block raises its scale" begin
        blk = AT(fill(Float16(100), 2, 2))
        s = MixedPrecision._solve_mixedprec_offdiag_block!(W -> (W .*= 1000), blk, 1f0, Float32)
        @test blk isa AT && s > 1
        @test Float32.(Array(blk)) .* s ≈ fill(1f5, 2, 2) rtol=2mxp_u(Float16)
    end

    @testset "a block of the base type is solved in place, scale unchanged" begin
        same = AT(fill(1f0, 2, 2))
        @test MixedPrecision._solve_mixedprec_offdiag_block!(W -> (W .*= 2), same, 1f0, Float32) == 1f0
        @test Array(same) == fill(2f0, 2, 2)
    end

    @testset "a solve past what the scale can carry throws" begin
        blk = AT(fill(1f0, 2, 2))
        @test_throws ArgumentError MixedPrecision._solve_mixedprec_offdiag_block!(W -> (W .*= 1e80), blk, 1f0, Float64)
    end

    @testset "an update past floatmax of a $T block raises its scale" for T in MXP_NARROW
        blk = AT(fill(T(1), 2, 2))
        big = T === Float16 ? 1f6 : floatmax(Float32)
        s = MixedPrecision._update_mixedprec_offdiag_block!(P -> (P .= big), blk, 1f0)
        @test blk isa AT && s > 1
        @test Float64.(Array(blk)) .* s ≈ fill(Float64(big), 2, 2) rtol=2mxp_u(T)
    end

    @testset "a non-finite result is stored as it is, $T block" for T in (Float16, Float32)
        blk = AT(fill(T(1), 2, 2))
        @test MixedPrecision._update_mixedprec_offdiag_block!(P -> (P .= NaN32), blk, 1f0) == 1f0
        @test all(isnan, Array(blk))
    end
end
