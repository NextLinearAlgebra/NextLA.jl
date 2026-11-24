# FullMixedPrec, src/mxp/core/fullmixedprec.jl.
# Helpers and error bounds: test/setup/mxp.jl.

@testset "FullMixedPrec [CPU]" begin
    Random.seed!(2047)

    @testset "$T, n=$n" for T in (Float32, Float64), n in (1, 4, 5, 8, 13, 16)
        A = mxp_rand(T, n, n)

        @testset "one precision round-trips exactly" begin
            M = FullMixedPrec(A; precisions=DataType[T])
            @test size(M) == (n, n)
            @test reconstruct_matrix(M) == A
        end

        @testset "precisions=$P" for P in (DataType[Float32, T], DataType[Float16, Float32, T],
                                           DataType[NextLA.BFloat16, Float16, Float32, T])
            M = FullMixedPrec(A; precisions=P)
            R = reconstruct_matrix(M)
            @test eltype(R) === T
            @test mxp_storage_ok(R, A, mxp_eltype_map(M))
            @test all(M[i, j] == R[i, j] for i in 1:n, j in 1:n)
            @test M[1, 1] isa T
        end
    end

    # An input matrix of a 16-bit type (the leaves are Float32): every element is
    # stored within one rounding of its block's type, and exactly where the block
    # type is the input type.
    @testset "$S input, n=$n, precisions=$P" for S in (Float16, NextLA.BFloat16), n in (5, 13),
            P in (DataType[Float16, Float32], DataType[NextLA.BFloat16, Float32])
        A = S.(mxp_rand(Float32, n, n))
        M = FullMixedPrec(A; precisions=P)
        R = reconstruct_matrix(M)
        @test eltype(R) === Float32
        @test mxp_storage_ok(R, A, mxp_eltype_map(M))
        P[1] === S && @test R == Float32.(A)
    end

    # A view as input (the routines pass views of larger matrices): the same
    # container as from a copy of the view.
    @testset "a view as input, n=$n, precisions=$P" for n in (5, 13),
            P in (DataType[Float32, Float64], DataType[Float16, Float32, Float64])
        B = mxp_rand(Float64, n + 3, n + 3)
        V = view(B, 2:n+1, 3:n+2)
        M = FullMixedPrec(V; precisions=P)
        R = reconstruct_matrix(M)
        @test R == reconstruct_matrix(FullMixedPrec(copy(V); precisions=P))
        @test mxp_storage_ok(R, copy(V), mxp_eltype_map(M))
    end

    @testset "each level stores its off-diagonal blocks in its precision" begin
        M = FullMixedPrec(mxp_rand(Float64, 16, 16);
                          precisions=DataType[NextLA.BFloat16, Float16, Float32, Float64])
        @test eltype(M.A21) === eltype(M.A12) === NextLA.BFloat16
        @test eltype(M.A11.A21) === Float16
        @test eltype(M.A11.A11.A21) === Float32
        @test eltype(M.A11.A11.A11.Base) === Float64
        @test size(M.A11.A11.A11.Base) == (2, 2)
    end

    @testset "every scale is Float32" begin
        M = FullMixedPrec(rand(8, 8); precisions=DataType[Float32, Float64])
        @test M.A21_scale isa Float32 && M.A12_scale isa Float32
        @test M.A11.Base_scale isa Float32
    end

    @testset "a scaled off-diagonal block reads back unscaled" begin
        A = fill(1.0, 4, 4)
        A[3:4, 1:2] .= 1.0e5
        M = FullMixedPrec(A; precisions=DataType[Float16, Float64])
        @test M.A21_scale != 1 && M.A12_scale == 1
        @test reconstruct_matrix(M) ≈ A rtol=2mxp_u(Float16)
        @test M[3, 1] ≈ 1.0e5 rtol=2mxp_u(Float16)
    end

    # A leaf is stored in T_Base, so values past floatmax(T_Base) are kept as
    # stored values times a scale; read back in T_Base they are Inf.
    @testset "a leaf past floatmax of its type is stored with a scale" begin
        A = fill(1.0e39, 4, 4) + 1.0e40I
        M = FullMixedPrec(A; precisions=DataType[Float32])
        @test M.Base_scale != 1
        @test Float64.(M.Base) .* M.Base_scale ≈ A rtol=2mxp_u(Float32)
        @test all(isinf, reconstruct_matrix(M))
    end

    @testset "typemin of an integer matrix is scaled, not stored as -Inf" begin
        M = FullMixedPrec([1 0; typemin(Int) 1]; precisions=DataType[Float16, Float64])
        @test M.A21_scale > 1 && all(isfinite, M.A21)
        R = reconstruct_matrix(M)
        @test all(isfinite, R)
        @test R[2, 1] ≈ Float64(typemin(Int)) rtol=2mxp_u(Float16)
    end

    @testset "arguments are checked" begin
        @test_throws DimensionMismatch FullMixedPrec(rand(4, 8); precisions=DataType[Float64])
        @test_throws ArgumentError FullMixedPrec(zeros(0, 0); precisions=DataType[Float64])
        @test_throws ArgumentError FullMixedPrec(rand(4, 4); precisions=DataType[])
        @test_throws ArgumentError FullMixedPrec(rand(4, 4); precisions=DataType[Int])
        @test_throws ArgumentError FullMixedPrec([1.0 NaN; 0.0 1.0]; precisions=DataType[Float64])
        @test_throws ArgumentError FullMixedPrec([1.0 Inf; 0.0 1.0]; precisions=DataType[Float64])
        @test_throws ArgumentError FullMixedPrec(fill(1.0e300, 4, 4); precisions=DataType[Float16, Float32])
    end

    @testset "out-of-bounds indexing throws" begin
        M = FullMixedPrec(rand(8, 8); precisions=DataType[Float32, Float64])
        @test_throws BoundsError M[0, 1]
        @test_throws BoundsError M[1, 9]
    end
end

# On each GPU backend: builds and reconstructs without scalar indexing.
@testset "FullMixedPrec [$name]" for (name, AT, _) in filter(b -> b[1] != "CPU", MXP_BACKENDS)
    Random.seed!(2048)
    n = 13
    P = DataType[Float16, Float32, Float64]

    @testset "FullMixedPrec builds and reconstructs on the device" begin
        A = mxp_rand(Float64, n, n)
        M = FullMixedPrec(AT(A); precisions=P)
        R = reconstruct_matrix(M)
        @test R isa AT
        @test eltype(M.A21) === Float16 && M.A21 isa AT
        @test mxp_storage_ok(Array(R), A, mxp_eltype_map(M))
    end
end
