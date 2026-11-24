# SymmMixedPrec, src/mxp/core/symmmixedprec.jl.
# Helpers and error bounds: test/setup/mxp.jl.

@testset "SymmMixedPrec [CPU]" begin
    Random.seed!(2047)

    @testset "$T, uplo=$uplo, n=$n, precisions=$P" for T in (Float32, Float64), uplo in ('L', 'U'),
            n in (1, 7, 8, 13), P in (DataType[Float32, T], DataType[Float16, Float32, T],
                                      DataType[NextLA.BFloat16, Float16, Float32, T])
        A = mxp_spd(T, n)
        M = SymmMixedPrec(A, uplo; precisions=P)
        @test size(M) == (n, n)
        R = reconstruct_matrix(M)
        @test eltype(R) === T
        @test R == transpose(R)
        @test mxp_storage_ok(R, A, mxp_eltype_map(M))
        @test all(M[i, j] == R[i, j] for i in 1:n, j in 1:n)
        @test transpose(M) === M
    end

    @testset "one precision: a single leaf, $T, uplo=$uplo" for T in (Float32, Float64), uplo in ('L', 'U')
        A = mxp_spd(T, 8)
        M = SymmMixedPrec(A, uplo; precisions=DataType[T])
        @test M.Base !== nothing && M.OffDiag === nothing
        @test reconstruct_matrix(M) == A
    end

    # A leaf is stored in T_Base, so values past floatmax(T_Base) are kept as
    # stored values times a scale; read back in T_Base they are Inf.
    @testset "a leaf past floatmax of its type is stored with a scale, uplo=$uplo" for uplo in ('L', 'U')
        A = fill(1.0e39, 4, 4) + 1.0e40I
        M = SymmMixedPrec(A, uplo; precisions=DataType[Float32])
        @test M.Base_scale != 1
        @test Float64.(M.Base) .* M.Base_scale ≈ (uplo == 'L' ? tril(A) : triu(A)) rtol=2mxp_u(Float32)
        @test all(isinf, reconstruct_matrix(M))
    end

    @testset "the other triangle is not read, uplo=$uplo" for uplo in ('L', 'U')
        A = mxp_spd(Float64, 8)
        G = copy(A)
        uplo == 'L' ? (G[1, 8] = NaN) : (G[8, 1] = NaN)
        M = SymmMixedPrec(G, uplo; precisions=DataType[Float32, Float64])
        @test mxp_storage_ok(reconstruct_matrix(M), A, mxp_eltype_map(M))
        # and inside a leaf (A11 is a 4x4 leaf here), whose unread half is zeroed
        # before it is stored
        H = copy(A)
        uplo == 'L' ? (H[1, 2] = NaN) : (H[2, 1] = NaN)
        M = SymmMixedPrec(H, uplo; precisions=DataType[Float32, Float64])
        @test mxp_storage_ok(reconstruct_matrix(M), A, mxp_eltype_map(M))
    end

    @testset "a scaled off-diagonal block reads back unscaled, uplo=$uplo" for uplo in ('L', 'U')
        A = ones(4, 4)
        A[3:4, 1:2] .= 1.0e5
        A[1:2, 3:4] .= 1.0e5
        M = SymmMixedPrec(A, uplo; precisions=DataType[Float16, Float64])
        @test M.OffDiag_scale != 1
        @test reconstruct_matrix(M) ≈ A rtol=2mxp_u(Float16)
        @test M[3, 1] ≈ 1.0e5 rtol=2mxp_u(Float16)
        @test M[1, 3] ≈ 1.0e5 rtol=2mxp_u(Float16)
    end

    # An input matrix of a 16-bit type (the leaves are Float32): every element is
    # stored within one rounding of its block's type, and exactly where the block
    # type is the input type.
    @testset "$S input, uplo=$uplo, precisions=$P" for S in (Float16, NextLA.BFloat16), uplo in ('L', 'U'),
            P in (DataType[Float16, Float32], DataType[NextLA.BFloat16, Float32])
        A = S.(mxp_spd(Float32, 13))
        M = SymmMixedPrec(A, uplo; precisions=P)
        R = reconstruct_matrix(M)
        @test eltype(R) === Float32
        @test mxp_storage_ok(R, A, mxp_eltype_map(M))
        P[1] === S && @test R == Float32.(A)
    end

    # A view as input (a principal submatrix of a symmetric matrix is symmetric):
    # the same container as from a copy of the view.
    @testset "a view as input, uplo=$uplo, precisions=$P" for uplo in ('L', 'U'),
            P in (DataType[Float32, Float64], DataType[Float16, Float32, Float64])
        V = view(mxp_spd(Float64, 16), 2:14, 2:14)
        M = SymmMixedPrec(V, uplo; precisions=P)
        R = reconstruct_matrix(M)
        @test R == reconstruct_matrix(SymmMixedPrec(copy(V), uplo; precisions=P))
        @test mxp_storage_ok(R, copy(V), mxp_eltype_map(M))
    end

    @testset "arguments are checked" begin
        @test_throws ArgumentError SymmMixedPrec(rand(4, 4), 'X'; precisions=DataType[Float64])
        @test_throws DimensionMismatch SymmMixedPrec(rand(4, 8), 'L'; precisions=DataType[Float64])
        @test_throws BoundsError SymmMixedPrec(rand(4, 4), 'L'; precisions=DataType[Float64])[5, 1]
    end
end

# On each GPU backend: built, stored and read back on the device.
@testset "SymmMixedPrec [$name]" for (name, AT, _) in filter(b -> b[1] != "CPU", MXP_BACKENDS)
    Random.seed!(2048)
    n = 13
    P = DataType[Float16, Float32, Float64]

    @testset "builds and reconstructs on the device, uplo=$uplo" for uplo in ('L', 'U')
        A = mxp_spd(Float64, n)
        M = SymmMixedPrec(AT(A), uplo; precisions=P)
        @test M.OffDiag isa AT
        R = reconstruct_matrix(M)
        @test R isa AT
        @test mxp_storage_ok(Array(R), A, mxp_eltype_map(M))
    end

    # One precision is no split: the whole matrix is a single leaf.
    @testset "a single leaf reconstructs on the device, uplo=$uplo" for uplo in ('L', 'U')
        M = SymmMixedPrec(AT(mxp_spd(Float64, n)), uplo; precisions=DataType[Float64])
        @test M.Base isa AT
        R = reconstruct_matrix(M)
        @test R isa AT
        @test Array(R) == reconstruct_matrix(mxp_host(M))
    end
end
