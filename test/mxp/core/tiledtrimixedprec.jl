# TiledTriMixedPrec, src/mxp/core/tiledtrimixedprec.jl.
# Helpers and error bounds: test/setup/mxp.jl.

@testset "TiledTriMixedPrec [CPU]" begin
    Random.seed!(2047)

    # The tile sizes are TriMixedPrec's split, n/2 for a power of 2 and else the
    # largest power of 2 below n, stopped once a block has at most threshold rows.
    @testset "uplo=$uplo, n=$n, threshold=$t, precisions=$P" for uplo in ('L', 'U'),
            (n, t, sizes) in ((8, 4, [4, 4]), (8, 2, [2, 2, 2, 2]), (7, 3, [2, 2, 3]),
                              (13, 4, [4, 4, 4, 1]), (5, 1, [1, 1, 1, 1, 1]), (6, 6, [6])),
            P in (DataType[Float32, Float64], DataType[Float16, Float64], DataType[NextLA.BFloat16, Float32],
                  DataType[Float64])
        A = mxp_tri(uplo, mxp_rand(Float64, n, n))
        M = TiledTriMixedPrec(A, uplo; precisions=P, threshold=t)
        @test size(M) == (n, n) && size(M, 1) == n
        @test size.(M.diag, 1) == sizes
        R = reconstruct_matrix(M)
        @test eltype(R) === P[end]
        @test mxp_storage_ok(R, A, mxp_eltype_map(M))
        @test all(M[i, j] == R[i, j] for i in 1:n, j in 1:n)
        @test all(transpose(M)[i, j] == M[j, i] for i in 1:n, j in 1:n)
    end

    @testset "a scaled tile reads back unscaled, uplo=$uplo" for uplo in ('L', 'U')
        A = mxp_tri(uplo, ones(8, 8))
        uplo == 'L' ? (A[5:8, 1:4] .= 1.0e5) : (A[1:4, 5:8] .= 1.0e5)
        M = TiledTriMixedPrec(A, uplo; precisions=DataType[Float16, Float64], threshold=4)
        @test M.off_scales[uplo == 'L' ? CartesianIndex(2, 1) : CartesianIndex(1, 2)] != 1
        @test reconstruct_matrix(M) ≈ A rtol=2mxp_u(Float16)
    end

    # Tiles are stored in their type, so values past its floatmax are kept as
    # stored values times a scale; read back in T_Diag they are Inf in the stored
    # triangle, and the other triangle stays zero.
    @testset "a diagonal tile past floatmax of its type is stored with a scale, uplo=$uplo" for uplo in ('L', 'U')
        A = mxp_tri(uplo, fill(1.0e39, 8, 8)) + 1.0e40I
        M = TiledTriMixedPrec(A, uplo; precisions=DataType[Float32, Float32], threshold=4)
        @test all(!=(1), M.diag_scales)
        @test Float64.(M.diag[2]) .* M.diag_scales[2] ≈ mxp_tri(uplo, A[5:8, 5:8]) rtol=2mxp_u(Float32)
        R = reconstruct_matrix(M)
        @test all(isinf(R[i, j]) for i in 1:8, j in 1:8 if (uplo == 'L' ? i >= j : i <= j))
        @test all(iszero(R[i, j]) for i in 1:8, j in 1:8 if (uplo == 'L' ? i < j : i > j))
    end

    # A full input keeps only its uplo triangle; a NaN in the other triangle, in an
    # off-diagonal tile and inside a diagonal tile, is never read.
    @testset "only the uplo triangle of a full input is read, uplo=$uplo" for uplo in ('L', 'U')
        A = mxp_rand(Float64, 8, 8)
        G = copy(A)
        uplo == 'L' ? (G[1, 8] = NaN; G[1, 2] = NaN) : (G[8, 1] = NaN; G[2, 1] = NaN)
        M = TiledTriMixedPrec(G, uplo; precisions=DataType[Float32, Float64], threshold=4)
        @test mxp_storage_ok(reconstruct_matrix(M), mxp_tri(uplo, A), mxp_eltype_map(M))
    end

    # An input matrix of a 16-bit type (diagonal tiles are Float32): every element
    # is stored within one rounding of its tile's type, and exactly where the
    # off-diagonal type is the input type.
    @testset "$S input, uplo=$uplo, precisions=$P" for S in (Float16, NextLA.BFloat16), uplo in ('L', 'U'),
            P in (DataType[Float16, Float32], DataType[NextLA.BFloat16, Float32])
        A = S.(mxp_tri(uplo, mxp_rand(Float32, 13, 13)))
        M = TiledTriMixedPrec(A, uplo; precisions=P, threshold=4)
        R = reconstruct_matrix(M)
        @test eltype(R) === Float32
        @test mxp_storage_ok(R, A, mxp_eltype_map(M))
        P[1] === S && @test R == Float32.(A)
    end

    # A view as input: the same container as from a copy of the view.
    @testset "a view as input, uplo=$uplo, precisions=$P" for uplo in ('L', 'U'),
            P in (DataType[Float32, Float64], DataType[Float16, Float64])
        V = view(mxp_rand(Float64, 16, 16), 2:14, 3:15)
        M = TiledTriMixedPrec(V, uplo; precisions=P, threshold=4)
        R = reconstruct_matrix(M)
        @test R == reconstruct_matrix(TiledTriMixedPrec(copy(V), uplo; precisions=P, threshold=4))
        @test mxp_storage_ok(R, mxp_tri(uplo, copy(V)), mxp_eltype_map(M))
    end

    @testset "arguments are checked" begin
        A = tril(rand(8, 8))
        @test_throws ArgumentError TiledTriMixedPrec(A, 'X'; precisions=DataType[Float64], threshold=2)
        @test_throws ArgumentError TiledTriMixedPrec(A, 'L'; precisions=DataType[Float64], threshold=0)
        @test_throws ArgumentError TiledTriMixedPrec(A, 'L'; precisions=DataType[Float16, Float32, Float64], threshold=2)
        @test_throws DimensionMismatch TiledTriMixedPrec(rand(4, 8), 'L'; precisions=DataType[Float64], threshold=2)
    end

    @testset "out-of-bounds indexing throws" begin
        M = TiledTriMixedPrec(tril(rand(8, 8)), 'L'; precisions=DataType[Float32, Float64], threshold=2)
        @test_throws BoundsError M[0, 1]
        @test_throws BoundsError M[9, 1]
    end
end

# On each GPU backend: built and read back on the device.
@testset "TiledTriMixedPrec [$name]" for (name, AT, _) in filter(b -> b[1] != "CPU", MXP_BACKENDS)
    Random.seed!(2048)
    n = 13

    @testset "builds and reconstructs on the device, uplo=$uplo" for uplo in ('L', 'U')
        A = mxp_tri(uplo, mxp_rand(Float64, n, n))
        M = TiledTriMixedPrec(AT(A), uplo; precisions=DataType[Float16, Float64], threshold=4)
        R = reconstruct_matrix(M)
        @test R isa AT
        @test mxp_storage_ok(Array(R), A, mxp_eltype_map(M))
    end
end
