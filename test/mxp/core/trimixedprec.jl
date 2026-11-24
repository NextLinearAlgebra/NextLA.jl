# TriMixedPrec, src/mxp/core/trimixedprec.jl.
# Helpers and error bounds: test/setup/mxp.jl.

@testset "TriMixedPrec [CPU]" begin
    Random.seed!(2047)

    @testset "$T, uplo=$uplo, n=$n, precisions=$P" for T in (Float32, Float64), uplo in ('L', 'U'),
            n in (1, 7, 8, 13), P in (DataType[Float32, T], DataType[Float16, Float32, T],
                                      DataType[NextLA.BFloat16, Float16, Float32, T])
        A = mxp_tri(uplo, mxp_rand(T, n, n))
        M = TriMixedPrec(A, uplo; precisions=P)
        @test size(M) == (n, n)
        R = reconstruct_matrix(M)
        @test eltype(R) === T
        @test mxp_storage_ok(R, A, mxp_eltype_map(M))
        @test all(M[i, j] == R[i, j] for i in 1:n, j in 1:n)
    end

    @testset "one precision: a single leaf, $T, uplo=$uplo" for T in (Float32, Float64), uplo in ('L', 'U')
        A = mxp_tri(uplo, mxp_rand(T, 8, 8))
        M = TriMixedPrec(A, uplo; precisions=DataType[T])
        @test M.Base !== nothing && M.OffDiag === nothing
        @test reconstruct_matrix(M) == A
    end

    # A leaf is stored in T_Base, so values past floatmax(T_Base) are kept as
    # stored values times a scale; read back in T_Base they are Inf in the stored
    # triangle, and the other triangle stays zero.
    @testset "a leaf past floatmax of its type is stored with a scale, uplo=$uplo" for uplo in ('L', 'U')
        A = mxp_tri(uplo, fill(1.0e39, 4, 4)) + 1.0e40I
        M = TriMixedPrec(A, uplo; precisions=DataType[Float32])
        @test M.Base_scale != 1
        @test Float64.(M.Base) .* M.Base_scale ≈ A rtol=2mxp_u(Float32)
        R = reconstruct_matrix(M)
        @test all(isinf(R[i, j]) for i in 1:4, j in 1:4 if (uplo == 'L' ? i >= j : i <= j))
        @test all(iszero(R[i, j]) for i in 1:4, j in 1:4 if (uplo == 'L' ? i < j : i > j))
    end

    # A NaN in the unread triangle, in the other off-diagonal block and inside a
    # leaf (A11 is a 4x4 leaf here), is never read.
    @testset "the other triangle is not read, uplo=$uplo" for uplo in ('L', 'U')
        A = mxp_tri(uplo, mxp_rand(Float64, 8, 8))
        G = copy(A)
        uplo == 'L' ? (G[1, 8] = NaN; G[1, 2] = NaN) : (G[8, 1] = NaN; G[2, 1] = NaN)
        M = TriMixedPrec(G, uplo; precisions=DataType[Float32, Float64])
        @test mxp_storage_ok(reconstruct_matrix(M), A, mxp_eltype_map(M))
    end

    @testset "every block is a copy of A, uplo=$uplo, precisions=$P" for uplo in ('L', 'U'),
            P in (DataType[Float64], DataType[Float32, Float64])
        A = mxp_tri(uplo, mxp_rand(Float64, 8, 8))
        M = TriMixedPrec(A, uplo; precisions=P)
        R = reconstruct_matrix(M)
        A .= 0
        @test reconstruct_matrix(M) == R
    end

    @testset "only the uplo triangle of a full input is kept, uplo=$uplo" for uplo in ('L', 'U')
        A = mxp_rand(Float64, 8, 8)
        M = TriMixedPrec(A, uplo; precisions=DataType[Float32, Float64])
        @test mxp_storage_ok(reconstruct_matrix(M), mxp_tri(uplo, A), mxp_eltype_map(M))
    end

    @testset "converts from SymmMixedPrec without copying, uplo=$uplo" for uplo in ('L', 'U')
        n = 13
        S = SymmMixedPrec(mxp_spd(Float64, n), uplo; precisions=DataType[Float16, Float32, Float64])
        M = TriMixedPrec(S)
        @test size(M) == size(S) && M.uplo == uplo
        @test M.OffDiag === S.OffDiag && M.A11.OffDiag === S.A11.OffDiag
        @test all(M[i, j] == S[i, j] for i in 1:n, j in 1:n if (uplo == 'L' ? i >= j : i <= j))
        @test all(iszero(M[i, j]) for i in 1:n, j in 1:n if (uplo == 'L' ? i < j : i > j))
    end

    @testset "converting from SymmMixedPrec keeps the scales, uplo=$uplo" for uplo in ('L', 'U')
        A = ones(4, 4)
        A[3:4, 1:2] .= 1.0e5
        A[1:2, 3:4] .= 1.0e5
        S = SymmMixedPrec(A, uplo; precisions=DataType[Float16, Float64])
        M = TriMixedPrec(S)
        @test M.OffDiag_scale == S.OffDiag_scale != 1
        @test reconstruct_matrix(M) ≈ mxp_tri(uplo, A) rtol=2mxp_u(Float16)
        L = SymmMixedPrec(fill(1.0e39, 4, 4) + 1.0e40I, uplo; precisions=DataType[Float32])
        ML = TriMixedPrec(L)
        @test ML.Base === L.Base && ML.Base_scale == L.Base_scale != 1
    end

    @testset "a scaled off-diagonal block reads back unscaled, uplo=$uplo" for uplo in ('L', 'U')
        A = mxp_tri(uplo, ones(4, 4))
        uplo == 'L' ? (A[3:4, 1:2] .= 1.0e5) : (A[1:2, 3:4] .= 1.0e5)
        M = TriMixedPrec(A, uplo; precisions=DataType[Float16, Float64])
        @test M.OffDiag_scale != 1
        @test reconstruct_matrix(M) ≈ A rtol=2mxp_u(Float16)
    end

    # An input matrix of a 16-bit type (the leaves are Float32): every element is
    # stored within one rounding of its block's type, and exactly where the block
    # type is the input type.
    @testset "$S input, uplo=$uplo, precisions=$P" for S in (Float16, NextLA.BFloat16), uplo in ('L', 'U'),
            P in (DataType[Float16, Float32], DataType[NextLA.BFloat16, Float32])
        A = S.(mxp_tri(uplo, mxp_rand(Float32, 13, 13)))
        M = TriMixedPrec(A, uplo; precisions=P)
        R = reconstruct_matrix(M)
        @test eltype(R) === Float32
        @test mxp_storage_ok(R, A, mxp_eltype_map(M))
        P[1] === S && @test R == Float32.(A)
    end

    # A view as input: the same container as from a copy of the view.
    @testset "a view as input, uplo=$uplo, precisions=$P" for uplo in ('L', 'U'),
            P in (DataType[Float32, Float64], DataType[Float16, Float32, Float64])
        V = view(mxp_rand(Float64, 16, 16), 2:14, 3:15)
        M = TriMixedPrec(V, uplo; precisions=P)
        R = reconstruct_matrix(M)
        @test R == reconstruct_matrix(TriMixedPrec(copy(V), uplo; precisions=P))
        @test mxp_storage_ok(R, mxp_tri(uplo, copy(V)), mxp_eltype_map(M))
    end

    @testset "arguments are checked" begin
        @test_throws ArgumentError TriMixedPrec(rand(4, 4), 'X'; precisions=DataType[Float64])
        @test_throws DimensionMismatch TriMixedPrec(rand(4, 8), 'L'; precisions=DataType[Float64])
        @test_throws BoundsError TriMixedPrec(rand(4, 4), 'L'; precisions=DataType[Float64])[1, 0]
    end
end

# On each GPU backend: built, stored and read back on the device.
@testset "TriMixedPrec [$name]" for (name, AT, _) in filter(b -> b[1] != "CPU", MXP_BACKENDS)
    Random.seed!(2048)
    n = 13
    P = DataType[Float16, Float32, Float64]

    @testset "builds and reconstructs on the device, uplo=$uplo" for uplo in ('L', 'U')
        A = mxp_tri(uplo, mxp_rand(Float64, n, n))
        M = TriMixedPrec(AT(A), uplo; precisions=P)
        @test M.OffDiag isa AT
        R = reconstruct_matrix(M)
        @test R isa AT
        @test mxp_storage_ok(Array(R), A, mxp_eltype_map(M))
    end

    # One precision is no split: the whole matrix is a single leaf.
    @testset "a single leaf reconstructs on the device, uplo=$uplo" for uplo in ('L', 'U')
        M = TriMixedPrec(AT(mxp_spd(Float64, n)), uplo; precisions=DataType[Float64])
        @test M.Base isa AT
        R = reconstruct_matrix(M)
        @test R isa AT
        @test Array(R) == reconstruct_matrix(mxp_host(M))
    end
end
