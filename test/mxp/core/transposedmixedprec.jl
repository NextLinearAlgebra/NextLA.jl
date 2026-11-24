# The lazy transpose and adjoint, src/mxp/core/transposedmixedprec.jl.
# Helpers and error bounds: test/setup/mxp.jl.

@testset "TransposedMixedPrec [CPU]" begin
    Random.seed!(2047)

    # Each element of the wrapper is the container's element at the swapped index,
    # exactly. For TriMixedPrec the stored triangle moves to the other side and the
    # rest reads zero. n = 13 with three types is three levels.
    @testset "$(nameof(kind))$(uplo === nothing ? "" : " uplo=$uplo"), n=$n, precisions=$P, $f" for
            (kind, uplo, n, P) in ((FullMixedPrec, nothing, 8, DataType[Float32, Float64]),
                                   (FullMixedPrec, nothing, 13, DataType[Float16, Float32, Float64]),
                                   (TriMixedPrec, 'L', 13, DataType[Float16, Float32, Float64]),
                                   (TriMixedPrec, 'U', 13, DataType[Float16, Float32, Float64])),
            f in (transpose, adjoint)
        A = mxp_rand(Float64, n, n)
        M = kind === FullMixedPrec ? FullMixedPrec(A; precisions=P) : TriMixedPrec(A, uplo; precisions=P)
        Mᵀ = f(M)
        @test parent(Mᵀ) === M
        @test transpose(Mᵀ) === M
        @test adjoint(Mᵀ) === M
        @test size(Mᵀ) == reverse(size(M))
        @test eltype(Mᵀ) === eltype(M)
        @test all(Mᵀ[i, j] == M[j, i] for i in 1:n, j in 1:n)
        if kind === TriMixedPrec
            @test all(iszero(Mᵀ[i, j]) for i in 1:n, j in 1:n if (uplo == 'L' ? i > j : i < j))
        end
        @test_throws BoundsError Mᵀ[n + 1, 1]
    end
end

# On each GPU backend: wrapping a device container copies nothing and keeps its
# type. (Reading one element of a device container is scalar indexing, which is
# not supported, so no element is read here.)
@testset "TransposedMixedPrec [$name]" for (name, AT, _) in filter(b -> b[1] != "CPU", MXP_BACKENDS)
    Random.seed!(2048)
    n = 13
    P = DataType[Float16, Float32, Float64]

    @testset "$(nameof(kind)), $f" for kind in (FullMixedPrec, TriMixedPrec), f in (transpose, adjoint)
        A = AT(mxp_rand(Float64, n, n))
        M = kind === FullMixedPrec ? FullMixedPrec(A; precisions=P) : TriMixedPrec(A, 'U'; precisions=P)
        Mᵀ = f(M)
        @test parent(Mᵀ) === M
        @test transpose(Mᵀ) === M
        @test adjoint(Mᵀ) === M
        @test size(Mᵀ) == (n, n)
        @test eltype(Mᵀ) === eltype(M)
    end
end
