# adaptive_precisions, src/mxp/core/adaptive.jl.
# Helpers and error bounds: test/setup/mxp.jl.

@testset "adaptive_precisions [CPU]" begin
    Random.seed!(2047)

    n = 16
    A = tril(rand(Float64, n, n)) + n * I

    @testset "one precision per level, then the leaf precision" begin
        u = adaptive_precisions(A)
        @test u isa Vector{DataType}
        @test all(T -> T in (Float32, Float64), u)
        @test u[end] === Float64
        @test size(FullMixedPrec(A; precisions=u)) == (n, n)
    end

    # The rule in the docstring, worked out for one level: n = 8 and n_min = 4
    # split once, into 4 + 4. E = ‖A21‖ / ‖tril(A)‖ (here 0.0141) and the level
    # takes the coarsest type with u(T) <= epsilon / (sqrt(2) E). The three
    # epsilons give a target of 5e-1, 5e-6 and 5e-11, so each type is chosen once.
    @testset "the level type follows the documented rule, epsilon=$ϵ" for (ϵ, T) in
            ((1e-2, Float16), (1e-7, Float32), (1e-12, Float64))
        B = Matrix(1.0I, 8, 8)
        B[5:8, 1:4] .= 0.01
        E = norm(B[5:8, 1:4]) / norm(tril(B))
        U = DataType[Float16, Float32, Float64]
        @test U[findfirst(S -> mxp_u(S) <= ϵ / (sqrt(2) * E), U)] === T
        @test adaptive_precisions(B, U, 4, ϵ) == DataType[T, Float64]
    end

    @testset "n_min controls the number of levels" begin
        @test length(adaptive_precisions(A, DataType[Float32, Float64], 2)) >
              length(adaptive_precisions(A, DataType[Float32, Float64], 8))
    end

    @testset "a matrix too small to split gets only the leaf precision" begin
        @test adaptive_precisions(A[1:3, 1:3]) == DataType[Float64]
    end

    @testset "the leaves are no finer than eltype(A)" begin
        u = adaptive_precisions(Float32.(A), DataType[Float16, Float32, Float64])
        @test u[end] === Float32
        @test all(T -> T in (Float16, Float32), u)
        @test adaptive_precisions(Float16.(A))[end] === Float32
        @test adaptive_precisions(Float32.(A), DataType[Float16])[end] === Float16
    end

    @testset "a non-float matrix gets the finest type of U for its leaves" begin
        @test adaptive_precisions(round.(Int, 10A))[end] === Float64
    end

    @testset "a 16-bit matrix with large block norms is finite" begin
        H = Float16.(fill(100, 2048, 2048) + 1000I)
        @test adaptive_precisions(H; uplo='F')[end] === Float32
    end

    @testset "a BFloat16 matrix with very large or very small entries" begin
        for v in (1f20, 1f-25)
            B = NextLA.BFloat16.(fill(v, 64, 64) + 10v * I)
            @test adaptive_precisions(B, DataType[NextLA.BFloat16, Float32]; uplo='F')[end] === Float32
        end
    end

    @testset "U may be given in any order" begin
        @test adaptive_precisions(A, DataType[Float64, Float32]) == adaptive_precisions(A)
    end

    @testset "uplo='U' reads the upper triangle" begin
        @test adaptive_precisions(copy(transpose(A)); uplo='U') == adaptive_precisions(A)
    end

    @testset "uplo='F' measures both off-diagonal blocks" begin
        F = [i == j ? 1.0 : i > j ? 1.0e-12 : 1.0 for i in 1:n, j in 1:n]
        @test adaptive_precisions(F; uplo='L')[1] === Float32
        @test adaptive_precisions(F; uplo='F')[1] === Float64
        @test size(FullMixedPrec(F; precisions=adaptive_precisions(F; uplo='F'))) == (n, n)
    end

    @testset "the other triangle is not read" begin
        G = copy(A)
        G[1, n] = NaN
        @test adaptive_precisions(G) == adaptive_precisions(A)
    end

    @testset "arguments are checked" begin
        @test_throws DimensionMismatch adaptive_precisions(rand(Float64, 4, 8))
        @test_throws ArgumentError adaptive_precisions(zeros(0, 0))
        @test_throws ArgumentError adaptive_precisions(A, DataType[])
        @test_throws ArgumentError adaptive_precisions(A, DataType[Int])
        @test_throws ArgumentError adaptive_precisions(A, DataType[Float32, Float64], 0)
        @test_throws ArgumentError adaptive_precisions(A, DataType[Float32, Float64], 4, 0.0)
        @test_throws ArgumentError adaptive_precisions(A; uplo='X')
        @test_throws ArgumentError adaptive_precisions(zeros(n, n))
        @test_throws ArgumentError adaptive_precisions(fill(NaN, n, n))
    end
end

# On each GPU backend: the same precisions from a device matrix.
@testset "adaptive_precisions [$name]" for (name, AT, _) in filter(b -> b[1] != "CPU", MXP_BACKENDS)
    Random.seed!(2048)
    n = 13

    @testset "adaptive_precisions takes a device matrix" begin
        S = mxp_spd(Float64, n)
        @test adaptive_precisions(AT(S)) == adaptive_precisions(S)
    end
end
