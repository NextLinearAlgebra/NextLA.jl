using LinearAlgebra
using Random
using Test

function _potrf_batched_spd(::Type{T}, n::Int, batch::Int) where {T}
    rng = MersenneTwister(73)
    A = Array{T}(undef, n, n, batch)
    for bid in 1:batch
        X = randn(rng, T, n, n)
        A[:, :, bid] = X * X' + T(n) * I
    end
    return A
end

@testset "batched POTRF wrappers" begin
    A = _potrf_batched_spd(Float32, 16, 3)
    expected = map(axes(A, 3)) do bid
        Matrix(cholesky(Hermitian(view(A, :, :, bid), :U)).U)
    end

    for (name, AT, sync) in backends
        @testset "$name" begin
            Adev = _to_backend(AT, copy(A))
            @test occursin(_expected_potrf_batched_file(name),
                           _method_file(NextLA.potrf_batched!, 'U', Adev))
            if name == "Metal"
                @test_throws ArgumentError NextLA.potrf_batched!('U', Adev)
                continue
            end

            result = NextLA.potrf_batched!('U', Adev)
            sync(Adev)
            if result isa Tuple
                @test Array(result[2]) == zeros(Int32, size(A, 3))
            end
            actual = Array(Adev)
            for bid in axes(A, 3)
                @test UpperTriangular(view(actual, :, :, bid)) ≈
                      UpperTriangular(expected[bid]) rtol=2f-5
            end
        end
    end

    failure = zeros(Float32, 8, 8, 2)
    failure[:, :, 1] = Matrix(Diagonal(fill(2f0, 8)))
    failure[:, :, 2] = Matrix(Diagonal(vcat(fill(2f0, 4), -1f0, fill(2f0, 3))))
    _, status = NextLA.potrf_batched!('U', failure)
    @test status == Int32[0, 5]
end
