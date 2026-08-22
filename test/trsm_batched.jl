using LinearAlgebra
using Random
using Test

function _trsm_batched_inputs(::Type{T}, n::Int, nrhs::Int, batch::Int) where {T}
    rng = MersenneTwister(41)
    A = Array{T}(undef, n, n, batch)
    B = rand(rng, T, n, nrhs, batch)
    for bid in 1:batch
        tile = Matrix(UpperTriangular(rand(rng, T, n, n)))
        tile .+= Diagonal(fill(T(4), n))
        A[:, :, bid] = tile
    end
    return A, B
end

function _trsm_batched_reference!(A, B)
    for bid in axes(B, 3)
        BLAS.trsm!('L', 'U', 'N', 'N', one(eltype(B)),
                   view(A, :, :, bid), view(B, :, :, bid))
    end
    return B
end

@testset "batched TRSM wrappers" begin
    A, B = _trsm_batched_inputs(Float32, 16, 4, 3)
    expected = _trsm_batched_reference!(copy(A), copy(B))

    for (name, AT, sync) in backends
        @testset "$name strided batch" begin
            Adev = _to_backend(AT, copy(A))
            Bdev = _to_backend(AT, copy(B))
            @test occursin(_expected_trsm_batched_file(name),
                           _method_file(NextLA.trsm_batched!,
                                        'L', 'U', 'N', 'N', Adev, Bdev))
            if name == "Metal"
                @test_throws ArgumentError NextLA.trsm_batched!('L',
                                                               'U', 'N', 'N', Adev, Bdev)
            else
                NextLA.trsm_batched!('L', 'U', 'N', 'N', Adev, Bdev)
                sync(Bdev)
                @test Array(Bdev) ≈ expected rtol=2f-5
            end
        end

        @testset "$name pointer batch" begin
            Abatch = [copy(view(A, :, :, bid)) for bid in axes(A, 3)]
            Bbatch = [copy(view(B, :, :, bid)) for bid in axes(B, 3)]
            Adev = _to_backend(AT, Abatch)
            Bdev = _to_backend(AT, Bbatch)
            if name == "Metal"
                @test_throws ArgumentError NextLA.trsm_batched!('L',
                                                               'U', 'N', 'N', Adev, Bdev)
            else
                NextLA.trsm_batched!('L', 'U', 'N', 'N', Adev, Bdev)
                sync(first(Bdev))
                for bid in eachindex(Bdev)
                    @test Array(Bdev[bid]) ≈ view(expected, :, :, bid) rtol=2f-5
                end
            end
        end
    end
end
