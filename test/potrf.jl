using Test
using LinearAlgebra
using Random

function _potrf_make_spd(::Type{T},
    n::Int;
    seed::Int=7) where {T}
    rng = MersenneTwister(seed)
    X = randn(rng, T, n, n)
    return X * X' + T(n) * I
end

_potrf_to_backend(::Type{Array}, x::AbstractArray) = copy(x)
_potrf_to_backend(AT, x::AbstractArray) = AT(copy(x))

function _potrf_triangle_matches(actual::AbstractMatrix,
                                 expected::AbstractMatrix;
                                 uplo::Char,
                                 rtol)
    if uplo == 'L'
        @test LowerTriangular(actual) ≈ LowerTriangular(expected) rtol=rtol
    else
        @test UpperTriangular(actual) ≈ UpperTriangular(expected) rtol=rtol
    end
end

function _potrf_make_spd_batch(::Type{T},
                               n::Int,
                               batch_count::Int) where {T}
    A = Array{T}(undef, n, n, batch_count)
    for batch in 1:batch_count
        A[:, :, batch] = _potrf_make_spd(T, n; seed=7 + batch)
    end
    return A
end

@testset "blocked potrf" begin
    for T in (Float32, Float64)
        for n in (48, 50)
            A_host = Matrix(_potrf_make_spd(T, n))

            for uplo in ('L', 'U')
                A_ref = copy(A_host)
                F_ref = cholesky!(Hermitian(A_ref, Symbol(uplo)))
                triangle_ref = uplo == 'L' ? Matrix(F_ref.L) : Matrix(F_ref.U)

                for (name, AT, sync) in backends
                    name == "CPU" && continue

                    @testset "$name $T n=$n uplo=$uplo" begin
                        A_dev = _potrf_to_backend(AT, A_host)
                        _, status_dev = NextLA.potrf!(uplo, A_dev, Val(16), Val(8), Val(4), Val(8); check=false)
                        sync(A_dev)

                        A_actual = Array(A_dev)
                        status_actual = Array(status_dev)
                        @test status_actual == Int32[0]
                        _potrf_triangle_matches(A_actual, triangle_ref; uplo, rtol=200 * eps(T))
                    end
                end
            end
        end
    end
end

@testset "batched blocked potrf" begin
    batch_count = 3

    for T in (Float32, Float64)
        for n in (48, 50)
            A_host = _potrf_make_spd_batch(T, n, batch_count)

            for uplo in ('L', 'U')
                triangle_refs = map(1:batch_count) do batch
                    A_ref = copy(@view A_host[:, :, batch])
                    F_ref = cholesky!(Hermitian(A_ref, Symbol(uplo)))
                    uplo == 'L' ? Matrix(F_ref.L) : Matrix(F_ref.U)
                end

                for (name, AT, sync) in backends
                    name == "CPU" && continue

                    @testset "$name $T n=$n uplo=$uplo" begin
                        A_dev = _potrf_to_backend(AT, A_host)
                        _, status_dev = NextLA.potrf!(uplo, A_dev, Val(16), Val(8), Val(4), Val(8); check=false)
                        sync(A_dev)

                        A_actual = Array(A_dev)
                        @test Array(status_dev) == zeros(Int32, batch_count)
                        for batch in 1:batch_count
                            _potrf_triangle_matches(
                                @view(A_actual[:, :, batch]),
                                triangle_refs[batch];
                                uplo,
                                rtol=200 * eps(T),
                            )
                        end
                    end
                end
            end
        end
    end
end

@testset "blocked potrf split outer tails" begin
    batch_count = 3

    for (n, NB, IB) in ((35, 32, 8), (49, 32, 8), (56, 32, 8))
        A_host = _potrf_make_spd_batch(Float32, n, batch_count)

        for uplo in ('L', 'U')
            triangle_refs = map(1:batch_count) do batch
                A_ref = copy(@view A_host[:, :, batch])
                F_ref = cholesky!(Hermitian(A_ref, Symbol(uplo)))
                uplo == 'L' ? Matrix(F_ref.L) : Matrix(F_ref.U)
            end

            for (name, AT, sync) in backends
                name == "CPU" && continue

                @testset "$name n=$n NB=$NB IB=$IB uplo=$uplo" begin
                    A_dev = _potrf_to_backend(AT, A_host)
                    _, status_dev = NextLA.potrf!(uplo, A_dev, Val(NB), Val(IB), Val(4), Val(8); check=false)
                    sync(A_dev)

                    A_actual = Array(A_dev)
                    @test Array(status_dev) == zeros(Int32, batch_count)
                    for batch in 1:batch_count
                        _potrf_triangle_matches(
                            @view(A_actual[:, :, batch]),
                            triangle_refs[batch];
                            uplo,
                            rtol=200 * eps(Float32),
                        )
                    end
                end
            end
        end
    end
end

@testset "blocked potrf failure status" begin
    A = Matrix(Diagonal(vcat(fill(4.0f0, 18), -1.0f0, fill(4.0f0, 13))))

    for uplo in ('L', 'U')
        for (name, AT, sync) in backends
            name == "CPU" && continue

            @testset "$name uplo=$uplo" begin
                A_dev = _potrf_to_backend(AT, A)
                _, status_dev = NextLA.potrf!(uplo, A_dev, Val(16), Val(8), Val(4), Val(8); check=false)
                sync(A_dev)

                @test Array(status_dev) == Int32[19]
                @test_throws PosDefException NextLA.potrf!(uplo, _potrf_to_backend(AT, A), Val(16), Val(8), Val(4), Val(8))
            end
        end
    end
end

@testset "batched blocked potrf failure status" begin
    n = 32
    A = _potrf_make_spd_batch(Float32, n, 3)
    A[:, :, 2] = Matrix(Diagonal(vcat(fill(4.0f0, 6), -1.0f0, fill(4.0f0, n - 7))))
    A[:, :, 3] = Matrix(Diagonal(vcat(fill(4.0f0, 24), -1.0f0, fill(4.0f0, n - 25))))

    for uplo in ('L', 'U')
        for (name, AT, sync) in backends
            name == "CPU" && continue

            @testset "$name uplo=$uplo" begin
                A_dev = _potrf_to_backend(AT, A)
                _, status_dev = NextLA.potrf!(uplo, A_dev, Val(16), Val(8), Val(4), Val(8); check=false)
                sync(A_dev)

                @test Array(status_dev) == Int32[0, 7, 25]
            end
        end
    end
end

@testset "batched blocked potrf tail failure status" begin
    n = 49
    A = zeros(Float32, n, n, 2)
    A[:, :, 1] = Matrix(Diagonal(vcat(fill(4.0f0, 39), -1.0f0, fill(4.0f0, 9))))
    A[:, :, 2] = Matrix(Diagonal(vcat(fill(4.0f0, 48), -1.0f0)))

    for uplo in ('L', 'U')
        for (name, AT, sync) in backends
            name == "CPU" && continue

            @testset "$name uplo=$uplo" begin
                A_dev = _potrf_to_backend(AT, A)
                _, status_dev = NextLA.potrf!(uplo, A_dev, Val(32), Val(8), Val(4), Val(8); check=false)
                sync(A_dev)

                @test Array(status_dev) == Int32[40, 49]
            end
        end
    end
end
