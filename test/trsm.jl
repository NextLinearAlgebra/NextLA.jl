const TRSM_KERNEL_CASES = (
    ("LeftLowerTRSM!", NextLA.LeftLowerTRSM!, 'L', 'L'),
    ("LeftUpperTRSM!", NextLA.LeftUpperTRSM!, 'L', 'U'),
    ("RightLowerTRSM!", NextLA.RightLowerTRSM!, 'R', 'L'),
    ("RightUpperTRSM!", NextLA.RightUpperTRSM!, 'R', 'U'),
)

_trsm_tol(::Type{Float32}) = 1f-5
_trsm_tol(::Type{Float64}) = 1e-11

function _trsm_matrix(::Type{T}, n::Int, uplo::Char) where {T}
    M = rand(T, n, n) .+ one(T)
    A = uplo == 'L' ? Matrix(LowerTriangular(M)) : Matrix(UpperTriangular(M))
    A .+= Diagonal(fill(T(10), n))
    return A
end

function _trsm_inputs(::Type{T}, side::Char, uplo::Char, n::Int, m::Int; batch::Int=1) where {T}
    if batch == 1
        A = _trsm_matrix(T, n, uplo)
        B = side == 'L' ? rand(T, n, m) .+ one(T) : rand(T, m, n) .+ one(T)
        return A, B
    end

    A = Array{T,3}(undef, n, n, batch)
    B = side == 'L' ? Array{T,3}(undef, n, m, batch) : Array{T,3}(undef, m, n, batch)
    for bid in 1:batch
        A[:, :, bid] = _trsm_matrix(T, n, uplo)
        B[:, :, bid] = side == 'L' ? rand(T, n, m) .+ one(T) : rand(T, m, n) .+ one(T)
    end
    return A, B
end

function _trsm_reference!(side::Char, uplo::Char, A::AbstractMatrix, B::AbstractMatrix)
    BLAS.trsm!(side, uplo, 'N', 'N', one(eltype(A)), A, B)
    return B
end

function _trsm_reference!(side::Char, uplo::Char, A::AbstractArray{<:Any,3}, B::AbstractArray{<:Any,3})
    for bid in axes(A, 3)
        BLAS.trsm!(side, uplo, 'N', 'N', one(eltype(A)), @view(A[:, :, bid]), @view(B[:, :, bid]))
    end
    return B
end

function _trsm_reference!(side::Char,
                          uplo::Char,
                          A::AbstractVector{<:AbstractMatrix},
                          B::AbstractVector{<:AbstractMatrix})
    length(A) == length(B) || throw(DimensionMismatch("trsm reference batches must have matching lengths"))
    for bid in eachindex(A, B)
        BLAS.trsm!(side, uplo, 'N', 'N', one(eltype(A[bid])), A[bid], B[bid])
    end
    return B
end

function _test_trsm_result(actual, expected, ::Type{T}) where {T}
    @test norm(Array(actual) - expected) / norm(expected) < _trsm_tol(T)
end

function _test_trsm_result(actual::AbstractVector{<:AbstractMatrix},
                           expected::AbstractVector{<:AbstractMatrix},
                           ::Type{T}) where {T}
    for bid in eachindex(actual, expected)
        @test norm(Array(actual[bid]) - Array(expected[bid])) / norm(Array(expected[bid])) < _trsm_tol(T)
    end
end

@testset "TRSM kernels" begin
    trsm_backends = [("CPU", Array, _ -> nothing)]

    for (backend_name, ArrayType, synchronize) in trsm_backends
        @testset "[$backend_name] 2D public kernels" begin
            for T in (Float32, Float64)
                for (label, kernel!, side, uplo) in TRSM_KERNEL_CASES
                    @testset "$label $T n=$n m=$m" for n in [16, 32, 128, 256, 512],
                                                        m in [1, 8, 64]
                        A, B = _trsm_inputs(T, side, uplo, n, m)
                        B_ref = copy(B)
                        _trsm_reference!(side, uplo, copy(A), B_ref)

                        A_dev = _to_backend(ArrayType, A)
                        B_dev = _to_backend(ArrayType, B)
                        kernel!(A_dev, B_dev)
                        synchronize(B_dev)

                        _test_trsm_result(B_dev, B_ref, T)
                    end
                end
            end
        end

        @testset "[$backend_name] trsm transpose n=$n m=$m" for n in [16, 32, 256],
                                                                    m in [1, 8]
            A = _trsm_matrix(Float32, n, 'L')
            B = rand(Float32, n, m) .+ 1
            Ac, Bc = copy(A), copy(B)
            B_dev = _to_backend(ArrayType, B)

            NextLA.trsm!('L', 'L', 'T', 'N', _to_backend(ArrayType, A), B_dev)
            synchronize(B_dev)
            BLAS.trsm!('L', 'L', 'T', 'N', 1f0, Ac, Bc)
            _test_trsm_result(B_dev, Bc, Float32)
        end

        @testset "[$backend_name] trsm adjoint n=$n m=$m" for n in [16, 32, 256],
                                                                  m in [1, 8]
            A = Matrix(UpperTriangular(complex.(rand(Float32, n, n) .+ 1, rand(Float32, n, n))))
            A .+= Diagonal(fill(ComplexF32(10, 0), n))
            B = complex.(rand(Float32, n, m) .+ 1, rand(Float32, n, m))
            Ac, Bc = copy(A), copy(B)
            B_dev = _to_backend(ArrayType, B)

            NextLA.trsm!('L', 'U', 'C', 'N', _to_backend(ArrayType, A), B_dev)
            synchronize(B_dev)
            BLAS.trsm!('L', 'U', 'C', 'N', ComplexF32(1, 0), Ac, Bc)
            @test norm(Array(B_dev) - Bc) / norm(Bc) < 1f-5
        end
    end
end

@testset "batched TRSM" begin
    batch = 3
    trsm_backends = available_gpu_backends()
    if isempty(trsm_backends)
        @test_skip "No GPU backends available"
    end

    for (backend_name, ArrayType, synchronize) in trsm_backends
        @testset "[$backend_name] 3D trsm_batched!" begin
            for T in (Float32, Float64)
                for (label, _, side, uplo) in TRSM_KERNEL_CASES
                    @testset "$label $T n=$n m=$m" for n in [16, 32, 128],
                                                        m in [1, 8]
                        A, B = _trsm_inputs(T, side, uplo, n, m; batch)
                        B_ref = copy(B)
                        _trsm_reference!(side, uplo, copy(A), B_ref)

                        A_dev = _to_backend(ArrayType, A)
                        B_dev = _to_backend(ArrayType, B)
                        @test occursin(_expected_trsm_batched_file(backend_name), _method_file(NextLA.trsm_batched!, side, uplo, 'N', 'N', A_dev, B_dev))
                        if backend_name == "Metal"
                            @test_throws ArgumentError NextLA.trsm_batched!(side, uplo, 'N', 'N', A_dev, B_dev)
                            continue
                        end
                        NextLA.trsm_batched!(side, uplo, 'N', 'N', A_dev, B_dev)
                        synchronize(B_dev)

                        _test_trsm_result(B_dev, B_ref, T)
                    end
                end
            end
        end
    end
end

@testset "pointer batched TRSM" begin
    batch = 3

    for (backend_name, ArrayType, synchronize) in backends
        @testset "[$backend_name] vector-of-views trsm_batched!" begin
            for T in (Float32, Float64)
                for (label, _, side, uplo) in TRSM_KERNEL_CASES
                    @testset "$label $T n=$n m=$m" for n in [16, 32, 128],
                                                        m in [1, 8]
                        A_host, B_host = _trsm_inputs(T, side, uplo, n, m; batch)
                        A_batch = [@view A_host[:, :, bid] for bid in 1:batch]
                        B_batch = [@view B_host[:, :, bid] for bid in 1:batch]
                        B_ref = copy.(B_batch)
                        _trsm_reference!(side, uplo, copy.(A_batch), B_ref)

                        A_dev = _to_backend(ArrayType, A_batch)
                        B_dev = _to_backend(ArrayType, B_batch)
                        @test occursin(_expected_trsm_batched_file(backend_name), _method_file(NextLA.trsm_batched!, side, uplo, 'N', 'N', A_dev, B_dev))
                        if backend_name == "Metal"
                            @test_throws ArgumentError NextLA.trsm_batched!(side, uplo, 'N', 'N', A_dev, B_dev)
                            continue
                        end
                        NextLA.trsm_batched!(side, uplo, 'N', 'N', A_dev, B_dev)
                        synchronize(first(B_dev))

                        _test_trsm_result(B_dev, B_ref, T)
                    end
                end
            end
        end

        @testset "[$backend_name] mixed-size pointer batch" begin
            side, uplo = 'L', 'L'
            A_batch = [
                _trsm_matrix(Float32, 16, uplo),
                _trsm_matrix(Float32, 20, uplo),
                _trsm_matrix(Float32, 24, uplo),
            ]
            B_batch = [
                rand(Float32, 16, 3) .+ 1,
                rand(Float32, 20, 3) .+ 1,
                rand(Float32, 24, 3) .+ 1,
            ]
            B_ref = copy.(B_batch)
            _trsm_reference!(side, uplo, copy.(A_batch), B_ref)

            A_dev = _to_backend(ArrayType, A_batch)
            B_dev = _to_backend(ArrayType, B_batch)
            @test occursin(_expected_trsm_batched_file(backend_name), _method_file(NextLA.trsm_batched!, side, uplo, 'N', 'N', A_dev, B_dev))
            if backend_name == "Metal"
                @test_throws ArgumentError NextLA.trsm_batched!(side, uplo, 'N', 'N', A_dev, B_dev)
                continue
            end
            NextLA.trsm_batched!(side, uplo, 'N', 'N', A_dev, B_dev)
            synchronize(first(B_dev))

            _test_trsm_result(B_dev, B_ref, Float32)
        end
    end
end
