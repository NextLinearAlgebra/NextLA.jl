using Test
using LinearAlgebra

function _syrk_expected(uplo::Char, trans::Char, alpha, A, beta, C)
    prod = trans == 'N' ? A * transpose(A) : transpose(A) * A
    return alpha * prod + beta * C
end

function _triangle_matches(uplo::Char, got::AbstractMatrix, expected::AbstractMatrix)
    n = size(got, 1)
    if uplo == 'U'
        for j in 1:n, i in 1:j
            @inbounds isapprox(got[i, j], expected[i, j]) || return false
        end
    else
        for j in 1:n, i in j:n
            @inbounds isapprox(got[i, j], expected[i, j]) || return false
        end
    end
    return true
end

@testset "syrk dispatch" begin
    uplo = 'U'
    trans = 'N'
    alpha = Float32(1.5)
    beta = Float32(0.25)

    A_single = Float32[1 2 3; 4 5 6]
    C_single = Float32[2 1; 3 4]
    expected_single = _syrk_expected(uplo, trans, alpha, A_single, beta, C_single)

    A_batch = [Float32[1 2 0; 3 4 1], Float32[2 1 3; 0 1 2]]
    C_batch = [Float32[1 0; 0 1], Float32[2 1; 1 2]]
    expected_batch = [
        _syrk_expected(uplo, trans, alpha, A_batch[i], beta, C_batch[i])
        for i in eachindex(A_batch)
    ]

    A_batch3 = cat(A_batch..., dims = 3)
    C_batch3 = cat(C_batch..., dims = 3)
    expected_batch3 = cat(expected_batch..., dims = 3)

    for (name, AT, sync) in backends
        @testset "$name dispatch" begin
            A_single_dev = _to_backend(AT, copy(A_single))
            C_single_dev = _to_backend(AT, copy(C_single))

            NextLA.syrk!(uplo, trans, alpha, A_single_dev, beta, C_single_dev)
            sync(C_single_dev)
            @test _triangle_matches(uplo, Array(C_single_dev), expected_single)

            A_batch_dev = _to_backend(AT, deepcopy(A_batch))
            C_batch_dev = _to_backend(AT, deepcopy(C_batch))

            if name in ("CPU", "CUDA", "oneAPI", "Metal")
                @test_logs (:warn, r"syrk_batched! falling back to batched gemm!") NextLA.syrk_batched!(
                    uplo, trans, alpha, A_batch_dev, beta, C_batch_dev
                )
            else
                NextLA.syrk_batched!(uplo, trans, alpha, A_batch_dev, beta, C_batch_dev)
            end
            sync(first(C_batch_dev))

            for i in eachindex(expected_batch)
                @test _triangle_matches(uplo, Array(C_batch_dev[i]), expected_batch[i])
            end

            A_batch3_dev = _to_backend(AT, copy(A_batch3))
            C_batch3_dev = _to_backend(AT, copy(C_batch3))

            if name in ("CPU", "CUDA", "Metal")
                @test_logs (:warn, r"syrk_batched! falling back to batched gemm!") NextLA.syrk_batched!(
                    uplo, trans, alpha, A_batch3_dev, beta, C_batch3_dev
                )
            else
                NextLA.syrk_batched!(uplo, trans, alpha, A_batch3_dev, beta, C_batch3_dev)
            end
            sync(C_batch3_dev)
            for i in axes(expected_batch3, 3)
                @test _triangle_matches(uplo, Array(@view(C_batch3_dev[:, :, i])), @view(expected_batch3[:, :, i]))
            end
        end
    end
end
