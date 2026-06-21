using Test
using LinearAlgebra
using Logging

@test NextLA.GEMM_COMPUTE_TYPES == (Float16, Float32, Float64, ComplexF32, ComplexF64, Int32)
@test !(:gemmEx! in names(NextLA))
@test !(:gemmEx_batched! in names(NextLA))
@test NextLA.default_compute_type(Float16(1), Float16[1 2; 3 4], Float16[1 0; 0 1], Float16(0), Float16[0 0; 0 0]) == Float16

_apply_transpose(A::AbstractMatrix, trans::Char) =
    trans == 'N' ? A : trans == 'T' ? transpose(A) : trans == 'C' ? adjoint(A) :
    throw(ArgumentError("Unsupported transpose flag `$trans`"))

@testset "batched gemm" begin
    # tiny fixture so every backend runs the same arithmetic path cheaply.
    transA_batch = 'N'
    transB_batch = 'N'
    alpha_batch = Float32(2)
    beta_batch = Float32(0.5)
    A_batch = [Float32[1 2; 3 4], Float32[1 0; 2 1]]
    B_batch = [Float32[0 1; 1 0], Float32[1 2; 0 1]]
    C_batch = [fill(Float32(1), 2, 2), fill(Float32(-2), 2, 2)]
    A_batch3 = cat(A_batch..., dims = 3)
    B_batch3 = cat(B_batch..., dims = 3)
    C_batch3 = cat(C_batch..., dims = 3)

    # reuse the pointer-batch reference for the equivalent strided layout.
    expected_batch = [
        alpha_batch * _apply_transpose(A_batch[i], transA_batch) *
        _apply_transpose(B_batch[i], transB_batch) +
        beta_batch * C_batch[i]
        for i in eachindex(A_batch)
    ]

    expected_batch3 = cat(expected_batch..., dims = 3)

    A_single = Float32[1 2; 3 4]
    B_single = Float32[0 1; 1 0]
    C_single = fill(Float32(0.25), 2, 2)
    expected_single = alpha_batch * A_single * B_single + beta_batch * C_single

    for (name, AT, sync) in backends
        @testset "$name dispatch" begin
            A_single_dev = _to_backend(AT, copy(A_single))
            B_single_dev = _to_backend(AT, copy(B_single))
            C_single_dev = _to_backend(AT, copy(C_single))

            if name in ("CUDA", "AMDGPU")
                NextLA.gemmEx!(transA_batch, transB_batch, alpha_batch, A_single_dev, B_single_dev, beta_batch, C_single_dev)
                sync(C_single_dev)
                @test Array(C_single_dev) ≈ expected_single
            else
                # unsupported backends should fail visibly
                @test_throws ArgumentError NextLA.gemmEx!(
                    transA_batch, transB_batch, alpha_batch, A_single_dev, B_single_dev, beta_batch, C_single_dev
                )
            end

            A_batch_dev = _to_backend(AT, deepcopy(A_batch))
            B_batch_dev = _to_backend(AT, deepcopy(B_batch))
            C_batch_dev = _to_backend(AT, deepcopy(C_batch))

            # check that GPU backends do not quietly drop to the generic CPU loop.
            @test occursin(_expected_gemm_batched_file(name), _method_file(NextLA.gemm_batched!, transA_batch, transB_batch, alpha_batch, A_batch_dev, B_batch_dev, beta_batch, C_batch_dev))
            @test_logs min_level=Logging.Warn NextLA.gemm_batched!(
                transA_batch, transB_batch, alpha_batch, A_batch_dev, B_batch_dev, beta_batch, C_batch_dev
            )
            sync(first(C_batch_dev))

            for i in eachindex(expected_batch)
                @test Array(C_batch_dev[i]) ≈ expected_batch[i]
            end

            A_batch_ex_dev = _to_backend(AT, deepcopy(A_batch))
            B_batch_ex_dev = _to_backend(AT, deepcopy(B_batch))
            C_batch_ex_dev = _to_backend(AT, deepcopy(C_batch))

            if name in ("CUDA", "AMDGPU")
                NextLA.gemmEx_batched!(transA_batch, transB_batch, alpha_batch, A_batch_ex_dev, B_batch_ex_dev, beta_batch, C_batch_ex_dev)
                sync(first(C_batch_ex_dev))

                for i in eachindex(expected_batch)
                    @test Array(C_batch_ex_dev[i]) ≈ expected_batch[i]
                end
            else
                @test_throws ArgumentError NextLA.gemmEx_batched!(
                    transA_batch, transB_batch, alpha_batch, A_batch_ex_dev, B_batch_ex_dev, beta_batch, C_batch_ex_dev
                )
            end

            A_batch3_dev = _to_backend(AT, copy(A_batch3))
            B_batch3_dev = _to_backend(AT, copy(B_batch3))
            C_batch3_dev = _to_backend(AT, copy(C_batch3))

            # strided 3d batches should resolve to the backend-specific implementation.
            @test occursin(_expected_gemm_batched_file(name), _method_file(NextLA.gemm_batched!, transA_batch, transB_batch, alpha_batch, A_batch3_dev, B_batch3_dev, beta_batch, C_batch3_dev))
            @test_logs min_level=Logging.Warn NextLA.gemm_batched!(
                transA_batch, transB_batch, alpha_batch, A_batch3_dev, B_batch3_dev, beta_batch, C_batch3_dev
            )
            sync(C_batch3_dev)
            @test Array(C_batch3_dev) ≈ expected_batch3

            A_batch3_ex_dev = _to_backend(AT, copy(A_batch3))
            B_batch3_ex_dev = _to_backend(AT, copy(B_batch3))
            C_batch3_ex_dev = _to_backend(AT, copy(C_batch3))

            if name in ("CUDA", "AMDGPU")
                NextLA.gemmEx_batched!(transA_batch, transB_batch, alpha_batch, A_batch3_ex_dev, B_batch3_ex_dev, beta_batch, C_batch3_ex_dev)
                sync(C_batch3_ex_dev)
                @test Array(C_batch3_ex_dev) ≈ expected_batch3
            else
                # unsupported backends should fail visibly
                @test_throws ArgumentError NextLA.gemmEx_batched!(
                    transA_batch, transB_batch, alpha_batch, A_batch3_ex_dev, B_batch3_ex_dev, beta_batch, C_batch3_ex_dev
                )
            end
        end
    end
end
