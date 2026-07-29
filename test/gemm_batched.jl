using Test
using LinearAlgebra
using Logging

@test NextLA.GEMM_COMPUTE_TYPES == (Float16, Float32, Float64, ComplexF32, ComplexF64, Int32)
@test !(:gemmEx! in names(NextLA))
@test !(:gemmEx_batched! in names(NextLA))
@test NextLA.gemm! === LinearAlgebra.mul!
@test NextLA.default_compute_type(Float16(1), Float16[1 2; 3 4], Float16[1 0; 0 1], Float16(0), Float16[0 0; 0 0]) == Float16
@test NextLA.gemm_compute_type(NextLA.TLRmodule.default_gemm_compute_mode(Float16)) == Float32

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
    A_batch_mixed = [Float16[1 2; 3 4], Float16[1 0; 2 1]]
    B_batch_mixed = [Float16[0 1; 1 0], Float16[1 2; 0 1]]
    C_batch_mixed = [zeros(Float32, 2, 2), zeros(Float32, 2, 2)]
    C_batch_half = [zeros(Float16, 2, 2), zeros(Float16, 2, 2)]

    # reuse the pointer-batch reference for the equivalent strided layout.
    expected_batch = [
        alpha_batch * _apply_transpose(A_batch[i], transA_batch) *
        _apply_transpose(B_batch[i], transB_batch) +
        beta_batch * C_batch[i]
        for i in eachindex(A_batch)
    ]

    expected_batch3 = cat(expected_batch..., dims = 3)
    expected_batch_mixed = [
        Float32(alpha_batch) .* Float32.(_apply_transpose(A_batch_mixed[i], transA_batch) * _apply_transpose(B_batch_mixed[i], transB_batch))
        for i in eachindex(A_batch_mixed)
    ]
    expected_batch3_mixed = cat(expected_batch_mixed..., dims = 3)
    expected_batch_half = [Float16.(Ci) for Ci in expected_batch_mixed]
    expected_batch3_half = cat(expected_batch_half..., dims = 3)

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

                A_single_mixed = _to_backend(AT, Float16[1 2; 3 4])
                B_single_mixed = _to_backend(AT, Float16[0 1; 1 0])
                C_single_mixed = _to_backend(AT, zeros(Float32, 2, 2))
                expected_single_mixed = Float32(alpha_batch) .* Float32.(Float16[1 2; 3 4] * Float16[0 1; 1 0])
                NextLA.gemmEx!(transA_batch, transB_batch, alpha_batch, A_single_mixed, B_single_mixed, 0.0f0, C_single_mixed)
                sync(C_single_mixed)
                @test Array(C_single_mixed) ≈ expected_single_mixed

                C_single_half = _to_backend(AT, zeros(Float16, 2, 2))
                NextLA.gemmEx!(
                    transA_batch, transB_batch, alpha_batch,
                    A_single_mixed, B_single_mixed, 0.0f0, C_single_half;
                    compute_type=Float32,
                )
                sync(C_single_half)
                @test Array(C_single_half) ≈ Float16.(expected_single_mixed)
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

            if name == "CUDA"
                cuda = Base.require(Main, :CUDA)
                if cuda.capability(cuda.device()) >= v"8.0"
                    Ab = _to_backend(AT, reshape(Core.BFloat16[1 2; 3 4], 2, 2, 1))
                    Bb = _to_backend(AT, reshape(Core.BFloat16[1 0; 0 1], 2, 2, 1))
                    Cb = _to_backend(AT, zeros(Core.BFloat16, 2, 2, 1))
                    NextLA.gemm_batched!(
                        'N', 'N', one(Core.BFloat16), [view(Ab, :, :, 1)],
                        [view(Bb, :, :, 1)], zero(Core.BFloat16),
                        [view(Cb, :, :, 1)])
                    sync(Cb)
                    @test Float32.(Array(Cb[:, :, 1])) ≈ Float32[1 2; 3 4]
                end
            end

            A_batch_ex_dev = _to_backend(AT, deepcopy(A_batch))
            B_batch_ex_dev = _to_backend(AT, deepcopy(B_batch))
            C_batch_ex_dev = _to_backend(AT, deepcopy(C_batch))

            NextLA.gemmEx_batched!(transA_batch, transB_batch, alpha_batch, A_batch_ex_dev, B_batch_ex_dev, beta_batch, C_batch_ex_dev)
            sync(first(C_batch_ex_dev))

            for i in eachindex(expected_batch)
                @test Array(C_batch_ex_dev[i]) ≈ expected_batch[i]
            end

            if name in ("CUDA", "AMDGPU")
                A_batch_mixed_dev = _to_backend(AT, deepcopy(A_batch_mixed))
                B_batch_mixed_dev = _to_backend(AT, deepcopy(B_batch_mixed))
                C_batch_mixed_dev = _to_backend(AT, deepcopy(C_batch_mixed))

                NextLA.gemmEx_batched!(
                    transA_batch, transB_batch, alpha_batch,
                    A_batch_mixed_dev, B_batch_mixed_dev, 0.0f0, C_batch_mixed_dev,
                )
                sync(first(C_batch_mixed_dev))

                for i in eachindex(expected_batch_mixed)
                    @test Array(C_batch_mixed_dev[i]) ≈ expected_batch_mixed[i]
                end
                C_batch_half_dev = _to_backend(AT, deepcopy(C_batch_half))
                NextLA.gemmEx_batched!(
                    transA_batch, transB_batch, alpha_batch,
                    A_batch_mixed_dev, B_batch_mixed_dev, 0.0f0, C_batch_half_dev;
                    compute_type=Float32,
                )
                sync(first(C_batch_half_dev))
                for i in eachindex(expected_batch_half)
                    @test Array(C_batch_half_dev[i]) ≈ expected_batch_half[i]
                end
            else
                @test_throws ArgumentError NextLA.gemmEx_batched!(
                    transA_batch, transB_batch, alpha_batch,
                    _to_backend(AT, deepcopy(A_batch_mixed)),
                    _to_backend(AT, deepcopy(B_batch_mixed)),
                    0.0f0,
                    _to_backend(AT, deepcopy(C_batch_mixed)),
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

            NextLA.gemmEx_batched!(transA_batch, transB_batch, alpha_batch, A_batch3_ex_dev, B_batch3_ex_dev, beta_batch, C_batch3_ex_dev)
            sync(C_batch3_ex_dev)
            @test Array(C_batch3_ex_dev) ≈ expected_batch3

            if name in ("CUDA", "AMDGPU")
                A_batch3_mixed = cat(A_batch_mixed..., dims = 3)
                B_batch3_mixed = cat(B_batch_mixed..., dims = 3)
                C_batch3_mixed = zeros(Float32, 2, 2, 2)
                A_batch3_mixed_dev = _to_backend(AT, copy(A_batch3_mixed))
                B_batch3_mixed_dev = _to_backend(AT, copy(B_batch3_mixed))
                C_batch3_mixed_dev = _to_backend(AT, copy(C_batch3_mixed))

                NextLA.gemmEx_batched!(transA_batch, transB_batch, alpha_batch, A_batch3_mixed_dev, B_batch3_mixed_dev, 0.0f0, C_batch3_mixed_dev)
                sync(C_batch3_mixed_dev)
                @test Array(C_batch3_mixed_dev) ≈ expected_batch3_mixed

                C_batch3_half_dev = _to_backend(AT, zeros(Float16, size(C_batch3_mixed)))
                NextLA.gemmEx_batched!(
                    transA_batch, transB_batch, alpha_batch,
                    A_batch3_mixed_dev, B_batch3_mixed_dev, 0.0f0, C_batch3_half_dev;
                    compute_type=Float32,
                )
                sync(C_batch3_half_dev)
                @test Array(C_batch3_half_dev) ≈ expected_batch3_half
            else
                # unsupported backends should fail visibly
                @test_throws ArgumentError NextLA.gemmEx_batched!(
                    transA_batch, transB_batch, alpha_batch,
                    _to_backend(AT, cat(A_batch_mixed..., dims = 3)),
                    _to_backend(AT, cat(B_batch_mixed..., dims = 3)),
                    0.0f0,
                    _to_backend(AT, zeros(Float32, 2, 2, 2)),
                )
            end

            if name in ("CUDA", "AMDGPU")
                Aptrs = _device_pointer_batch(name, A_batch_ex_dev)
                Bptrs = _device_pointer_batch(name, B_batch_ex_dev)
                C_ptr_batch_dev = _to_backend(AT, deepcopy(C_batch))
                Cptrs = _device_pointer_batch(name, C_ptr_batch_dev)

                NextLA.gemm_batched!(
                    transA_batch, transB_batch, alpha_batch,
                    Aptrs, A_batch_ex_dev[1], Bptrs, B_batch_ex_dev[1], beta_batch, Cptrs, C_ptr_batch_dev[1], length(C_ptr_batch_dev),
                )
                sync(first(C_ptr_batch_dev))
                for i in eachindex(expected_batch)
                    @test Array(C_ptr_batch_dev[i]) ≈ expected_batch[i]
                end

                C_ptr_ex_batch_dev = _to_backend(AT, deepcopy(C_batch))
                Cptrs_ex = _device_pointer_batch(name, C_ptr_ex_batch_dev)
                NextLA.gemmEx_batched!(
                    transA_batch, transB_batch, alpha_batch,
                    Aptrs, A_batch_ex_dev[1], Bptrs, B_batch_ex_dev[1], beta_batch, Cptrs_ex, C_ptr_ex_batch_dev[1], length(C_ptr_ex_batch_dev),
                )
                sync(first(C_ptr_ex_batch_dev))
                for i in eachindex(expected_batch)
                    @test Array(C_ptr_ex_batch_dev[i]) ≈ expected_batch[i]
                end

                A_batch_mixed_dev = _to_backend(AT, deepcopy(A_batch_mixed))
                B_batch_mixed_dev = _to_backend(AT, deepcopy(B_batch_mixed))
                Aptrs_mixed = _device_pointer_batch(name, A_batch_mixed_dev)
                Bptrs_mixed = _device_pointer_batch(name, B_batch_mixed_dev)
                C_ptr_mixed_dev = _to_backend(AT, deepcopy(C_batch_mixed))
                Cptrs_mixed = _device_pointer_batch(name, C_ptr_mixed_dev)

                NextLA.gemmEx_batched!(
                    transA_batch, transB_batch, alpha_batch,
                    Aptrs_mixed, A_batch_mixed_dev[1], Bptrs_mixed, B_batch_mixed_dev[1], 0.0f0, Cptrs_mixed, C_ptr_mixed_dev[1], length(C_ptr_mixed_dev),
                )
                sync(first(C_ptr_mixed_dev))
                for i in eachindex(expected_batch_mixed)
                    @test Array(C_ptr_mixed_dev[i]) ≈ expected_batch_mixed[i]
                end
                C_ptr_half_dev = _to_backend(AT, deepcopy(C_batch_half))
                Cptrs_half = _device_pointer_batch(name, C_ptr_half_dev)
                NextLA.gemmEx_batched!(
                    transA_batch, transB_batch, alpha_batch,
                    Aptrs_mixed, A_batch_mixed_dev[1], Bptrs_mixed, B_batch_mixed_dev[1],
                    0.0f0, Cptrs_half, C_ptr_half_dev[1], length(C_ptr_half_dev);
                    compute_type=Float32,
                )
                sync(first(C_ptr_half_dev))
                for i in eachindex(expected_batch_half)
                    @test Array(C_ptr_half_dev[i]) ≈ expected_batch_half[i]
                end
            end
        end
    end
end

@testset "persistent batch pointer descriptors" begin
    # `BatchPtrDescriptor` exists so a caller issuing the same batched-GEMM
    # shape on many successive calls (one per ARA sampling pass, say) can
    # build the device pointer table once instead of paying a fresh
    # allocate/upload/free on every call, as the `Vector`-of-matrix
    # `gemm_batched!`/`gemmEx_batched!` methods do. These tests check: (1) a
    # descriptor drives a batched GEMM to the same result as the transient
    # pointer-batch path, (2) `swap_batch_ptrs!` moves which address a slot
    # refers to without touching the underlying data, single-slot and block
    # forms, and (3) CPU (and any backend without a `_build_batch_ptrs`
    # method) fails to construct one, since CPU batched GEMM never needs a
    # pointer array.
    transA_d, transB_d = 'N', 'N'
    alpha_d, beta_d = Float32(2), Float32(0.5)
    mode_d = NextLA.GEMMCompute{Float32}()
    A_batch_d = [Float32[1 2; 3 4], Float32[1 0; 2 1], Float32[2 1; 0 1]]
    B_batch_d = [Float32[0 1; 1 0], Float32[1 2; 0 1], Float32[1 1; 1 0]]
    C_batch_d = [fill(Float32(1), 2, 2), fill(Float32(-2), 2, 2), fill(Float32(0.5), 2, 2)]
    expected_d = [
        alpha_d * A_batch_d[i] * B_batch_d[i] + beta_d * C_batch_d[i]
        for i in eachindex(A_batch_d)
    ]

    for (name, AT, sync) in backends
        @testset "$name" begin
            A_dev = _to_backend(AT, deepcopy(A_batch_d))
            B_dev = _to_backend(AT, deepcopy(B_batch_d))

            if name in ("CUDA", "AMDGPU")
                Ad = BatchPtrDescriptor(A_dev)
                Bd = BatchPtrDescriptor(B_dev)
                @test length(Ad) == length(A_dev) == length(Bd)

                C_dev = _to_backend(AT, deepcopy(C_batch_d))
                Cd = BatchPtrDescriptor(C_dev)
                NextLA.precision_gemm_batched_ptrs!(
                    transA_d, transB_d, alpha_d, Ad, A_dev[1], Bd, B_dev[1],
                    beta_d, Cd, C_dev[1], length(C_dev), mode_d,
                )
                sync(first(C_dev))
                for i in eachindex(expected_d)
                    @test Array(C_dev[i]) ≈ expected_d[i]
                end

                # Single-slot swap: only Cd2 is swapped, not Ad/Bd, so slot k
                # still computes with A[k]/B[k] but accumulates into whatever
                # C address slot k's swapped pointer now targets. Swapping
                # slots 1 and 3 means slot 1 writes alpha*A[1]*B[1] +
                # beta*C_batch_d[3] into C_dev2[3]'s memory (untouched by the
                # swap itself), and slot 3 writes alpha*A[3]*B[3] +
                # beta*C_batch_d[1] into C_dev2[1]'s memory.
                C_dev2 = _to_backend(AT, deepcopy(C_batch_d))
                Cd2 = BatchPtrDescriptor(C_dev2)
                swap_batch_ptrs!(Cd2, 1, 3)
                NextLA.precision_gemm_batched_ptrs!(
                    transA_d, transB_d, alpha_d, Ad, A_dev[1], Bd, B_dev[1],
                    beta_d, Cd2, C_dev2[1], length(C_dev2), mode_d,
                )
                sync(first(C_dev2))
                mix(iAB, iC) = alpha_d * A_batch_d[iAB] * B_batch_d[iAB] +
                               beta_d * C_batch_d[iC]
                @test Array(C_dev2[3]) ≈ mix(1, 3)
                @test Array(C_dev2[2]) ≈ expected_d[2]
                @test Array(C_dev2[1]) ≈ mix(3, 1)

                # Block swap (blocklen=2): descriptor slots [1,2,3,4] holding
                # members [D1,D2,D3,D4] become [D3,D4,D1,D2] after swapping
                # the two-slot blocks at member offsets 1 and 2.
                D_batch = [Float32[1 0; 0 1], Float32[2 0; 0 2],
                          Float32[0 1; 1 0], Float32[1 1; 0 1]]
                D_dev = _to_backend(AT, deepcopy(D_batch))
                Dd = BatchPtrDescriptor(D_dev)
                swap_batch_ptrs!(Dd, 1, 2, 2)
                I_dev = _to_backend(AT, [Float32[1 0; 0 1] for _ in 1:4])
                Id = BatchPtrDescriptor(I_dev)
                Out_dev = _to_backend(AT, [zeros(Float32, 2, 2) for _ in 1:4])
                Outd = BatchPtrDescriptor(Out_dev)
                NextLA.precision_gemm_batched_ptrs!(
                    'N', 'N', one(Float32), Dd, D_dev[1], Id, I_dev[1],
                    zero(Float32), Outd, Out_dev[1], 4, mode_d,
                )
                sync(first(Out_dev))
                @test Array(Out_dev[1]) ≈ D_batch[3]
                @test Array(Out_dev[2]) ≈ D_batch[4]
                @test Array(Out_dev[3]) ≈ D_batch[1]
                @test Array(Out_dev[4]) ≈ D_batch[2]
            else
                @test_throws ArgumentError BatchPtrDescriptor(A_dev)
            end
        end
    end
end
