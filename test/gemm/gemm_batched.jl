using Test
using LinearAlgebra
using Logging

@test NextLA.GEMM_COMPUTE_TYPES == (Float16, Float32, Float64, ComplexF32, ComplexF64, Int32)
@test :gemmEx! in names(NextLA)
@test !(:gemmEx_batched! in names(NextLA))
@test NextLA.gemm! !== LinearAlgebra.mul!
@test parentmodule(NextLA.gemm!) === NextLA

let
    A = Float64[1 2; 3 4]
    B = Float64[5 6; 7 8]
    C = zeros(2, 2)
    @test NextLA.gemm!(C, A, B) === C
    @test C == A * B
    @test NextLA.gemm!(C, A, B; alpha=2.0, beta=3.0) === C
    @test C == 5 .* (A * B)
end
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

            if name in ("CPU", "CUDA", "AMDGPU")
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
                # Ampere or newer is necessary but not sufficient: NextLA's
                # gemm_batched! forwards straight to CUBLAS.gemm_batched!, and
                # CUDA.jl 5.9.2 has no BFloat16 method for it. Check the
                # library, not just the device. See KNOWN_ISSUES.md.
                cublas_has_bf16 = any(m -> occursin("BFloat16", string(m.sig)),
                                      methods(cuda.CUBLAS.gemm_batched!))
                if isdefined(Core, :BFloat16) && cuda.capability(cuda.device()) >= v"8.0" && cublas_has_bf16
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

# A batch member the CPU cannot hand to BLAS is refused by name rather than
# reaching BLAS.gemm! as a MethodError.
@testset "CPU batch refuses members BLAS cannot take" begin
    @test_throws ArgumentError NextLA.gemm_batched!(
        'N', 'N', 1.0, rand(Float64, 6, 6, 2), rand(Float32, 6, 6, 2), 0.0,
        zeros(Float64, 6, 6, 2))
    @test_throws ArgumentError NextLA.gemm_batched!(
        'N', 'N', 1.0f0, [rand(Float16, 3, 3)], [rand(Float16, 3, 3)], 0.0f0,
        [zeros(Float16, 3, 3)])
end

# swap_batch_ptrs! moves a slot on the device; set_batch_ptrs! republishes the
# whole host table to bring one slot across. If the swap is not mirrored into
# the host table, the set silently reverts it and the batch multiplies the
# wrong members -- with no error anywhere.
@testset "a swap survives a later set" begin
    for (name, AT, sync) in backends
        name in ("CUDA", "AMDGPU") || continue
        @testset "$name" begin
            n = 2
            mats = [AT(fill(Float32(k), n, n)) for k in 1:3]
            d = NextLA.BatchPtrDescriptor(mats)
            before = Array(d.ptrs)

            NextLA.swap_batch_ptrs!(d, 1, 2)
            @test Array(d.ptrs)[1] == before[2]
            @test Array(d.ptrs)[2] == before[1]

            # touch an unrelated slot; slots 1 and 2 must stay swapped
            NextLA.set_batch_ptrs!(d, 3, [mats[3]])
            @test Array(d.ptrs)[1] == before[2]
            @test Array(d.ptrs)[2] == before[1]
            @test Array(d.ptrs)[3] == before[3]

            # the two tables agree, which is what makes the next set safe
            @test Array(d.ptrs) == Array(d.host)
        end
    end
end

# The precision policy is a table, and the table disagreed with itself: the
# BFloat16 row rejected a Float32 destination that the identical Float16 row
# allowed, and Int8 was documented as supported while every backend's answer
# was no. These assert the shared table CUDA and AMDGPU read, not a
# computation, so they run anywhere.
@testset "precision signatures the table claims" begin
    sig(TA, TB, TC, W) = NextLA._tensor_core_gemm_supported(TA, TB, TC, W)

    @test sig(Float32, Float32, Float32, Float32)
    @test sig(Float64, Float64, Float64, Float64)
    @test sig(Float16, Float16, Float16, Float16)
    @test sig(Float16, Float16, Float32, Float32)

    # the two that were refused
    # Core.BFloat16 exists from Julia 1.11.
    isdefined(Core, :BFloat16) && @test sig(Core.BFloat16, Core.BFloat16, Float32, Float32)
    @test sig(Int8, Int8, Int32, Int32)

    # and things that must still be refused
    @test !sig(Float32, Float32, Float64, Float64)
    @test !sig(Float64, Float64, Float64, Float32)
end

# Widening, not narrowing, when the destination is the narrow one. Rounding the
# operands to Float16 before multiplying loses roughly an order of magnitude
# more than accumulating wide and rounding once on store.
@testset "a narrow destination does not narrow the operands" begin
    n = 32
    A = rand(Float32, n, n)
    B = rand(Float32, n, n)
    C = zeros(Float16, n, n)
    NextLA.recgemm!(1.0, A, B, 0.0, C)
    ref = Float64.(A) * Float64.(B)
    err = maximum(abs.(Float64.(C) .- ref)) / maximum(abs, ref)
    # Float16 storage alone costs ~1e-3; rounding the operands first costs an
    # order of magnitude more than that.
    @test err < 2e-3
end

# Backend coverage of the batched entry points.
#
# These run wherever the backend exists. On the machine this was integrated on
# only CPU and CUDA do, so the oneAPI and Metal branches have never executed --
# they are written so that the first person with the hardware finds out
# immediately rather than discovering it through a wrong result. Metal refuses
# the vector-of-matrices form outright; what matters there is that the refusal
# comes from its own method, not from the generic CPU fallback handing
# BLAS.gemm! a device matrix.
@testset "batched GEMM stays on its backend" begin
    for (name, AT, sync) in backends
        name == "CPU" && continue
        @testset "$name" begin
            nb, n = 3, 4
            Ah = [rand(Float32, n, n) for _ in 1:nb]
            Bh = [rand(Float32, n, n) for _ in 1:nb]
            Ad = [AT(a) for a in Ah]
            Bd = [AT(b) for b in Bh]
            Cd = [AT(zeros(Float32, n, n)) for _ in 1:nb]

            # the vector-of-matrices form must be served by the backend's own
            # method, not by the generic loop in src/gemm/gemm_batched.jl
            # occursin, not ==: _method_file returns an absolute path and the
            # expected value is the repo-relative suffix, which is the idiom the
            # rest of this file already uses.
            @test occursin(_expected_gemm_batched_file(name),
                           _method_file(NextLA.gemm_batched!, 'N', 'N', 1.0f0,
                                        Ad, Bd, 0.0f0, Cd))

            # and the same shape through gemmEx_batched!, which reaches
            # gemm_batched! via _try_same_type_batched! for a same-type batch
            Cd2 = [AT(zeros(Float32, n, n)) for _ in 1:nb]
            if name == "Metal"
                @test_throws ArgumentError NextLA.gemm_batched!('N', 'N', 1.0f0, Ad, Bd, 0.0f0, Cd)
                @test_throws ArgumentError NextLA.gemmEx_batched!('N', 'N', 1.0f0, Ad, Bd, 0.0f0, Cd2)
            else
                NextLA.gemm_batched!('N', 'N', 1.0f0, Ad, Bd, 0.0f0, Cd)
                sync(Cd[1])
                for i in 1:nb
                    @test Array(Cd[i]) ≈ Ah[i] * Bh[i] rtol=1e-5
                end

                NextLA.gemmEx_batched!('N', 'N', 1.0f0, Ad, Bd, 0.0f0, Cd2)
                sync(Cd2[1])
                for i in 1:nb
                    @test Array(Cd2[i]) ≈ Ah[i] * Bh[i] rtol=1e-5
                end
            end

            # an invalid compute type is diagnosed rather than accepted; oneAPI
            # and Metal skipped this check while CUDA and AMDGPU made it
            @test_throws ArgumentError NextLA.gemmEx_batched!(
                'N', 'N', 1.0f0, Ad, Bd, 0.0f0, Cd2; compute_type=Int8)
        end
    end
end
