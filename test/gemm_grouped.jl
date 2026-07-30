using Test

@testset "grouped GEMM" begin
    @test !NextLA.supports_grouped_gemm(KernelAbstractions.CPU())
    cpu_task = NextLA.GroupedGemmTask('N', 'N', 1.0f0, ones(Float32, 2, 1),
                                       ones(Float32, 1, 3), 0.0f0, zeros(Float32, 2, 3))
    @test_throws ArgumentError NextLA.precision_gemm_grouped!(
        [cpu_task], NextLA.GEMMCompute{Float32}())

    for (name, AT, sync) in backends
        name == "CUDA" || continue
        A1 = _to_backend(AT, Float32[1 2; 3 4])
        B1 = _to_backend(AT, Float32[2 0 1; 1 3 4])
        C1 = _to_backend(AT, fill(0.5f0, 2, 3))
        C3 = _to_backend(AT, fill(0.25f0, 2, 3))
        A2 = _to_backend(AT, Float32[1 2 3])
        B2 = _to_backend(AT, reshape(Float32[2, 4, 1], 3, 1))
        C2 = _to_backend(AT, reshape(Float32[3], 1, 1))
        expected1 = 2f0 .* (Float32[1 2; 3 4] * Float32[2 0 1; 1 3 4]) .+ 0.5f0 .* fill(0.5f0, 2, 3)
        expected3 = Float32[1 2; 3 4] * Float32[2 0 1; 1 3 4]
        expected2 = 3f0 .* (Float32[1 2 3] * Float32[2; 4; 1]) .+ 0.25f0 .* reshape(Float32[3], 1, 1)
        tasks = [
            NextLA.GroupedGemmTask('N', 'N', 2f0, A1, B1, 0.5f0, C1),
            NextLA.GroupedGemmTask('N', 'N', 3f0, A2, B2, 0.25f0, C2),
            # Same shape as the first task but a different alpha/beta pair:
            # this must be a second cuBLAS group.
            NextLA.GroupedGemmTask('N', 'N', 1f0, A1, B1, 0f0, C3),
        ]
        NextLA.precision_gemm_grouped!(tasks, NextLA.GEMMCompute{Float32}())
        sync(C1)
        @test Array(C1) ≈ expected1
        @test Array(C2) ≈ expected2
        @test Array(C3) ≈ expected3

        Ah = _to_backend(AT, Float16[1 2; 3 4])
        Bh = _to_backend(AT, Float16[2 0 1; 1 3 4])
        Ch = _to_backend(AT, fill(Float16(0.5), 2, 3))
        mixed = [NextLA.GroupedGemmTask('N', 'N', 2f0, Ah, Bh, 0.5f0, Ch)]
        NextLA.precision_gemm_grouped!(mixed, NextLA.GEMMCompute{Float32}())
        sync(Ch)
        @test Float32.(Array(Ch)) ≈ 2f0 .* (Float32.(Float16[1 2; 3 4]) *
                                             Float32.(Float16[2 0 1; 1 3 4])) .+ 0.25f0

        # Offset by one Float32 element. cuBLAS grouped GEMM accepts unaligned
        # pointers (measured correct on SM75 and SM90), so this member stays in
        # the group rather than being split out to an ordinary GEMMEx call.
        Astore = _to_backend(AT, Float32[0 0; 1 2; 3 4])
        Aunsafe = view(Astore, 2:3, :)
        Cunsafe = _to_backend(AT, zeros(Float32, 2, 3))
        unsafe = [NextLA.GroupedGemmTask('N', 'N', 1f0, Aunsafe, B1, 0f0, Cunsafe)]
        NextLA.precision_gemm_grouped!(unsafe, NextLA.GEMMCompute{Float32}())
        sync(Cunsafe)
        @test Array(Cunsafe) ≈ Float32[1 2; 3 4] * Float32[2 0 1; 1 3 4]
        prepared_unsafe = NextLA.prepare_precision_gemm_grouped(
            unsafe, NextLA.GEMMCompute{Float32}())
        @test prepared_unsafe isa NextLA.AbstractPreparedGroupedGemm
        fill!(Cunsafe, 0f0)
        # Prepared descriptors take host-resident scalar arrays, so the caller
        # owns the pointer-mode window (the execution paths hoist it out of
        # their run loops rather than paying it per submission).
        NextLA._with_grouped_host_pointer_mode(NextLA.get_backend(Cunsafe)) do
            NextLA.precision_gemm_grouped_prepared!(prepared_unsafe)
        end
        sync(Cunsafe)
        @test Array(Cunsafe) ≈ Float32[1 2; 3 4] * Float32[2 0 1; 1 3 4]
        NextLA.destroy_prepared_grouped_gemm!(prepared_unsafe)
    end
end
