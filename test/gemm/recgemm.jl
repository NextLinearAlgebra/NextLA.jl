# recgemm! over dense destinations, on the CPU backend.

@testset "RECGEMM [CPU]" begin
    # Complex belongs here: the rest of the library is four types wide, and
    # recgemm! used to throw InexactError on both complex types because the
    # dispatcher steered them into a Float32 accumulation.
    @testset "$T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        rtol = test_rtol(T)

        @testset "dense C matches A*B, n=$n" for n in [4, 8, 16, 32]
            A = rand(T, n, n)
            B = rand(T, n, n)
            C = rand(T, n, n)
            alpha, beta = T(2), T(3)
            expected = alpha * A * B + beta * C

            Cout = copy(C)
            NextLA.recgemm!(alpha, A, B, beta, Cout)
            @test Cout ≈ expected rtol=rtol
        end

        @testset "beta=0 overwrites C" begin
            n = 8
            A = rand(T, n, n)
            B = rand(T, n, n)
            C = fill(T(99), n, n)
            NextLA.recgemm!(one(T), A, B, zero(T), C)
            @test C ≈ A * B rtol=rtol
        end

        @testset "alpha=0 scales C only" begin
            n = 8
            A = rand(T, n, n)
            B = rand(T, n, n)
            C = rand(T, n, n)
            C0 = copy(C)
            NextLA.recgemm!(zero(T), A, B, T(2), C)
            @test C ≈ 2 .* C0 rtol=rtol
        end
    end

    # Float16 takes the other arm of _gemm_dispatch!, through gemmEx!. Its
    # absence here is why a missing CPU gemmEx! went unnoticed: the two types
    # above both route to gemm!, so nothing in this file reached the arm that
    # threw. The reference is in double precision, and the tolerance is
    # Float16 storage's rather than the accumulation's.
    @testset "Float16 through gemmEx!" begin
        @testset "destination $TC" for TC in (Float16, Float32)
            n = 16
            A = rand(Float16, n, n)
            B = rand(Float16, n, n)
            C = zeros(TC, n, n)
            NextLA.recgemm!(1.0, A, B, 0.0, C)
            @test C ≈ Float64.(A) * Float64.(B) rtol=(TC === Float16 ? 5e-3 : 1e-3)
        end

        # the untyped literals recgemm!'s own callers use promote the compute
        # type to Float64 over Float16 storage; that combination has to work
        @testset "accumulate and subtract" begin
            n = 16
            A = rand(Float16, n, n)
            B = rand(Float16, n, n)
            C0 = rand(Float16, n, n)
            C = copy(C0)
            NextLA.recgemm!(-1.0, A, B, 1.0, C)
            @test C ≈ Float64.(C0) .- Float64.(A) * Float64.(B) rtol=5e-3
        end
    end
end

# A destination wider than both operands: the operands are widened exactly and
# the product is formed in C's type, so the result carries C's precision.
# Computing in the operands' type gave Float32 accuracy on the CPU (relative
# error ~2.5e-7), and cuBLAS refused it with CUBLAS_STATUS_NOT_SUPPORTED.
@testset "RECGEMM C wider than the operands [$backend_name]" for (backend_name, ArrayType, synchronize) in
        filter(b -> b[1] in ("CPU", "CUDA", "AMDGPU"), available_backends())
    @testset "$TA, $TB -> $TC" for (TA, TB, TC) in ((Float32, Float32, Float64),
                                                    (Float16, Float16, Float64),
                                                    (Float16, Float32, Float64),
                                                    (ComplexF32, ComplexF32, ComplexF64))
        n = 32
        A, B, C0 = rand(TA, n, n), rand(TB, n, n), rand(TC, n, n)
        C = ArrayType(copy(C0))
        NextLA.recgemm!(2, ArrayType(A), ArrayType(B), 3, C)
        synchronize(C)
        @test Array(C) ≈ 2 .* (TC.(A) * TC.(B)) .+ 3 .* C0 rtol=1e-12
    end
end

# A destination narrower than the operands. The CPU accumulates at the operands'
# width and rounds once on store; a device rounds the operands to C's type
# first, because cuBLAS refuses Float64 compute into a Float32 C and Float32
# compute into a Float16 C (CUBLAS_STATUS_NOT_SUPPORTED).
@testset "RECGEMM C narrower than the operands [$backend_name]" for (backend_name, ArrayType, synchronize) in
        filter(b -> b[1] in ("CPU", "CUDA", "AMDGPU"), available_backends())
    @testset "$TA, $TB -> $TC" for (TA, TB, TC, rtol) in ((Float64, Float64, Float32, 1e-5),
                                                          (Float64, Float32, Float32, 1e-5),
                                                          (Float32, Float32, Float16, 1e-2),
                                                          (Float32, Float16, Float16, 1e-2))
        n = 32
        A, B, C0 = rand(TA, n, n), rand(TB, n, n), rand(TC, n, n)
        C = ArrayType(copy(C0))
        NextLA.recgemm!(2, ArrayType(A), ArrayType(B), 3, C)
        synchronize(C)
        @test Float64.(Array(C)) ≈ 2 .* (Float64.(A) * Float64.(B)) .+ 3 .* Float64.(C0) rtol=rtol
    end
end
