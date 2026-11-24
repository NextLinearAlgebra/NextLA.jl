# gemmEx! across backends.
#
# gemmEx! is the mixed-storage entry point: A, B and C may each have their own
# element type, and the accumulation happens in compute_type. On CUDA and
# AMDGPU that is a vendor GEMMEx call. On the CPU there is no such primitive, so
# a triple BLAS accepts goes to BLAS.gemm! and everything else is accumulated in
# compute_type and rounded once on store -- the same operation, not the same
# speed. oneAPI and Metal have no gemmEx!: it throws there for every signature.
#
# The reference is always computed in double precision and compared against,
# rather than in the storage type, so a wrong accumulation order shows up
# instead of being hidden by the rounding it causes.

for (backend_name, ArrayType, synchronize) in available_backends()
    if !(backend_name in ("CPU", "CUDA", "AMDGPU"))
        @testset "GEMMEX [$backend_name] refuses" begin
            A = ArrayType(rand(Float32, 4, 4))
            C = ArrayType(zeros(Float32, 4, 4))
            @test_throws ArgumentError NextLA.gemmEx!('N', 'N', 1.0f0, A, A, 0.0f0, C)
        end
        continue
    end
    # Local to the loop, not top level: these are this file's reference
    # arithmetic, not names the library gains, and the review pages read an
    # unindented definition in a test file as a contribution.
    _ex_op(M, t::Char) = t == 'N' ? M : (t == 'C' ? adjoint(M) : transpose(M))
    _ex_refr(transA, transB, alpha, A, B, beta, C) =
        alpha .* (_ex_op(Float64.(A), transA) * _ex_op(Float64.(B), transB)) .+
        beta .* Float64.(C)

    @testset "GEMMEX [$backend_name]" begin

        # Every backend supports the same-type BLAS-shaped case.
        @testset "same type $T" for T in (Float32, Float64)
            rtol = test_rtol(T)

            @testset "trans $ta$tb, n=$n k=$k m=$m" for
                    (ta, tb) in (('N', 'N'), ('T', 'N'), ('N', 'T'), ('T', 'T')),
                    (n, k, m) in ((16, 16, 16), (32, 16, 8), (17, 9, 5))

                A = ta == 'N' ? rand(T, n, k) : rand(T, k, n)
                B = tb == 'N' ? rand(T, k, m) : rand(T, m, k)
                C0 = rand(T, n, m)
                alpha, beta = T(2), T(3)
                expected = _ex_refr(ta, tb, 2.0, A, B, 3.0, C0)

                Ad, Bd = ArrayType(copy(A)), ArrayType(copy(B))
                Cd = ArrayType(copy(C0))
                NextLA.gemmEx!(ta, tb, alpha, Ad, Bd, beta, Cd)
                synchronize(Cd)
                @test Array(Cd) ≈ expected rtol=rtol
            end

            @testset "alpha=0 scales C only" begin
                n = 8
                A, B, C0 = rand(T, n, n), rand(T, n, n), rand(T, n, n)
                Cd = ArrayType(copy(C0))
                NextLA.gemmEx!('N', 'N', zero(T), ArrayType(A), ArrayType(B), T(2), Cd)
                synchronize(Cd)
                @test Array(Cd) ≈ 2 .* C0 rtol=rtol
            end

            @testset "beta=0 overwrites C" begin
                n = 8
                A, B = rand(T, n, n), rand(T, n, n)
                Cd = ArrayType(fill(T(99), n, n))
                NextLA.gemmEx!('N', 'N', one(T), ArrayType(A), ArrayType(B), zero(T), Cd)
                synchronize(Cd)
                @test Array(Cd) ≈ Float64.(A) * Float64.(B) rtol=rtol
            end
        end

        # 'C' is the adjoint and must not be collapsed into 'T'. TLR passes
        # adjoint_blas_char(T), which is 'C' exactly where the two differ.
        @testset "adjoint is not transpose" begin
            n = 12
            A, B = rand(ComplexF32, n, n), rand(ComplexF32, n, n)
            Cc = ArrayType(zeros(ComplexF32, n, n))
            Ct = ArrayType(zeros(ComplexF32, n, n))
            NextLA.gemmEx!('C', 'N', 1.0f0, ArrayType(A), ArrayType(B), 0.0f0, Cc)
            NextLA.gemmEx!('T', 'N', 1.0f0, ArrayType(A), ArrayType(B), 0.0f0, Ct)
            synchronize(Cc); synchronize(Ct)
            @test Array(Cc) ≈ adjoint(ComplexF64.(A)) * ComplexF64.(B) rtol=1e-5
            @test Array(Ct) ≈ transpose(ComplexF64.(A)) * ComplexF64.(B) rtol=1e-5
            @test !(Array(Cc) ≈ Array(Ct))
        end

        @testset "Float16 storage" begin
            n = 16
            A, B = rand(Float16, n, n), rand(Float16, n, n)
            ref = Float64.(A) * Float64.(B)

            for TC in (Float16, Float32)
                Cd = ArrayType(zeros(TC, n, n))
                # The storage decides the compute type: Float16 for an
                # all-Float16 triple, Float32 once C is Float32. The scalars
                # take no part.
                NextLA.gemmEx!('N', 'N', 1.0f0, ArrayType(A), ArrayType(B), 0.0f0, Cd)
                synchronize(Cd)
                # Float16 storage carries ~3 decimal digits, so the
                # tolerance is the storage's, not the accumulation's.
                @test Array(Cd) ≈ ref rtol=(TC === Float16 ? 1e-2 : 1e-3)
            end
        end

        @testset "compute_type is honoured" begin
            n = 16
            A, B = rand(Float16, n, n), rand(Float16, n, n)
            ref = Float64.(A) * Float64.(B)
            Cd = ArrayType(zeros(Float32, n, n))
            NextLA.gemmEx!('N', 'N', 1.0, ArrayType(A), ArrayType(B), 0.0, Cd;
                           compute_type=Float32)
            synchronize(Cd)
            @test Array(Cd) ≈ ref rtol=1e-3
        end

        # A compute type wider than the storage is legitimate, and the CPU
        # honours it -- the accumulation happens in Float64 and is rounded once
        # on store. cuBLAS rejects the same request outright, which is why
        # _gemm_dispatch! names Float32; this asserts the CPU capability the
        # fallback relies on.
        if backend_name == "CPU"
            @testset "Float64 compute over Float16 storage" begin
                n = 16
                A, B = rand(Float16, n, n), rand(Float16, n, n)
                C = zeros(Float16, n, n)
                NextLA.gemmEx!('N', 'N', -1.0, A, B, 1.0, C; compute_type=Float64)
                @test C ≈ -(Float64.(A) * Float64.(B)) rtol=5e-3
            end

            # BLAS guarantees a zero scalar means the corresponding operand is
            # never read. That matters here because Float16 overflows to Inf
            # readily, and 0 * Inf is NaN -- so a caller that scales a product
            # out of the result would otherwise still be poisoned by it.
            @testset "a zero scalar does not read its operand" begin
                n = 4
                Binf = ones(Float16, n, n)

                # wide path: alpha = 0 must ignore an Inf in A
                Ainf = fill(Float16(Inf), n, n)
                C = fill(Float16(2), n, n)
                NextLA.gemmEx!('N', 'N', 0.0, Ainf, Binf, 3.0, C; compute_type=Float32)
                @test all(C .== 6)

                # wide path: beta = 0 must ignore a NaN already in C
                Cnan = fill(Float16(NaN), n, n)
                NextLA.gemmEx!('N', 'N', 1.0, ones(Float16, n, n), Binf, 0.0, Cnan;
                               compute_type=Float32)
                @test all(Cnan .== n)

                # both zero: C is zeroed, operands untouched
                Cz = fill(Float16(7), n, n)
                NextLA.gemmEx!('N', 'N', 0.0, Ainf, Binf, 0.0, Cz; compute_type=Float32)
                @test all(iszero, Cz)

                # BLAS-type operands too: alpha = 0 must not reach BLAS, whose kernels may read A
                Ad = fill(Inf, n, n)
                Cd = fill(2.0, n, n)
                NextLA.gemmEx!('N', 'N', 0.0, Ad, ones(n, n), 3.0, Cd)
                @test all(Cd .== 6)
            end

            @testset "argument errors" begin
                n = 8
                A, B = rand(Float32, n, n), rand(Float32, n, n)
                @test_throws DimensionMismatch NextLA.gemmEx!(
                    'N', 'N', 1.0f0, A, B, 0.0f0, zeros(Float32, n + 1, n))
                @test_throws ArgumentError NextLA.gemmEx!(
                    'N', 'N', 1.0f0, A, B, 0.0f0, zeros(Float32, n, n);
                    compute_type=Int8)
            end

            # Only single-matrix gemmEx! widens on the CPU. The batched and
            # precision entry points take what BLAS takes, and say so.
            @testset "batched and precision refuse Float16" begin
                n = 4
                A, B = rand(Float16, n, n), rand(Float16, n, n)
                mode = NextLA.GEMMCompute{Float32}()
                @test !NextLA.gemm_signature_supported(
                    KernelAbstractions.CPU(), Float16, Float16, Float32, mode)
                @test_throws ArgumentError NextLA.precision_gemm!(
                    'N', 'N', 1.0f0, A, B, 0.0f0, zeros(Float32, n, n), mode)
                @test_throws ArgumentError NextLA.gemmEx_batched!(
                    'N', 'N', 1.0, rand(Float16, n, n, 2), rand(Float16, n, n, 2),
                    0.0, zeros(Float32, n, n, 2))
                @test_throws ArgumentError NextLA.gemm_batched!(
                    'N', 'N', 1.0f0, rand(Float16, n, n, 2), rand(Float16, n, n, 2),
                    0.0f0, zeros(Float16, n, n, 2))
            end

            # Pointer batches stay GPU-only: there is deliberately no CPU
            # _build_batch_ptrs, and this records that as intended.
            @testset "pointer batches remain unsupported" begin
                n = 4
                A, B = rand(Float32, n, n), rand(Float32, n, n)
                C = zeros(Float32, n, n)
                @test !NextLA.supports_pointer_batched(KernelAbstractions.CPU())
                @test_throws ArgumentError NextLA.gemm_batched_ptrs!(
                    'N', 'N', 1.0f0, nothing, A, nothing, B, 0.0f0, nothing, C, 1)
            end
        end
    end
end

# gemm! in the BLAS argument order, against LinearAlgebra.BLAS over all three
# transposes, with alpha and beta away from 1 and 0 so both scalings run.
# Complex only where the backend's GEMM takes it.
@testset "gemm! flag form [$backend_name]" for (backend_name, ArrayType, synchronize) in available_backends()
    types = backend_name in ("CPU", "CUDA", "AMDGPU") ?
        (Float32, Float64, ComplexF32, ComplexF64) : (Float32, Float64)
    @testset "$T $ta$tb" for T in types, ta in ('N', 'T', 'C'), tb in ('N', 'T', 'C')
        m, k, n = 7, 5, 6
        A0 = ta == 'N' ? rand(T, m, k) : rand(T, k, m)
        B0 = tb == 'N' ? rand(T, k, n) : rand(T, n, k)
        C0 = rand(T, m, n)
        alpha, beta = T(1.5), T(-0.5)
        ref = copy(C0)
        LinearAlgebra.BLAS.gemm!(ta, tb, alpha, A0, B0, beta, ref)
        C = ArrayType(copy(C0))
        out = gemm!(ta, tb, alpha, ArrayType(A0), ArrayType(B0), beta, C)
        synchronize(C)
        @test out === C
        @test Array(C) ≈ ref rtol = (real(T) === Float32 ? 1e-5 : 1e-12)
    end
    @testset "rejects a bad flag" begin
        A = ArrayType(rand(Float32, 2, 2))
        @test_throws ArgumentError gemm!('X', 'N', 1f0, A, A, 0f0, similar(A))
    end
end

# The compute type comes from the storage types: an untyped alpha or beta (a
# Float64 literal on Float32 data) must not raise it. On CUDA it used to, and
# cuBLAS refused the Float64 compute it asked for.
@testset "compute type follows the data [$backend_name]" for (backend_name, ArrayType, synchronize) in available_backends()
    backend_name in ("CPU", "CUDA", "AMDGPU") || continue
    A0, B0, C0 = rand(Float32, 6, 4), rand(Float32, 4, 5), rand(Float32, 6, 5)
    A, B, C = ArrayType(A0), ArrayType(B0), ArrayType(copy(C0))
    @test NextLA.default_compute_type(0.5, A, B, 2.0, C) == Float32
    @test NextLA.default_compute_type(1.0, A0, B0, 0.0, zeros(Float64, 6, 5)) == Float64
    NextLA.gemmEx!('N', 'N', 0.5, A, B, 2.0, C)
    synchronize(C)
    @test Array(C) ≈ 0.5f0 .* (A0 * B0) .+ 2f0 .* C0 rtol = 1e-5
end
