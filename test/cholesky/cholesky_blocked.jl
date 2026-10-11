# Blocked right-looking Cholesky. GPU only: the kernel is declared cpu=false on
# the branch it came from, so the CPU path is asserted to refuse rather than to
# return numbers.
#
# Diagonally dominant input throughout. There is no positive-definiteness check
# in the kernel -- it takes sqrt as it goes -- so an indefinite matrix would
# produce NaN rather than an error, which is recorded rather than tested for.

@testset "CHOLESKY_BLOCKED [GPU]" begin
    gpus = available_gpu_backends()
    if isempty(gpus)
        @test_skip "no GPU backend available"
    end

    for (backend_name, ArrayType, synchronize) in gpus
        @testset "[$backend_name] $T" for T in (Float32, Float64)
            rtol = test_rtol(T)

            @testset "n=$n block=$bs" for (n, bs) in [
                (32, 32), (64, 32), (64, 64), (128, 64), (100, 64),
            ]
                A0 = rand(T, n, n)
                A0 = A0 * A0' + T(n) * I          # symmetric positive definite
                Ad = ArrayType(copy(A0))

                NextLA.cholesky_blocked!(Ad; block = bs)
                synchronize(Ad)

                L = LowerTriangular(Array(Ad))
                @test norm(L * L' - A0) / norm(A0) < rtol
            end

            # The register kernel is the other way of factoring a diagonal
            # block: same driver, same panel solve and trailing update. It is
            # checked over the same shapes, and against the shared kernel --
            # both walk k in the same order, so they agree bit for bit and a
            # mere tolerance check would not notice one of them drifting.
            @testset "register kernel n=$n block=$bs" for (n, bs) in [
                (32, 32), (64, 32), (64, 64), (128, 64), (100, 64),
            ]
                A0 = rand(T, n, n)
                A0 = A0 * A0' + T(n) * I
                Ad = ArrayType(copy(A0))
                Ashared = ArrayType(copy(A0))

                NextLA.cholesky_blocked!(Ad; block = bs, kernel = :register)
                NextLA.cholesky_blocked!(Ashared; block = bs, kernel = :shared)
                synchronize(Ad)
                synchronize(Ashared)

                L = LowerTriangular(Array(Ad))
                @test norm(L * L' - A0) / norm(A0) < rtol
                @test Array(Ad) == Array(Ashared)
            end

            # Launched on its own, without the driver's panel sweep around it.
            @testset "register kernel alone on one block, n=$n" for n in (16, 33, 48, 64)
                A0 = rand(T, n, n)
                A0 = A0 * A0' + T(n) * I
                Ad = ArrayType(copy(A0))
                before = triu(Array(Ad), 1)

                backend = KernelAbstractions.get_backend(Ad)
                NextLA.chol_kernel_register!(backend, NextLA.CHOL_REG_THREADS)(
                    Ad, Val(n); ndrange = NextLA.CHOL_REG_THREADS)
                synchronize(Ad)

                L = LowerTriangular(Array(Ad))
                @test norm(L * L' - A0) / norm(A0) < rtol
                @test triu(Array(Ad), 1) == before
            end

            @testset "leaves the strict upper triangle alone" begin
                n = 64
                A0 = rand(T, n, n)
                A0 = A0 * A0' + T(n) * I
                Ad = ArrayType(copy(A0))
                before = triu(Array(Ad), 1)

                NextLA.cholesky_blocked!(Ad; block = 64)
                synchronize(Ad)

                @test triu(Array(Ad), 1) == before
            end
        end
    end

    # The kernel cannot run on the CPU backend, so the entry point refuses there
    # instead of failing somewhere inside KernelAbstractions.
    @testset "refuses the CPU backend" begin
        A = rand(Float64, 32, 32)
        A = A * A' + 32I
        @test_throws ArgumentError NextLA.cholesky_blocked!(A)
    end

    @testset "rejects an unknown kernel" begin
        for (_, AT, _) in available_gpu_backends()
            A = AT(rand(Float64, 64, 64) + 64I)
            @test_throws ArgumentError NextLA.cholesky_blocked!(A; kernel = :nosuch)
            break
        end
    end

    @testset "rejects a block wider than the shared tile" begin
        for (_, AT, _) in available_gpu_backends()
            A = AT(rand(Float64, 64, 64) + 64I)
            @test_throws ArgumentError NextLA.cholesky_blocked!(A; block = 128)
            break
        end
    end
end

# Half precision, with both diagonal-block kernels. The panel solve hands the
# Float16 TRSM base case a transposed view of A, which is the case that has to
# stay on the device.
@testset "CHOLESKY_BLOCKED Float16 [GPU]" begin
    gpus = available_gpu_backends()
    if isempty(gpus)
        @test_skip "no GPU backend available"
    end

    for (backend_name, ArrayType, synchronize) in gpus
        @testset "[$backend_name] kernel=$kernel n=$n block=$bs" for
                kernel in (:shared, :register),
                (n, bs) in [(32, 32), (64, 32), (128, 64), (100, 64)]
            R = rand(n, n)
            A0 = Float16.(R * R' + n * I)
            Ad = ArrayType(copy(A0))

            NextLA.cholesky_blocked!(Ad; block = bs, kernel = kernel)
            synchronize(Ad)

            L = LowerTriangular(Float64.(Array(Ad)))
            @test norm(L * L' - Float64.(A0)) / norm(Float64.(A0)) < 5e-3
        end
    end
end
