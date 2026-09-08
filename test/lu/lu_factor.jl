# Both routines pack L and U into A and return the row permutation as a vector
# p, with A₀[p, :] == L*U, and LAPACK's info: 0, or the first zero pivot.
#
# lu_base! runs on every backend. tile_lu_factor! is GPU-only: it is correct on
# CUDA and returns wrong numbers on the CPU backend, so it refuses there. See
# KNOWN_ISSUES.md.

for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "LU_BASE [$backend_name]" begin
        @testset "$T" for T in (Float32, Float64)
            rtol = test_rtol(T)

            @testset "n=$n" for n in [4, 8, 16, 32]
                # diagonally dominant: well conditioned, and the pivoting has
                # something to do without being pathological
                A0 = rand(T, n, n) + T(n) * I
                Ad = ArrayType(copy(A0))

                Af, p, info = NextLA.lu_base!(Ad)
                synchronize(Af)

                A = Array(Af)
                L = UnitLowerTriangular(A)
                U = UpperTriangular(A)
                @test info == 0
                @test p isa Vector{Int} && sort(p) == 1:n
                @test norm(A0[p, :] - L * U) / norm(A0) < rtol
            end
        end
    end
end

@testset "TILE_LU_FACTOR [GPU]" begin
    gpus = available_gpu_backends()
    if isempty(gpus)
        @test_skip "no GPU backend available"
    end

    for (backend_name, ArrayType, synchronize) in gpus
        @testset "[$backend_name] $T" for T in (Float32, Float64)
            rtol = test_rtol(T)

            @testset "N=$N tile=$tile" for (N, tile) in [
                (16, 4), (16, 8), (32, 8), (64, 16),
            ]
                A0 = rand(T, N, N) + T(N) * I
                Ad = ArrayType(copy(A0))

                Af, p, info = NextLA.tile_lu_factor!(Ad, tile)
                synchronize(Af)

                A = Array(Af)
                L = UnitLowerTriangular(A)
                U = UpperTriangular(A)
                @test info == 0
                @test p isa Vector{Int} && sort(p) == 1:N
                @test norm(A0[p, :] - L * U) / norm(A0) < rtol
            end
        end
    end

    # The CPU path is wrong rather than merely unsupported, so it refuses
    # instead of returning plausible numbers. Guard the guard.
    @testset "refuses the CPU backend" begin
        A = rand(Float64, 16, 16) + 16I
        @test_throws ArgumentError NextLA.tile_lu_factor!(A, 4)
    end

    @testset "a work-group past 32 x 32 is refused on a GPU" begin
        for (name, AT, _) in available_gpu_backends()
            @test_throws ArgumentError NextLA.lu_base!(AT(rand(Float64, 33, 33) + 33I))
            @test_throws ArgumentError NextLA.tile_lu_factor!(AT(rand(Float64, 66, 66) + 66I), 33)
            break
        end
    end

    @testset "a zero pivot is reported through info" begin
        for (name, AT, _) in available_gpu_backends()
            @test NextLA.lu_base!(AT(zeros(Float64, 4, 4)))[3] == 1
            @test NextLA.tile_lu_factor!(AT(zeros(Float64, 8, 8)), 4)[3] == 1
            break
        end
        @test NextLA.lu_base!(zeros(Float64, 4, 4))[3] == 1
    end

    @testset "tile size must divide the matrix" begin
        for (name, AT, _) in available_gpu_backends()
            A = AT(rand(Float64, 16, 16) + 16I)
            @test_throws ArgumentError NextLA.tile_lu_factor!(A, 5)
            break
        end
    end
end
