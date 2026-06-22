using Test
using LinearAlgebra

@testset "TLR black-box sampling" begin
    @testset "rademacher generator" begin
        seed = 0x12345678
        word11 = NextLA.rademacher_word(1, 1, seed)
        word11_repeat = NextLA.rademacher_word(1, 1, seed)
        word12 = NextLA.rademacher_word(1, 2, seed)
        word21 = NextLA.rademacher_word(2, 1, seed)
        word11_batched = NextLA.rademacher_word(1, 1, seed; batch=2, use_batch_in_rng=true)

        @test word11 == word11_repeat
        @test word11 != word12
        @test word11 != word21
        @test word11 != word11_batched
        @test NextLA.rademacher_sign(Float32, 3, 1, seed) in (-1.0f0, 1.0f0)
        @test NextLA.rademacher_sign(ComplexF32, 3, 17, seed) in (ComplexF32(-1), ComplexF32(1))
    end

    @testset "inverse_tile_index" begin
        col = NextLA.TileColMajor(3, 2)
        row = NextLA.TileRowMajor(3, 2)
        @test NextLA.inverse_tile_index(col, 1) == (1, 1)
        @test NextLA.inverse_tile_index(col, 4) == (1, 2)
        @test NextLA.inverse_tile_index(row, 1) == (1, 1)
        @test NextLA.inverse_tile_index(row, 4) == (2, 2)
    end

    cases = [
        ("full tiles", Float32.(reshape(1:64, 8, 8)), 4, 4, 8, 32, 4, NextLA.TileColMajor(cld(8, 4), cld(8, 4))),
        ("boundary row", Float32.(reshape(1:80, 10, 8)), 4, 4, 8, 32, 4, NextLA.TileRowMajor(cld(10, 4), cld(8, 4))),
        ("boundary col", Float32.(reshape(1:80, 8, 10)), 4, 4, 8, 32, 4, NextLA.TileColMajor(cld(8, 4), cld(10, 4))),
        ("S not divisible by CPW", Float32.(reshape(1:99, 9, 11)), 4, 4, 10, 32, 4, NextLA.TileRowMajor(cld(9, 4), cld(11, 4))),
        ("runtime S=37", Float32.(reshape(1:143, 11, 13)), 4, 4, 37, 32, 4, NextLA.TileColMajor(cld(11, 4), cld(13, 4))),
        ("larger sketch S=96", Float32.(reshape(1:168, 12, 14)), 4, 4, 96, 32, 8, NextLA.TileRowMajor(cld(12, 4), cld(14, 4))),
    ]

    @testset "sample_range_rademacher!" begin
        seed = 0x42

        for (case_name, A, tile_m, tile_n, S, SBLK, CPW, order) in cases
            @testset "$case_name" begin
                Y_ref = zeros(Float32, tile_m, S, prod(size(order)))
                NextLA.sample_range_rademacher_reference!(
                    Y_ref, A, order, tile_m, tile_n;
                    S,
                    seed,
                    CPW,
                    use_batch_in_rng=true,
                )

                for (backend_name, AT, sync) in backends
                    @testset "$backend_name" begin
                        A_dev = _to_backend(AT, copy(A))
                        Y_dev = _to_backend(AT, fill(Float32(-99), size(Y_ref)))

                        NextLA.sample_range_rademacher!(
                            Y_dev, A_dev, order, tile_m, tile_n;
                            S,
                            seed,
                            nthreads=256,
                            ROWS_CTA=128,
                            BK=16,
                            SBLK,
                            CPW,
                            use_batch_in_rng=true,
                        )
                        sync(Y_dev)

                        Y_actual = Array(Y_dev)
                        @test Y_actual ≈ Y_ref

                        for batch in 1:prod(size(order))
                            tile_i, tile_j = NextLA.inverse_tile_index(order, batch)
                            p0 = (tile_i - 1) * tile_m + 1
                            q0 = (tile_j - 1) * tile_n + 1
                            tile_m_eff = min(tile_m, size(A, 1) - p0 + 1)
                            tile_n_eff = min(tile_n, size(A, 2) - q0 + 1)

                            @test Y_actual[1:tile_m_eff, 1:S, batch] ≈
                                  Y_ref[1:tile_m_eff, 1:S, batch]

                            if tile_m_eff < tile_m
                                @test all(iszero, Y_actual[(tile_m_eff + 1):end, :, batch])
                            end

                            Omega_tile = Array{Float32}(undef, tile_n_eff, S)
                            for k in 1:tile_n_eff, s in 1:S
                                Omega_tile[k, s] = NextLA.rademacher_sign(Float32, k, s, seed; batch, use_batch_in_rng=true)
                            end
                            rows = p0:(p0 + tile_m_eff - 1)
                            cols = q0:(q0 + tile_n_eff - 1)
                            tile = @view A[rows, cols]
                            @test Y_actual[1:tile_m_eff, 1:S, batch] ≈ tile * Omega_tile
                        end
                    end
                end
            end
        end
    end

    @testset "sample_range_rademacher! offdiag_only" begin
        seed = 0x31415926
        tile_m = 256
        tile_n = 128
        S = 16
        order = NextLA.TileColMajor(3, 2)
        tile_dims = size(order)
        A = randn(Float32, tile_m * tile_dims[1], tile_n * tile_dims[2])
        ndiag = min(size(order)...)
        noffdiag = prod(size(order)) - ndiag

        Y_ref = zeros(Float32, tile_m, S, noffdiag)
        NextLA.sample_range_rademacher_reference!(
            Y_ref, A, order, tile_m, tile_n;
            S,
            seed,
            CPW=4,
            use_batch_in_rng=true,
            offdiag_only=true,
        )

        for (backend_name, AT, sync) in backends
            @testset "$backend_name" begin
                A_dev = _to_backend(AT, copy(A))
                Y_dev = _to_backend(AT, fill(Float32(-99), size(Y_ref)))

                NextLA.sample_range_rademacher!(
                    Y_dev, A_dev, order, tile_m, tile_n;
                    S,
                    seed,
                    nthreads=256,
                    ROWS_CTA=128,
                    BK=16,
                    SBLK=32,
                    CPW=4,
                    use_batch_in_rng=true,
                    offdiag_only=true,
                )
                sync(Y_dev)

                @test Array(Y_dev) ≈ Y_ref
            end
        end
    end
end
