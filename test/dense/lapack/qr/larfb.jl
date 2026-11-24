for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "LARFB [$backend_name]" begin
        @testset "$T" for T in TEST_TYPES
                rtol = test_rtol(T)

                @testset "side=$side, trans=$trans, m=$m, n=$n" for
                        side  in ['L', 'R'],
                        trans in ['N', 'C'],
                        (m, n) in [(5, 4), (10, 8), (20, 15)]

                    k = min(side == 'L' ? m : n, 4)
                    panel_rows = side == 'L' ? m : n

                    panel = randn(T, panel_rows, k)
                    tau_vec = zeros(T, k)
                    qr_work = zeros(T, k)
                    NextLA.geqr2!(panel_rows, k, panel, tau_vec, qr_work)

                    V_init = copy(panel)
                    for j in 1:k
                        for i in 1:j-1
                            V_init[i, j] = zero(T)
                        end
                        V_init[j, j] = one(T)
                    end
                    T_mat_init = zeros(T, k, k)
                    NextLA.larft!('F', 'C', panel_rows, k, V_init, tau_vec, T_mat_init)

                    V = ArrayType(copy(V_init))
                    T_mat = ArrayType(copy(T_mat_init))
                    C_init = randn(T, m, n)
                    C_orig = copy(C_init)
                    C = ArrayType(copy(C_init))

                    NextLA.larfb!(side, trans, 'F', 'C', V, T_mat, C)
                    synchronize(C)

                    I_mat = Matrix{T}(I, panel_rows, panel_rows)
                    H = I_mat - V_init * T_mat_init * V_init'
                    Ht = trans == 'N' ? H : H'
                    C_ref = side == 'L' ? Ht * C_orig : C_orig * Ht
                    @test Array(C) ≈ C_ref rtol=rtol * max(1, m, n)
                end

                @testset "Roundtrip H then Hᴴ" begin
                    m, n, k = 12, 10, 3
                    panel = randn(T, m, k)
                    tau_vec = zeros(T, k)
                    qr_work = zeros(T, k)
                    NextLA.geqr2!(m, k, panel, tau_vec, qr_work)
                    V_init = copy(panel)
                    for j in 1:k
                        for i in 1:j-1; V_init[i, j] = zero(T); end
                        V_init[j, j] = one(T)
                    end
                    T_mat_init = zeros(T, k, k)
                    NextLA.larft!('F', 'C', m, k, V_init, tau_vec, T_mat_init)

                    V = ArrayType(copy(V_init))
                    T_mat = ArrayType(copy(T_mat_init))
                    C_init = randn(T, m, n)
                    C_orig = copy(C_init)
                    C = ArrayType(copy(C_init))

                    NextLA.larfb!('L', 'N', 'F', 'C', V, T_mat, C)
                    synchronize(C)
                    NextLA.larfb!('L', 'C', 'F', 'C', V, T_mat, C)
                    synchronize(C)

                    @test Array(C) ≈ C_orig rtol=rtol * m
                end

                @testset "Helper matches kernel" begin
                    m, n, k = 10, 8, 3
                    panel = randn(T, m, k)
                    tau_vec = zeros(T, k)
                    qr_work = zeros(T, k)
                    NextLA.geqr2!(m, k, panel, tau_vec, qr_work)
                    V_init = copy(panel)
                    for j in 1:k
                        for i in 1:j-1; V_init[i, j] = zero(T); end
                        V_init[j, j] = one(T)
                    end
                    T_mat_init = zeros(T, k, k)
                    NextLA.larft!('F', 'C', m, k, V_init, tau_vec, T_mat_init)

                    V = ArrayType(copy(V_init))
                    T_mat = ArrayType(copy(T_mat_init))
                    C_init = randn(T, m, n)
                    C1 = ArrayType(copy(C_init))
                    work = ArrayType(zeros(T, n, k))
                    NextLA.larfb!('L', 'N', 'F', 'C', m, n, k, V, m, T_mat, C1, work)
                    synchronize(C1)

                    C2 = ArrayType(copy(C_init))
                    NextLA.larfb!('L', 'N', 'F', 'C', V, T_mat, C2)
                    synchronize(C2)

                    @test Array(C1) ≈ Array(C2) rtol=rtol
                end
        end
    end
end

@testset "LARFB Edge cases" begin
    @test_nowarn NextLA.larfb!('L', 'N', 'F', 'C', 0, 0, 0,
        zeros(0, 0), 1, zeros(0, 0), zeros(0, 0), zeros(0, 0))
    @test_nowarn NextLA.larfb!('L', 'N', 'F', 'C', 5, 5, 0,
        zeros(5, 0), 5, zeros(0, 0), randn(5, 5), zeros(5, 0))
end
