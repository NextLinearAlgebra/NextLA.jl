@testset "PARFB" begin
    @testset "$T" for T in TEST_TYPES
        rtol = test_rtol(T)

        @testset "side=$side, trans=$trans, direct=$direct, storev=$storev" for
                side in ['L', 'R'],
                trans in ['N', 'C'],
                direct in ['F', 'B'],
                storev in ['C', 'R']

            # Skip the LAPACK-incompatible combo
            if side == 'L' && direct == 'F' && storev == 'R'
                continue
            end

            m2, n2, k, l = 20, 24, 8, 4

            # A1 dimensions depend on side
            if side == 'L'
                m1, n1 = k, n2  # A is K×N when side='L'
            else
                m1, n1 = m2, k  # A is M×K when side='R'
            end

            A1 = rand(T, m1, n1)
            A2 = rand(T, m2, n2)

            # V dimensions depend on storev and side
            if storev == 'C'
                V = side == 'L' ? rand(T, m2, k) : rand(T, n2, k)
            else
                V = side == 'L' ? rand(T, k, m2) : rand(T, k, n2)
            end

            Tee = rand(T, k, k)

            # Workspace
            work_n = side == 'L' ? rand(T, k, n2) : rand(T, m2, k)

            A1_n = copy(A1); A2_n = copy(A2)
            NextLA.parfb!(side, trans, direct, storev, m1, n1, m2, n2, k, l,
                          A1_n, A2_n, copy(V), copy(Tee), copy(work_n))

            # LAPACK reference
            A1_l = copy(A1); A2_l = copy(A2)
            work_l = lapack_tprfb!(T, side, trans, direct, storev, l,
                                   copy(V), copy(Tee), A1_l, A2_l)

            @test A1_n ≈ A1_l rtol=rtol
            @test A2_n ≈ A2_l rtol=rtol
        end
    end

    @testset "Error handling" begin
        @test_throws ArgumentError NextLA.parfb!('X', 'N', 'F', 'C', 4, 4, 4, 4, 2, 0,
            zeros(2, 4), zeros(4, 4), zeros(4, 2), zeros(2, 2), zeros(2, 4))
        @test_throws ArgumentError NextLA.parfb!('L', 'X', 'F', 'C', 4, 4, 4, 4, 2, 0,
            zeros(2, 4), zeros(4, 4), zeros(4, 2), zeros(2, 2), zeros(2, 4))
        @test_throws ArgumentError NextLA.parfb!('L', 'N', 'X', 'C', 4, 4, 4, 4, 2, 0,
            zeros(2, 4), zeros(4, 4), zeros(4, 2), zeros(2, 2), zeros(2, 4))
        @test_throws ArgumentError NextLA.parfb!('L', 'N', 'F', 'X', 4, 4, 4, 4, 2, 0,
            zeros(2, 4), zeros(4, 4), zeros(4, 2), zeros(2, 2), zeros(2, 4))
    end
end
