# syrk! and syrk_batched!.
#
# Source: origin/pr/19, by Alessandro <axelcar482@gmail.com>. Adapted from that
# branch's own `backends` / `_to_backend` locals to this repo's
# available_backends(), and extended with the untouched-triangle and
# dispatch-path assertions below.
#
# The point of SYRK over GEMM is that it writes one triangle. Checking only the
# written half, as the branch test did, cannot tell a correct SYRK from a GEMM
# that happens to agree there -- so the other triangle is asserted unchanged.

function _syrk_expected(trans::Char, alpha, A, beta, C)
    prod = trans == 'N' ? A * transpose(A) : transpose(A) * A
    return alpha * prod + beta * C
end

function _triangle_matches(uplo::Char, got::AbstractMatrix, expected::AbstractMatrix)
    n = size(got, 1)
    if uplo == 'U'
        for j in 1:n, i in 1:j
            @inbounds isapprox(got[i, j], expected[i, j]) || return false
        end
    else
        for j in 1:n, i in j:n
            @inbounds isapprox(got[i, j], expected[i, j]) || return false
        end
    end
    return true
end

# the half syrk! must not write: strictly outside the `uplo` triangle
function _other_triangle_untouched(uplo::Char, got::AbstractMatrix, before::AbstractMatrix)
    n = size(got, 1)
    if uplo == 'U'
        for j in 1:n, i in (j + 1):n
            @inbounds got[i, j] == before[i, j] || return false
        end
    else
        for j in 1:n, i in 1:(j - 1)
            @inbounds got[i, j] == before[i, j] || return false
        end
    end
    return true
end

@testset "SYRK" begin
    A_single = Float32[1 2 3; 4 5 6]
    C_single = Float32[2 1; 3 4]

    A_batch = [Float32[1 2 0; 3 4 1], Float32[2 1 3; 0 1 2]]
    C_batch = [Float32[1 0; 0 1], Float32[2 1; 1 2]]
    A_batch3 = cat(A_batch..., dims = 3)
    C_batch3 = cat(C_batch..., dims = 3)

    alpha = Float32(1.5)
    beta = Float32(0.25)

    for (name, AT, sync) in backends
        # Metal's non-MPS path falls through to mul!, which writes the whole
        # square; and it has no SYRK at all. Recorded in KNOWN_ISSUES.md.
        metal = name == "Metal"

        @testset "[$name] uplo=$uplo trans=$trans" for uplo in ('L', 'U'), trans in ('N', 'T')
            A = trans == 'N' ? A_single : permutedims(A_single)
            C = C_single
            expected = _syrk_expected(trans, alpha, A, beta, C)

            Ad = _to_backend(AT, copy(A))
            Cd = _to_backend(AT, copy(C))
            NextLA.syrk!(uplo, trans, alpha, Ad, beta, Cd)
            sync(Cd)

            got = Array(Cd)
            @test _triangle_matches(uplo, got, expected)
            if metal
                @test_broken _other_triangle_untouched(uplo, got, C)
            else
                @test _other_triangle_untouched(uplo, got, C)
            end
        end

        @testset "[$name] strided views" begin
            Ap = Float32[9 8 7 6; 1 2 3 4; 5 6 7 8; 2 3 4 5]
            Cp = Float32[5 4 3; 2 1 0; 7 6 5]
            expected = _syrk_expected('N', alpha, Ap[2:3, 2:4], beta, Cp[2:3, 2:3])

            Ad = _to_backend(AT, copy(Ap))
            Cd = _to_backend(AT, copy(Cp))
            NextLA.syrk!('L', 'N', alpha, @view(Ad[2:3, 2:4]), beta, @view(Cd[2:3, 2:3]))
            sync(Cd)
            @test _triangle_matches('L', Array(Cd)[2:3, 2:3], expected)
        end

        # A missing ext file would silently fall back to CPU BLAS on a device
        # array and still pass every numeric check above.
        @testset "[$name] dispatches to its own backend" begin
            Ad = _to_backend(AT, copy(A_single))
            Cd = _to_backend(AT, copy(C_single))
            @test endswith(_method_file(NextLA.syrk!, 'L', 'N', alpha, Ad, beta, Cd),
                           _expected_syrk_file(name))
        end

        @testset "[$name] batched" begin
            expected = [_syrk_expected('N', alpha, A_batch[i], beta, C_batch[i])
                        for i in eachindex(A_batch)]
            expected3 = cat(expected..., dims = 3)

            # AMDGPU is the only backend with a native batched SYRK in both
            # layouts; everywhere else warns before falling back.
            warns_pointer = name != "AMDGPU"
            warns_strided = !(name in ("AMDGPU", "oneAPI"))

            Ad = _to_backend(AT, deepcopy(A_batch))
            Cd = _to_backend(AT, deepcopy(C_batch))
            if warns_pointer
                @test_logs (:warn,) match_mode = :any NextLA.syrk_batched!(
                    'L', 'N', alpha, Ad, beta, Cd)
            else
                NextLA.syrk_batched!('L', 'N', alpha, Ad, beta, Cd)
            end
            sync(first(Cd))
            for i in eachindex(expected)
                @test _triangle_matches('L', Array(Cd[i]), expected[i])
            end

            Ad3 = _to_backend(AT, copy(A_batch3))
            Cd3 = _to_backend(AT, copy(C_batch3))
            if warns_strided
                @test_logs (:warn,) match_mode = :any NextLA.syrk_batched!(
                    'L', 'N', alpha, Ad3, beta, Cd3)
            else
                NextLA.syrk_batched!('L', 'N', alpha, Ad3, beta, Cd3)
            end
            sync(Cd3)
            for i in axes(expected3, 3)
                @test _triangle_matches('L', Array(@view(Cd3[:, :, i])), @view(expected3[:, :, i]))
            end
        end

        @testset "[$name] rejects bad arguments" begin
            Ad = _to_backend(AT, copy(A_single))
            Cd = _to_backend(AT, copy(C_single))
            @test_throws ArgumentError NextLA.syrk!('X', 'N', alpha, Ad, beta, Cd)
            Cbad = _to_backend(AT, rand(Float32, 3, 2))
            @test_throws DimensionMismatch NextLA.syrk!('L', 'N', alpha, Ad, beta, Cbad)
        end
    end
end
