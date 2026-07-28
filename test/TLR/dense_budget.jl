# Workspace-budget compliance gate (ROADMAP milestone 3, step 0).
#
# The direct term helpers accept a scheduler budget that bounds their scratch.
# The correctness tests in `gemm.jl` run `budget=1` and a large budget, but they only
# assert the *result*, so a term that ignores the budget entirely still passes them.
# These tests assert the budget is actually honoured.
#
# The probe is `@allocated` around a direct call to each term: on CPU every scratch
# buffer is a plain `Array`, so heap bytes track scratch. A term that reads its budget
# allocates strictly less at a small budget than at a large one; a term that ignores it
# allocates the identical byte count across the whole range. No threshold is needed —
# the distinction is exact.
#
# Terms without a `budget` parameter are gated differently: their scratch must not grow
# with the tile grid, since an unbounded panel is the same defect without a knob.
#
# CPU-only by construction: `@allocated` counts Julia heap bytes, so on a GPU backend it
# would see the `CuArray` wrappers rather than the device allocation behind them. This
# costs no coverage — budget arithmetic lives in `schedule.jl` and the term bodies and is
# backend-independent, so CPU exercises the logic under test. Correctness across backends
# is covered by `gemm.jl`.

# Depends on `TLR/helpers.jl` (the `_TLRM` module alias and the `fill_random_tlr!`
# fixtures), which `runtests.jl` includes first.

# Bytes allocated by one call at `budget`, excluding compilation.
function term_bytes(f, budget)
    f(budget)
    return @allocated f(budget)
end

const _BUDGET_TINY = 1
const _BUDGET_HUGE = 128 * 1024 * 1024

# A boundary-tiled full-LR operand pair: every axis has a tail, so the interior, both
# panels and the corner are all non-empty.
function fulllr_operand_pair(::Type{T}, b::Int, r::Int, nt::Int; order=NextLA.TileRowMajor()) where {T}
    mA, kk, nB = b * nt + 3, b * nt + 5, b * nt + 2
    A = NextLA.PaddedFTLRMatrix(zeros(T, mA, kk), b, r; tile_order=order)
    B = NextLA.PaddedFTLRMatrix(zeros(T, kk, nB), b, r; tile_order=order)
    fill_random_tlr!(A, Array; seed=101)
    fill_random_tlr!(B, Array; seed=202)
    return (NextLA.TLRmodule.logical_operand(A, 'N'),
            NextLA.TLRmodule.logical_operand(B, 'N'),
            zeros(T, mA, nB))
end

function densediag_operand_pair(::Type{T}, b::Int, r::Int, nt::Int; order=NextLA.TileRowMajor()) where {T}
    n = b * nt + 3
    A = NextLA.TLRMatrix(zeros(T, n, n), b, r; tile_order=order)
    B = NextLA.TLRMatrix(zeros(T, n, n), b, r; tile_order=order)
    fill_random_tlr!(A, Array; seed=101)
    fill_random_tlr!(B, Array; seed=202)
    return (NextLA.TLRmodule.logical_operand(A, 'N'),
            NextLA.TLRmodule.logical_operand(B, 'N'),
            zeros(T, n, n))
end

@testset "GEMM workspace budget compliance" begin
    @testset "budget-accepting terms respond to their scheduler budget" begin
        @testset "PaddedFTLRMatrix (full low-rank)" begin
            LA, LB, C = fulllr_operand_pair(Float64, 8, 4, 10)

            @testset "_offdiag_offdiag_gemm!" begin
                f = bud -> _TLRM._offdiag_offdiag_gemm!(C, LA, LB; alpha=1.0, beta=1.0, budget=bud)
                @test term_bytes(f, _BUDGET_TINY) < term_bytes(f, _BUDGET_HUGE)
            end

            @testset "tlr_gemm_rpanel_by_bpanel" begin
                f = bud -> _TLRM.tlr_gemm_rpanel_by_bpanel(C, LA, LB, 1.0; beta=1.0, budget=bud)
                @test term_bytes(f, _BUDGET_TINY) < term_bytes(f, _BUDGET_HUGE)
            end

            # These two used to accept `budget::Int` and never read it, sizing S/T from the
            # full grid unconditionally — byte-identical at every budget, while their
            # dense-diagonal counterparts below honoured it. Both now use the direct
            # budgeted regular core.
            @testset "tlr_gemm_int_by_rpanel" begin
                f = bud -> _TLRM.tlr_gemm_int_by_rpanel(C, LA, LB, 1.0; beta=1.0, budget=bud)
                @test term_bytes(f, _BUDGET_TINY) < term_bytes(f, _BUDGET_HUGE)
            end

            @testset "tlr_gemm_bpanel_by_int" begin
                f = bud -> _TLRM.tlr_gemm_bpanel_by_int(C, LA, LB, 1.0; beta=1.0, budget=bud)
                @test term_bytes(f, _BUDGET_TINY) < term_bytes(f, _BUDGET_HUGE)
            end

            # Gained a budget in milestone 3, step 4: the reduction over the shared panel
            # index is now blocked (it was O(q_c) scratch with no knob at all).
            @testset "tlr_gemm_bpanel_by_rpanel" begin
                f = bud -> _TLRM.tlr_gemm_bpanel_by_rpanel(C, LA, LB, 1.0; beta=1.0, budget=bud)
                @test term_bytes(f, _BUDGET_TINY) < term_bytes(f, _BUDGET_HUGE)
            end

            # Also gained a budget in step 4: single-tile reductions batched over one free
            # axis, previously O(q_m) / O(q_n) scratch with no knob.
            @testset "tlr_gemm_rpanel_by_corner" begin
                f = bud -> _TLRM.tlr_gemm_rpanel_by_corner(C, LA, LB, 1.0; beta=1.0, budget=bud)
                @test term_bytes(f, _BUDGET_TINY) < term_bytes(f, _BUDGET_HUGE)
            end

            @testset "tlr_gemm_corner_by_bpanel" begin
                f = bud -> _TLRM.tlr_gemm_corner_by_bpanel(C, LA, LB, 1.0; beta=1.0, budget=bud)
                @test term_bytes(f, _BUDGET_TINY) < term_bytes(f, _BUDGET_HUGE)
            end
        end

        @testset "TLRMatrix" begin
            LA, LB, C = densediag_operand_pair(Float64, 8, 4, 10)

            @testset "tlr_gemm_int_by_int" begin
                f = bud -> _TLRM.tlr_gemm_int_by_int(C, LA, LB, 1.0, 1.0; budget=bud)
                @test term_bytes(f, _BUDGET_TINY) < term_bytes(f, _BUDGET_HUGE)
            end

            @testset "tlr_gemm_int_by_rpanel" begin
                f = bud -> _TLRM.tlr_gemm_int_by_rpanel(C, LA, LB, 1.0; beta=1.0, budget=bud)
                @test term_bytes(f, _BUDGET_TINY) < term_bytes(f, _BUDGET_HUGE)
            end

            @testset "tlr_gemm_bpanel_by_int" begin
                f = bud -> _TLRM.tlr_gemm_bpanel_by_int(C, LA, LB, 1.0; beta=1.0, budget=bud)
                @test term_bytes(f, _BUDGET_TINY) < term_bytes(f, _BUDGET_HUGE)
            end

            # The dense-corner (two-stage) lowerings were unbudgeted too — step 4 blocked
            # their free axis without materialising an identity factor for the dense leaf.
            @testset "tlr_gemm_rpanel_by_corner" begin
                f = bud -> _TLRM.tlr_gemm_rpanel_by_corner(C, LA, LB, 1.0; beta=1.0, budget=bud)
                @test term_bytes(f, _BUDGET_TINY) < term_bytes(f, _BUDGET_HUGE)
            end

            @testset "tlr_gemm_corner_by_bpanel" begin
                f = bud -> _TLRM.tlr_gemm_corner_by_bpanel(C, LA, LB, 1.0; beta=1.0, budget=bud)
                @test term_bytes(f, _BUDGET_TINY) < term_bytes(f, _BUDGET_HUGE)
            end
        end
    end

    # `corner_by_corner` is the only term that legitimately needs no budget: the corner is
    # a single tile in every geometry, so its scratch is already grid-independent. Every
    # other term is budgeted as of step 4.
    @testset "corner_by_corner needs no budget" begin
        LA_s, LB_s, C_s = fulllr_operand_pair(Float64, 8, 4, 6)
        LA_l, LB_l, C_l = fulllr_operand_pair(Float64, 8, 4, 18)

        @testset "tlr_gemm_corner_by_corner" begin
            _TLRM.tlr_gemm_corner_by_corner(C_s, LA_s, LB_s, 1.0; beta=1.0)
            _TLRM.tlr_gemm_corner_by_corner(C_l, LA_l, LB_l, 1.0; beta=1.0)
            bs = @allocated _TLRM.tlr_gemm_corner_by_corner(C_s, LA_s, LB_s, 1.0; beta=1.0)
            bl = @allocated _TLRM.tlr_gemm_corner_by_corner(C_l, LA_l, LB_l, 1.0; beta=1.0)
            @test bl <= 2 * bs
        end
    end
end
