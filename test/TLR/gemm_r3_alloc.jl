# Allocation-regression coverage for the R3 performance-closure work
# (docs/TODO.md, "R3 performance closure"): the canonical `gemm!(C::TLRMatrix,
# A::TLRMatrix, B::TLRMatrix; ...)` hot sampler must not rebuild a
# `Vector`-of-views/device-pointer-array per ARA pass.
#
# CUDA-only: `gemm_budget.jl`'s `@allocated`-based `term_bytes` helper counts
# Julia heap bytes and is explicitly CPU-only for exactly this reason (a GPU
# backend's device allocations don't show up there) -- this file is new
# coverage for the device-side cost `term_bytes` cannot see. AMDGPU is
# skipped, matching the rest of the TLR suite (its NextLA extension does not
# currently precompile on this project's machines).
#
# `CUDA.@allocated` is a macro, so it needs `CUDA` to be a statically
# resolvable name -- unlike the rest of this test suite's `Base.require`
# pattern (fine for plain function calls, e.g. `backend_test_helpers.jl`'s
# `_device_pointer_batch`), a runtime module reference does not work as the
# receiver of `.@macro` syntax inside a later-defined function. `using CUDA`
# below is therefore wrapped in `@eval` and a `try`/`catch` so this file stays
# loadable (as a no-op) on machines without CUDA installed.
#
# The budget is not zero and is not meant to be: `ara_cholesky_pass!`'s
# `trsm_batched!`/`potrfBatched!` calls allocate a transient device pointer
# array on every ARA pass independent of anything in `ara/tile_apply.jl`, and
# a few same-call operands that are run-owned but not reshape-representable
# as a strided batch (T-formation/W-formation in `RowRightRunCoupling`/
# `RowLeftRunCoupling`) are left on the `Vector`-of-views path -- both are
# documented, deliberate residuals in `docs/TODO.md`'s workspace contract,
# not oversights this test should fail on. The threshold below is instead
# calibrated with generous headroom (roughly 3x) over the measured
# allocation for this fixture, so it catches a regression back toward
# rebuilding *every* batched-GEMM operand's pointer list per pass -- which
# scales with contraction-tile count and would be many times larger, not a
# reintroduction of the small, bounded, already-documented residual.

try
    @eval using CUDA
catch
end

if isdefined(@__MODULE__, :CUDA)
    const _R3_ALLOC_BUDGET = 500_000  # bytes; see rationale above

    function _r3_alloc_bytes(ArrayType, synchronize;
                             transA, transB, rA=3, rB=4, seed=1200)
        T = Float64
        A, B, C = _canonical_tlr_fixture(T, ArrayType; rA, rB, seed)
        f = () -> NextLA.TLRmodule.gemm!(
            C, A, B; alpha=1.2, beta=0.0, transA, transB,
            tol=1e-7, rel=true, eps_rel=1e-7, r_required=3, block=4,
        )
        f()  # warm up: first call pays JIT/dispatch-resolution cost
        synchronize(C.int_U)
        bytes = CUDA.@allocated f()
        synchronize(C.int_U)
        return bytes
    end
end

@testset "canonical row-major TLR-result gemm! allocation (R3)" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        backend_name == "CUDA" || continue
        @testset "$backend_name" begin
            cases = [
                ("NN", (transA='N', transB='N')),
                ("NT right-selected", (transA='N', transB='T', rA=2, rB=6)),
                ("NT left-selected", (transA='N', transB='T', rA=6, rB=2)),
                ("TT", (transA='T', transB='T')),
            ]
            for (name, kw) in cases
                @testset "$name" begin
                    bytes = _r3_alloc_bytes(ArrayType, synchronize; kw...)
                    @test bytes <= _R3_ALLOC_BUDGET
                end
            end
        end
    end
end
