# Allocation-regression coverage for the canonical TLR-output GEMM's hot
# sampler (docs/TODO.md's dated worklog, "R3 performance closure"): the
# canonical `gemm!(C::TLRMatrix, A::TLRMatrix, B::TLRMatrix; ...)` hot sampler
# must not rebuild a `Vector`-of-views/device-pointer-array per ARA pass.
#
# CUDA-only: `dense_budget.jl`'s `@allocated`-based `term_bytes` helper counts
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
# array on every ARA pass independent of anything in `tlr_result/run_coupling.jl`, and
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
    const _TLR_ALLOC_BUDGET = 500_000  # bytes; see rationale above

    function _tlr_alloc_bytes(ArrayType, synchronize;
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

@testset "canonical row-major TLR-result gemm! allocation" begin
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
                    bytes = _tlr_alloc_bytes(ArrayType, synchronize; kw...)
                    @test bytes <= _TLR_ALLOC_BUDGET
                end
            end
        end
    end
end

# Cross-run arena reuse (docs/TODO.md's dated worklog, "R4 arena -- reusable
# run/ARA workspace"): the canonical driver now builds one ARARunArena before
# its traversal loop and resets it once per row/column run instead of letting
# every run's ColumnRunCoupling/RowRightRunCoupling/RowLeftRunCoupling and
# ARAWorkspace allocate fresh device storage.
#
# A naive "grow the whole grid" comparison can't isolate this: for a square
# matrix, growing n grows both the per-run size (nmember, scales with the
# arena's own budget -- expected to grow) and the loop trip count together,
# so a bigger allocation total doesn't distinguish "the arena legitimately
# grew because a run holds more members" from "a regression reintroduced
# per-iteration allocation." This uses a rectangular A (more row-tiles, same
# column-tile count) instead: for the NN family the driver's loop runs once
# per row-tile (`for i in 1:qm`) with a *fixed* per-run shape (nmember=qn,
# qk both held constant), so only the number of loop iterations changes.
# With reuse intact, allocated bytes should stay close to flat as qm grows;
# a regression back to per-run allocation would scale with qm instead.
if isdefined(@__MODULE__, :CUDA)
    function _tlr_alloc_bytes_tall(ArrayType, synchronize; qm::Int, rA=3, rB=4, seed=1200)
        T = Float64
        b = 16
        qk = qn = 3
        A = NextLA.TLRMatrix(ArrayType(zeros(T, qm * b, qk * b)), b, rA)
        B = NextLA.TLRMatrix(ArrayType(zeros(T, qk * b, qn * b)), b, rB)
        C = NextLA.TLRMatrix(ArrayType(zeros(T, qm * b, qn * b)), b, 16)
        fill_random_tlr!(A, ArrayType; seed=seed + 1)
        fill_random_tlr!(B, ArrayType; seed=seed + 2)
        workspace = NextLA.TLRGemmWorkspace(C, A, B; block=4)
        f = () -> NextLA.TLRmodule.gemm!(
            C, A, B; alpha=1.2, beta=0.0, transA='N', transB='N',
            tol=1e-7, rel=true, eps_rel=1e-7, r_required=3, block=4,
            workspace,
        )
        f()
        synchronize(C.int_U)
        bytes = CUDA.@allocated f()
        synchronize(C.int_U)
        return bytes
    end
end

@testset "canonical row-major TLR-result gemm! cross-run arena reuse" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        backend_name == "CUDA" || continue
        @testset "$backend_name" begin
            few = _tlr_alloc_bytes_tall(ArrayType, synchronize; qm=3)
            many = _tlr_alloc_bytes_tall(ArrayType, synchronize; qm=9)
            # Same per-run shape (nmember=qn=3, qk=3) both times; only the
            # number of loop iterations (qm) triples. A per-run-allocating
            # driver would scale allocated bytes with qm; with the arena
            # reused across the loop, it should stay within one run's budget
            # of the qm=3 measurement regardless of qm.
            @test many <= few + _TLR_ALLOC_BUDGET
        end
    end
end
