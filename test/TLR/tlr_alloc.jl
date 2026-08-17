# Allocation-regression coverage for the canonical TLR-output GEMM's hot
# sampler: the
# canonical `gemm!(C::CompressedFTLRMatrix, A::CompressedFTLRMatrix, B::CompressedFTLRMatrix; ...)` hot sampler
# must not rebuild a `Vector`-of-views/device-pointer-array per ARA pass.
#
# CUDA-only: Julia's `@allocated` counts host heap bytes but cannot see a GPU
# backend's device allocations. This file therefore adds explicit coverage for
# that device-side cost. AMDGPU is
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
# array on every ARA pass independent of anything in `padded_result/run_coupling.jl`, and
# a few same-call operands that are run-owned but not reshape-representable
# as a strided batch (coupling-sketch formation in `RunCoupling`) are left on
# the `Vector`-of-views path -- both are documented, deliberate residuals,
# not oversights this test should fail on.
# The threshold below is instead
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
    _cuda_allocated(f) = Core.eval(@__MODULE__, :(CUDA.@allocated $f()))

    function _tlr_alloc_bytes(ArrayType, synchronize;
                             transA, transB, rA=3, rB=4, seed=1200)
        T = Float64
        A, B, C = _canonical_tlr_fixture(T, ArrayType; rA, rB, seed)
        f = () -> NextLA.gemm!(
            C, A, B; alpha=1.2, beta=0.0, transA, transB,
            tol=1e-7, rel=true, eps_rel=1e-7, r_required=3, block=4,
        )
        f()  # warm up: first call pays JIT/dispatch-resolution cost
        synchronize(C.outer.data)
        bytes = _cuda_allocated(f)
        synchronize(C.outer.data)
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

if isdefined(@__MODULE__, :CUDA)
    function _tlr_alloc_bytes_capacity(ArrayType, synchronize; minimum::Bool)
        T = Float64
        A, B, C = _canonical_tlr_fixture(
            T, ArrayType; rA=2, rB=6, seed=1380)
        bytes = minimum ?
            NextLA.tlr_gemm_minimum_workspace_bytes(
                C, A, B; transA='N', transB='T', block=4) :
            NextLA.tlr_gemm_maximum_workspace_bytes(
                C, A, B; transA='N', transB='T', block=4)
        workspace = NextLA.TLRGemmWorkspace(
            C, A, B; bytes, transA='N', transB='T', block=4)
        f = () -> NextLA.gemm!(
            C, A, B; transA='N', transB='T', tol=1e-7, rel=true,
            eps_rel=1e-7, r_required=3, block=4, workspace)
        f()
        synchronize(C.outer.data)
        allocated = _cuda_allocated(f)
        synchronize(C.outer.data)
        return allocated
    end
end

@testset "rolling admission does not allocate per wave on CUDA" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        backend_name == "CUDA" || continue
        full = _tlr_alloc_bytes_capacity(ArrayType, synchronize; minimum=false)
        rolling = _tlr_alloc_bytes_capacity(ArrayType, synchronize; minimum=true)
        # Capacity one performs three admission/finalization waves per lane.
        # Its device allocation remains within the same fixed residual budget
        # as a complete-lane run rather than scaling with those waves.
        @test rolling <= full + _TLR_ALLOC_BUDGET
    end
end

# Cross-run arena reuse: the canonical driver builds one ARARunArena before
# its traversal loop and resets it once per row/column run instead of letting
# every run's RunCoupling and ARAWorkspace allocate fresh device storage.
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
        backend = _canonical_tlr_backend(ArrayType)
        A = NextLA.CompressedFTLRMatrix(backend, T, qm * b, qk * b, b, fill(rA, qm, qk);
                                        execution_rank_policy=:exact)
        B = NextLA.CompressedFTLRMatrix(backend, T, qk * b, qn * b, b, fill(rB, qk, qn);
                                        execution_rank_policy=:exact)
        C = NextLA.CompressedFTLRMatrix(backend, T, qm * b, qn * b, b, zeros(Int, qm, qn);
                                        execution_ranks=fill(16, qm, qn))
        fill_random_tlr!(A, ArrayType; seed=seed + 1)
        fill_random_tlr!(B, ArrayType; seed=seed + 2)
        workspace = NextLA.TLRGemmWorkspace(C, A, B; block=4)
        f = () -> NextLA.gemm!(
            C, A, B; alpha=1.2, beta=0.0, transA='N', transB='N',
            tol=1e-7, rel=true, eps_rel=1e-7, r_required=3, block=4,
            workspace,
        )
        f()
        synchronize(C.outer.data)
        bytes = _cuda_allocated(f)
        synchronize(C.outer.data)
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
