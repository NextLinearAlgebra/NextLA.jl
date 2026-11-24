export recgemm!

# Recursive GEMM over the mixed-precision containers in src/mxp/.
#
# Source: vicki-development @ 694d019, by Vicki <vickicar@mit.edu>. Only the
# three routines below are taken; the branch file also began with
# include("wrappers.jl"), which is not carried over. That file declares its own
# gemm!, gemmEx!, potrf!, trsm!, trmm! and syrk! as thin CUBLAS/rocBLAS
# wrappers behind a top-level `using CUDA` and `using AMDGPU`. It cannot be
# used here: the vendor code now lives in ext/ as package extensions, those
# wrappers have no CPU method at all, and four of the six names belong to other
# features.
#
# The two gemm! call sites are therefore adapted to the signature this package
# provides, gemm!(C, A, B, alpha, beta) in gemmEx.jl, rather than the BLAS
# argument order the branch used. Both calls pass 'N', 'N', so no transpose is
# lost. The gemmEx! calls are unchanged — that signature already is BLAS-ordered
# here and matches verbatim.
#
# The branch's second recgemm! method dispatches on FullMixedPrec, so it does
# not live here: mixed precision owns its own tree under src/mxp/, one folder
# per feature, and that method arrives with it. This file is the dense half and
# depends on nothing above it.

# The compute type is stated rather than inferred. Inferred from the storage,
# a Float16 triple would accumulate in Float16.
# Float32 is the accumulation _tensor_core_gemm_supported blesses for Float16
# and BFloat16 storage on both CUDA and AMDGPU, and it is what the CPU wants
# anyway: wider than the storage, rounded once on store.
# Complex is admitted alongside the real types. Without it a complex triple falls
# to the else branch below, which states compute_type=Float32 -- and Float32.(A)
# on a complex matrix is an InexactError, so recgemm! did not work on complex at
# all. mul! handles it; nothing else in the library is real-only.
function _gemm_dispatch!(alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::AbstractMatrix)
    TA, TB, TC = eltype(A), eltype(B), eltype(C)

    if TA == TB == TC && TC in (Float32, Float64, ComplexF32, ComplexF64)
        gemm!(C, A, B, TC(alpha), TC(beta))
    elseif TA == Float16 && TB == Float16 && TC in (Float16, Float32)
        gemmEx!('N', 'N', alpha, A, B, beta, C; compute_type=Float32)
    else
        # Converting the operands to the destination type first throws away
        # precision the destination never needed to lose: Float32 operands into
        # a Float16 C were rounded to Float16 and only then multiplied. Widen
        # instead -- accumulate at least as wide as the operands and round once,
        # on store, which is what gemmEx! does natively.
        W = promote_type(TA, TB)
        if TC in (Float32, Float64, ComplexF32, ComplexF64) && promote_type(W, TC) == TC
            # C is at least as wide as both operands (Float32 into Float64, or a
            # Float32 operand beside a Float64 one into Float64). Widening is
            # exact, so convert whichever operand differs from C and make one
            # call in C's type. Passing mixed types on instead sends mul! to
            # Julia's generic loop rather than BLAS -- 10x slower at n=1024 for
            # Float32 x Float64 -- and computing in W gave the result only W's
            # accuracy, while cuBLAS refuses a Float64 C under Float32 compute.
            gemm!(C, TA == TC ? A : TC.(A), TB == TC ? B : TC.(B), TC(alpha), TC(beta))
        else
            # C is narrower than the operands (Float64 into Float32, Float32 into
            # Float16). The CPU accumulates at the operands' width and rounds
            # once on store. cuBLAS refuses those pairings (NOT_SUPPORTED), so a
            # device rounds the operands to C's type first and takes the
            # same-type or Float16 arm, as vicki-development did.
            if promote_type(W, TC) != TC && !(get_backend(C) isa KernelAbstractions.CPU)
                return _gemm_dispatch!(alpha, TC.(A), TC.(B), beta, C)
            end
            # Accumulate in at least Float32: BFloat16 and Float8 carry too few
            # bits to hold a running sum, as Float16 does.
            gemmEx!('N', 'N', alpha, A, B, beta, C;
                    compute_type=promote_type(W, Float32))
        end
    end
end

function recgemm!(alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::AbstractMatrix)
    _gemm_dispatch!(alpha, A, B, beta, C)
end
