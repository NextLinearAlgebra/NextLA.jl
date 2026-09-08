export recsyrk!

# Recursive blocked symmetric rank-k update: C := alpha*A*Aᵀ + beta*C, split
# 2x2 and driven down to a vendor SYRK at the leaves. The off-diagonal block is
# not symmetric, so it takes a GEMM while the two diagonal blocks recurse.
#
# Source: vicki-development @ 694d019, by Vicki <vickicar@mit.edu>. Ported the
# same way recgemm.jl was, and for the same reasons:
#
#  - include("wrappers.jl") is not carried over. That file declares its own
#    gemm!, gemmEx!, potrf!, trsm!, trmm! and syrk! as thin CUBLAS/rocBLAS
#    wrappers behind a top-level `using CUDA` and `using AMDGPU`. The vendor
#    code now lives in ext/ as package extensions, those wrappers have no CPU
#    method at all, and four of the six names belong to other features.
#  - `using CUDA` goes with it, and `using StochasticRounding` was imported and
#    never referenced -- as on the branch's other files.
#  - _syrk_dispatch! was typed on CUDA.StridedCuArray, which is what made the
#    routine CUDA-only. It takes AbstractMatrix here and reaches the device
#    through syrk!/gemm!/gemmEx!, all of which dispatch per backend.
#
# The vendor-pinned siblings are worth a note: vicki-development-amd and its
# three relatives add separate AMDGPU and oneAPI methods to this file, and
# vicki-development-intel-v2 collapses those two into one AnyGPUArray method.
# None of that is needed once the vendor split lives in ext/, but -intel-v2 is
# the one place that generalisation was written down.
#
# The branch's GEMM calls, gemm!('N', 'T', alpha, A, B, beta, C), are kept as
# written: this package's gemm! has that BLAS-order form too
# (src/gemm/gemmEx.jl), forwarding to the output-first method with the second
# operand transposed -- the whole point of the off-diagonal update. 'T', not
# 'C': SYRK does not conjugate.

# Float16 storage, accumulated in Float32 and rounded once on store. gemmEx!
# does that on the CPU, CUDA and AMDGPU. oneAPI has no gemmEx!, so
# ext/oneapi/syrk.jl adds a method through gemm!, as the branch's
# recsyrk_dev.jl did with oneMKL.gemm!.
# A SYRK leaves the other triangle of C as it was; a GEMM writes all of it. So the
# half-precision SYRK computes the full product in a copy and stores back only the
# uplo triangle -- twice the arithmetic a triangular update needs and one copy of C,
# where there is no Float16 SYRK to call.
@inline _in_triangle(uplo::Char, i, j) = uplo == 'L' ? i >= j : i <= j

function _syrk_half_tri!(uplo::Char, alpha, A, beta, C)
    W = copy(C)
    _syrk_half_gemm!(alpha, A, A, beta, W)
    # one fused broadcast, so the mask is computed on the device, not as a host BitMatrix
    C .= ifelse.(_in_triangle.(uplo, axes(C, 1), axes(C, 2)'), W, C)
    return C
end

_syrk_half_gemm!(alpha, A, B, beta, C) =
    gemmEx!('N', 'T', alpha, A, B, beta, C; compute_type=Float32)

function _syrk_dispatch!(
    op::Symbol,
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::AbstractMatrix,
    uplo::Char='L'
)
    TC = eltype(C)
    TA = eltype(A)

    if op === :SYRK
        if TA == TC && TC in (Float32, Float64, ComplexF32, ComplexF64)
            syrk!(uplo, 'N', TC(alpha), A, TC(beta), C)
        elseif TA == Float16 && TC in (Float16, Float32)
            _syrk_half_tri!(uplo, alpha, A, beta, C)
        elseif TC in (Float32, Float64, ComplexF32, ComplexF64)
            # A and C differ in type: compute in C's type, as the branch's
            # recsyrk_dev.jl does. A wider C takes A widened exactly, a
            # narrower one takes A rounded once; C itself is never rounded.
            # The Float32 fallback below used to take these, rounding a
            # Float64 C to Float32 and throwing on mismatched complex types.
            syrk!(uplo, 'N', TC(alpha), TC.(A), TC(beta), C)
        else
            compute_type = Float32

            C_temp = (TC == compute_type) ? C : compute_type.(C)

            if TA == Float32
                syrk!(uplo, 'N', compute_type(alpha), A, compute_type(beta), C_temp)
            elseif TA == Float16
                _syrk_half_tri!(uplo, alpha, A, beta, C_temp)
            else
                A_temp = compute_type.(A)
                syrk!(uplo, 'N', compute_type(alpha), A_temp, compute_type(beta), C_temp)
            end

            if C !== C_temp
                copy!(C, C_temp)
            end
        end

    elseif op === :GEMM
        TB = eltype(B)
        if TA == TB == TC && TC in (Float32, Float64, ComplexF32, ComplexF64)
            gemm!('N', 'T', TC(alpha), A, B, TC(beta), C)
        elseif TA == Float16 && TB == Float16 && TC in (Float16, Float32)
            _syrk_half_gemm!(alpha, A, B, beta, C)
        else
            A_final = (TA == TC) ? A : TC.(A)
            B_final = (TB == TC) ? B : TC.(B)
            if TC in (Float32, Float64, ComplexF32, ComplexF64)
                gemm!('N', 'T', TC(alpha), A_final, B_final, TC(beta), C)
            else
                _syrk_half_gemm!(alpha, A_final, B_final, beta, C)
            end
        end
    end
end

"""
    _recsyrk_impl!(alpha, A, beta, C, threshold, uplo; parallel)

Internal implementation for nested recursive symmetric rank-k updates of the `uplo`
triangle of `C`.
Recursively divides matrices into sub-blocks and applies updates in-place,
falling back to standard hardware routines using the dispatch helper at the
base case.
"""
function _recsyrk_impl!(
    alpha::Number, A::AbstractMatrix, beta::Number, C::AbstractMatrix,
    threshold::Int, uplo::Char='L'; parallel::Bool
)
    n = size(C, 1)
    if n <= threshold
        _syrk_dispatch!(:SYRK, alpha, A, A, beta, C, uplo)
        return
    end

    n1 = _rec_split(n)
    m = size(A, 2)

    A1 = @view A[1:n1, 1:m]; A2 = @view A[n1+1:end, 1:m]
    C11 = @view C[1:n1, 1:n1]; C22 = @view C[n1+1:end, n1+1:end]

    # The off-diagonal block of the stored triangle: A2 * A1ᵀ below, A1 * A2ᵀ above.
    if uplo == 'L'
        _syrk_dispatch!(:GEMM, alpha, A2, A1, beta, @view C[n1+1:end, 1:n1])
    else
        _syrk_dispatch!(:GEMM, alpha, A1, A2, beta, @view C[1:n1, n1+1:end])
    end

    if parallel
        @sync begin
            Threads.@spawn _recsyrk_impl!(alpha, A1, beta, C11, threshold, uplo, parallel=false)
            Threads.@spawn _recsyrk_impl!(alpha, A2, beta, C22, threshold, uplo, parallel=false)
        end
    else
        _recsyrk_impl!(alpha, A1, beta, C11, threshold, uplo, parallel=false)
        _recsyrk_impl!(alpha, A2, beta, C22, threshold, uplo, parallel=false)
    end
end

const RECSYRK_PARALLEL_THRESHOLD = 4096

"""
    recsyrk!(alpha, A, beta, C, threshold=256; uplo='L')

Performs an in-place, nested recursive block symmetric rank-k update
(`C = alpha*A*Aᵀ + beta*C`), writing the `uplo` triangle, `'L'` or `'U'`. Falls back
to standard hardware routines using the dispatch helper at the specified base case
threshold.
"""
function recsyrk!(
    alpha::Number, A::AbstractMatrix, beta::Number, C::AbstractMatrix, threshold::Int=256;
    uplo::Char='L'
)
    uplo in ('L', 'U') || throw(ArgumentError("uplo must be 'L' or 'U', got '$uplo'"))
    # Spawn when the first half has the work, the rule the mixed-precision SYRK uses.
    should_parallelize = _rec_split(size(C, 1)) > RECSYRK_PARALLEL_THRESHOLD
    _recsyrk_impl!(alpha, A, beta, C, threshold, uplo, parallel=should_parallelize)
end
