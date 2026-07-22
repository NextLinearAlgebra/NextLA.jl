using LinearAlgebra
using CUDA
using CUDA.CUSOLVER
using StochasticRounding
include("symmmixedprec.jl")
include("recmixedprectri.jl")
include("trsm.jl")
include("trmm.jl")
include("matmul.jl")
include("rectrxm.jl")
include("recsyrk.jl")
include("wrappers.jl")

# =============================================================================
# FullMixedPrec: hierarchical mixed-precision structure for full dense matrices
# =============================================================================

"""
    FullMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}

A hierarchical, recursive mixed-precision data structure that maps to full dense
square matrices. It recursively partitions the matrix into two diagonal sub-blocks
(`A11`, `A22`) and two dense off-diagonal blocks (`A12`, upper-right; `A21`,
lower-left). Unlike `SymmMixedPrec`/`TriMixedPrec`, both off-diagonal blocks are
explicitly stored since a general matrix has no symmetry to exploit.
"""
struct FullMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}
    A11::Union{FullMixedPrec{T_Base}, Nothing}
    A22::Union{FullMixedPrec{T_Base}, Nothing}
    A12::Union{AbstractMatrix, Nothing}
    A21::Union{AbstractMatrix, Nothing}
    A12_scale::Union{Float32, Nothing}
    A21_scale::Union{Float32, Nothing}
    base_scale::Union{Float32, Nothing}
    BaseCase::Union{AbstractMatrix{T_Base}, Nothing}
    sz::Tuple{Int, Int}
end

"""
    _fp16_quantize_block(block::AbstractMatrix)

Applies the same `Float16` dynamic quantization used by `SymmMixedPrec` and
`TriMixedPrec`: if the maximum absolute value exceeds `65504.0f0`, computes a
`Float32` scaling factor, scales, clamps, and rounds; otherwise converts directly.
Returns `(quantized_matrix, scale)` where `scale` is `nothing` when no scaling
was required.
"""
function _fp16_quantize_block(block::AbstractMatrix)
    FP16_MAX_VAL = 65504.0f0
    alpha = maximum(abs, block)

    if alpha > FP16_MAX_VAL
        scale = Float32(alpha / FP16_MAX_VAL)
        quantized = similar(block, Float16, size(block))
        @. quantized = Float16(round(clamp(block / scale, -FP16_MAX_VAL, FP16_MAX_VAL)))
        return quantized, scale
    else
        quantized = similar(block, Float16, size(block))
        quantized .= block
        return quantized, nothing
    end
end

"""
    FullMixedPrec(A::AbstractMatrix; precisions::Vector{DataType})

Constructs a `FullMixedPrec` representation of the full dense square matrix `A`.

Uses a base-2 recursive splitting scheme to partition the matrix into two diagonal
sub-blocks and two rectangular off-diagonal blocks, mapping the given `precisions`
to each depth level. For blocks stored as `Float16`, per-block dynamic quantization
detects values exceeding `65504.0f0`, computes a scaling factor, and applies
clamping to avoid numerical overflow.
"""
function FullMixedPrec(
    A::AbstractMatrix;
    precisions::Vector{DataType}
)
    n = size(A, 1)
    @assert n == size(A, 2) "A must be square"

    if length(precisions) == 1 || n <= 1
        T_Base = precisions[1]
        local base_matrix
        local base_scale

        if T_Base == Float16
            base_matrix, base_scale = _fp16_quantize_block(A)
        else
            base_matrix = similar(A, T_Base, size(A))
            base_matrix .= A
            base_scale = nothing
        end

        return FullMixedPrec{T_Base}(
            nothing, nothing, nothing, nothing,
            nothing, nothing, base_scale, base_matrix, (n, n)
        )
    end

    mid = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))

    T_OffDiag = precisions[1]
    remaining_precisions = precisions[2:end]

    A11 = FullMixedPrec(view(A, 1:mid, 1:mid); precisions=remaining_precisions)
    A22 = FullMixedPrec(view(A, mid+1:n, mid+1:n); precisions=remaining_precisions)

    A12_view = view(A, 1:mid, mid+1:n)
    A21_view = view(A, mid+1:n, 1:mid)

    local A12_matrix, A21_matrix
    local A12_scale = nothing
    local A21_scale = nothing

    if T_OffDiag == Float16
        A12_matrix, A12_scale = _fp16_quantize_block(A12_view)
        A21_matrix, A21_scale = _fp16_quantize_block(A21_view)
    else
        A12_matrix = similar(A, T_OffDiag, size(A12_view))
        A12_matrix .= A12_view
        A21_matrix = similar(A, T_OffDiag, size(A21_view))
        A21_matrix .= A21_view
    end

    T_Final_Base = precisions[end]
    return FullMixedPrec{T_Final_Base}(
        A11, A22, A12_matrix, A21_matrix,
        A12_scale, A21_scale, nothing, nothing, (n, n)
    )
end

function Base.size(A::FullMixedPrec)
    return A.sz
end

function Base.getindex(A::FullMixedPrec{T_Base}, i::Int, j::Int) where {T_Base}
    if A.BaseCase !== nothing
        return A.BaseCase[i, j]
    end

    mid = size(A.A11, 1)

    if i <= mid && j <= mid
        return A.A11[i, j]
    elseif i > mid && j > mid
        return A.A22[i - mid, j - mid]
    elseif i > mid && j <= mid
        return A.A21[i - mid, j]
    else
        return A.A12[i, j - mid]
    end
end

function Base.sizeof(A::FullMixedPrec)
    if A.BaseCase !== nothing
        return sizeof(A.BaseCase)
    end
    return sizeof(A.A11) + sizeof(A.A22) + sizeof(A.A12) + sizeof(A.A21)
end

"""
    TriMixedPrec(A::FullMixedPrec{T_Base}, uplo::Char)

Dynamically converts an existing `FullMixedPrec` structure into a `TriMixedPrec`
view of one of its triangles ('L' or 'U') without copying block storage. This is
used after factorizing a diagonal block in-place, so that the packed L/U factors
can be fed to `unified_rectrxm!` for the panel solves.
"""
function TriMixedPrec(A::FullMixedPrec{T_Base}, uplo::Char) where {T_Base}
    if A.BaseCase !== nothing
        return TriMixedPrec{T_Base}(
            nothing, nothing, nothing,
            nothing, A.base_scale, A.BaseCase,
            uplo, A.sz
        )
    end

    OffDiag = (uplo == 'L') ? A.A21 : A.A12
    offDiag_scale = (uplo == 'L') ? A.A21_scale : A.A12_scale

    return TriMixedPrec{T_Base}(
        TriMixedPrec(A.A11, uplo),
        TriMixedPrec(A.A22, uplo),
        OffDiag,
        offDiag_scale,
        nothing,
        nothing,
        uplo,
        A.sz
    )
end

"""
    reconstruct_matrix(A::FullMixedPrec{T_Base})

Reconstructs a full dense matrix from the mixed-precision recursive block
structure `A`, applying any stored quantization scales. Used primarily for
validation and returning to standard dense formats.
"""
function reconstruct_matrix(A::FullMixedPrec{T_Base}) where {T_Base}
    if A.BaseCase !== nothing
        C = CuArray{T_Base}(undef, size(A.BaseCase))
        C .= A.BaseCase
        if A.base_scale !== nothing
            C .*= A.base_scale
        end
        return C
    end

    C11 = reconstruct_matrix(A.A11)
    C22 = reconstruct_matrix(A.A22)
    n1 = size(C11, 1)
    n2 = size(C22, 1)
    n = n1 + n2

    C_full = CuArray{T_Base}(undef, n, n)
    C_full[1:n1, 1:n1] .= C11
    C_full[n1+1:n, n1+1:n] .= C22
    C_full[1:n1, n1+1:n] .= A.A12
    C_full[n1+1:n, 1:n1] .= A.A21
    if A.A12_scale !== nothing
        C_full[1:n1, n1+1:n] .*= A.A12_scale
    end
    if A.A21_scale !== nothing
        C_full[n1+1:n, 1:n1] .*= A.A21_scale
    end

    return C_full
end

# =============================================================================
# recgemm!: recursive mixed-precision GEMM for the Schur complement update
# =============================================================================

"""
    _gemm_dispatch!(alpha::Number, A::CUDA.StridedCuArray, B::CUDA.StridedCuArray, beta::Number, C::CUDA.StridedCuArray)

Handles type-conversion and hardware routing for general (non-transposed) matrix
multiplications `C = alpha*A*B + beta*C`, mirroring `_syrk_dispatch!`.
"""
function _gemm_dispatch!(
    alpha::Number, A::CUDA.StridedCuArray, B::CUDA.StridedCuArray, beta::Number, C::CUDA.StridedCuArray
)
    TA = eltype(A)
    TB = eltype(B)
    TC = eltype(C)

    if TA == TB == TC && TC in (Float32, Float64)
        gemm!('N', 'N', TC(alpha), A, B, TC(beta), C)
    elseif TA == Float16 && TB == Float16 && TC in (Float16, Float32)
        gemmEx!('N', 'N', alpha, A, B, beta, C)
    else
        A_final = (TA == TC) ? A : TC.(A)
        B_final = (TB == TC) ? B : TC.(B)
        if TC in (Float32, Float64)
            gemm!('N', 'N', TC(alpha), A_final, B_final, TC(beta), C)
        else
            gemmEx!('N', 'N', alpha, A_final, B_final, beta, C)
        end
    end
end

"""
    _recgemm_impl!(alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::AbstractMatrix, threshold::Int; parallel::Bool)

Internal implementation for nested recursive general matrix multiplications.
Recursively divides matrices into sub-blocks and applies updates in-place, falling
back to standard hardware routines using the dispatch helper at the base case.
"""
function _recgemm_impl!(
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::AbstractMatrix,
    threshold::Int; parallel::Bool
)
    n = size(C, 1)
    m = size(C, 2)
    if n <= threshold || m <= threshold
        _gemm_dispatch!(alpha, A, B, beta, C)
        return
    end

    n1 = 2^floor(Int, log2(n)) ÷ 2
    m1 = 2^floor(Int, log2(m)) ÷ 2
    k = size(A, 2)

    A1 = @view A[1:n1, 1:k];  A2 = @view A[n1+1:end, 1:k]
    B1 = @view B[1:k, 1:m1];  B2 = @view B[1:k, m1+1:end]

    C11 = @view C[1:n1, 1:m1];      C12 = @view C[1:n1, m1+1:end]
    C21 = @view C[n1+1:end, 1:m1];  C22 = @view C[n1+1:end, m1+1:end]

    _gemm_dispatch!(alpha, A1, B2, beta, C12)
    _gemm_dispatch!(alpha, A2, B1, beta, C21)

    if parallel
        @sync begin
            @async _recgemm_impl!(alpha, A1, B1, beta, C11, threshold, parallel=false)
            @async _recgemm_impl!(alpha, A2, B2, beta, C22, threshold, parallel=false)
        end
    else
        _recgemm_impl!(alpha, A1, B1, beta, C11, threshold, parallel=false)
        _recgemm_impl!(alpha, A2, B2, beta, C22, threshold, parallel=false)
    end
end

"""
    _recgemm_impl!(alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::FullMixedPrec; parallel::Bool)

Internal implementation for nested recursive general matrix multiplications
specifically for the `FullMixedPrec` block structure. Recursively divides matrices
into sub-blocks and applies updates in-place, falling back to standard hardware
routines using the dispatch helper at the base case.
"""
function _recgemm_impl!(
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::FullMixedPrec;
    parallel::Bool
)
    if C.BaseCase !== nothing
        recgemm!(alpha, A, B, beta, C.BaseCase, 4096)
        return
    end

    n1 = size(C.A11, 1)
    A1 = @view A[1:n1, :];  A2 = @view A[n1+1:end, :]
    B1 = @view B[:, 1:n1];  B2 = @view B[:, n1+1:end]

    _gemm_dispatch!(alpha, A1, B2, beta, C.A12)
    _gemm_dispatch!(alpha, A2, B1, beta, C.A21)

    if parallel
        @sync begin
            @async _recgemm_impl!(alpha, A1, B1, beta, C.A11, parallel=false)
            @async _recgemm_impl!(alpha, A2, B2, beta, C.A22, parallel=false)
        end
    else
        _recgemm_impl!(alpha, A1, B1, beta, C.A11, parallel=false)
        _recgemm_impl!(alpha, A2, B2, beta, C.A22, parallel=false)
    end
end

"""
    recgemm!(alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::FullMixedPrec)

Performs an in-place, nested recursive block general matrix multiplication
(`C = alpha*A*B + beta*C`) on a full mixed-precision matrix structure. Falls back
to standard hardware routines using the dispatch helper at the base case.
"""
function recgemm!(
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::FullMixedPrec
)
    if C.BaseCase !== nothing
        recgemm!(alpha, A, B, beta, C.BaseCase)
        return
    end
    n_subproblem = size(C.A11, 1)
    should_parallelize = n_subproblem > PARALLEL_THRESHOLD
    _recgemm_impl!(alpha, A, B, beta, C, parallel=should_parallelize)
end

"""
    recgemm!(alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::AbstractMatrix, threshold::Int=256)

Performs an in-place, nested recursive block general matrix multiplication
(`C = alpha*A*B + beta*C`). Falls back to standard hardware routines using the
dispatch helper at the specified base case threshold.
"""
function recgemm!(
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::AbstractMatrix, threshold::Int=256
)
    should_parallelize = size(C, 1) > PARALLEL_THRESHOLD
    _recgemm_impl!(alpha, A, B, beta, C, threshold, parallel=should_parallelize)
end

# =============================================================================
# CUSOLVER nonpivoting LU base case
# =============================================================================

for (bname, fname, elty) in
    ((:cusolverDnSgetrf_bufferSize, :cusolverDnSgetrf, :Float32),
     (:cusolverDnDgetrf_bufferSize, :cusolverDnDgetrf, :Float64))
    @eval begin
        """
            getrf_nopiv!(A::StridedCuMatrix)

        Computes the nonpivoting LU factorization of `A` in-place using CUSOLVER.
        Passing `CU_NULL` as `devIpiv` selects cuSOLVER's non-pivoting variant of
        `getrf`, so `A` is overwritten with the packed `L` (unit lower) and `U` factors.
        """
        function getrf_nopiv!(A::StridedCuMatrix{$elty})
            m, n = size(A)
            lda = max(1, stride(A, 2))
            dh = CUSOLVER.dense_handle()

            lwork = Ref{Cint}(0)
            CUSOLVER.$bname(dh, m, n, A, lda, lwork)
            workspace = CuVector{$elty}(undef, lwork[])
            devinfo = CuVector{Cint}(undef, 1)

            CUSOLVER.$fname(dh, m, n, A, lda, workspace, CU_NULL, devinfo)

            info = only(Array(devinfo))
            info < 0 && throw(ArgumentError("Invalid argument $(-info) to getrf"))
            info > 0 && @warn "getrf_nopiv!: U($info,$info) is exactly zero; the factorization is singular"
            return A
        end
    end
end

"""
    dispatch_getrf!(A)

Handles type-conversion and hardware routing for the LU base case, mirroring
`dispatch_trsm!`. `Float16` blocks are promoted to `Float32` for the CUSOLVER
factorization and copied back in-place.
"""
function dispatch_getrf!(A)
    if eltype(A) == Float16
        A_temp = Float32.(A)
        getrf_nopiv!(A_temp)
        copy!(A, A_temp)
    else
        getrf_nopiv!(A)
    end
end

# =============================================================================
# getrf_recursive!: nested recursive block nonpivoting LU factorization
# =============================================================================

"""
    getrf_recursive!(A, block_size)

Performs an in-place, nested recursive nonpivoting LU factorization on the matrix
`A`. The recursion dynamically splits the matrix until the sub-block size is less
than or equal to `block_size`, at which point it falls back to CUSOLVER's
nonpivoting `getrf`. On exit, `A` holds the packed factors: `L` in the strict
lower triangle (unit diagonal implied) and `U` in the upper triangle.
"""
function getrf_recursive!(A, block_size)
    n = size(A, 1)

    if n <= block_size
        dispatch_getrf!(A)
        return
    end

    n1 = 2^floor(Int, log2(n)) ÷ 2

    A11 = @view A[1:n1, 1:n1]
    A12 = @view A[1:n1, n1+1:end]
    A21 = @view A[n1+1:end, 1:n1]
    A22 = @view A[n1+1:end, n1+1:end]

    # Step 1: A11 -> L11 U11
    getrf_recursive!(A11, block_size)

    # Step 2: A12 <- L11^-1 A12  (L11 is unit lower triangular)
    # Step 3: A21 <- A21 U11^-1  (U11 is non-unit upper triangular)
    if (eltype(A11) == Float16)
        unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', A11, A12)
        unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', A11, A21)
    else
        trsm!('L', 'L', 'N', 'U', 1.0, A11, A12)
        trsm!('R', 'U', 'N', 'N', 1.0, A11, A21)
    end

    # Step 4: A22 <- A22 - A21 * A12  (Schur complement)
    if (eltype(A21) == Float16)
        recgemm!(-1.0, A21, A12, 1.0, A22)
    else
        gemm!('N', 'N', eltype(A22)(-1.0), A21, A12, eltype(A22)(1.0), A22)
    end

    # Step 5: A22 -> L22 U22
    getrf_recursive!(A22, block_size)
end

"""
    getrf_recursive!(A::FullMixedPrec)

Performs an in-place, nested recursive nonpivoting LU factorization on a full
mixed-precision matrix structure `A`. The recursion handles the off-diagonal panel
solves via `unified_rectrxm!` (viewing the factored `A11` block as `TriMixedPrec`
triangles) and the Schur complement via `recgemm!`, falling back to CUSOLVER at
the base case. Off-diagonal quantization scales are folded into the Schur
complement's scalar multiplier.
"""
function getrf_recursive!(A::FullMixedPrec)
    if A.BaseCase !== nothing
        getrf_recursive!(A.BaseCase, 4096)
        return
    end

    # Step 1: A11 -> L11 U11
    getrf_recursive!(A.A11)

    # Step 2: A12 <- L11^-1 A12  (unit lower triangle of the factored A11)
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', TriMixedPrec(A.A11, 'L'), A.A12)

    # Step 3: A21 <- A21 U11^-1  (non-unit upper triangle of the factored A11)
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', TriMixedPrec(A.A11, 'U'), A.A21)

    # Step 4: A22 <- A22 - A21 * A12, folding any Float16 block scales into alpha
    alpha = -1.0
    if A.A21_scale !== nothing
        alpha *= A.A21_scale
    end
    if A.A12_scale !== nothing
        alpha *= A.A12_scale
    end
    recgemm!(alpha, A.A21, A.A12, 1.0, A.A22)

    # Step 5: A22 -> L22 U22
    getrf_recursive!(A.A22)
end