using LinearAlgebra
using CUDA
using CUDA.CUSOLVER
using StochasticRounding
include("symmmixedprec.jl")
include("recmixedprectri.jl")
include("rectrxm.jl")
include("recsyrk.jl")
include("wrappers.jl")

# =============================================================================
# FullMixedPrec: hierarchical mixed-precision structure for dense square matrices
# =============================================================================

"""
    FullMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}

A hierarchical, recursive mixed-precision data structure that maps to full dense
square matrices. It recursively partitions the matrix into two dense diagonal
sub-blocks (`A11`, `A22`) and TWO corresponding off-diagonal blocks: the lower
off-diagonal block `A21` and the upper off-diagonal block `A12`, each carrying an
independent dynamic-quantization scale (`A21_scale`, `A12_scale`).

The field names `A21`/`A12` and `A21_scale`/`A12_scale` are the ones probed by
`unified_rec_mixed` via `hasproperty(A, :A21)` / `hasproperty(A, :A21_scale)`,
which allows a `FullMixedPrec` to be consumed directly as a triangular operand
(the `uplo` argument selects which off-diagonal block participates).
"""
struct FullMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}
    A11::Union{FullMixedPrec{T_Base}, Nothing}
    A22::Union{FullMixedPrec{T_Base}, Nothing}
    A21::Union{AbstractMatrix, Nothing}
    A12::Union{AbstractMatrix, Nothing}
    A21_scale::Union{Float32, Nothing}
    A12_scale::Union{Float32, Nothing}
    base_scale::Union{Float32, Nothing}
    BaseCase::Union{AbstractMatrix{T_Base}, Nothing}
    sz::Tuple{Int, Int}
end

"""
    FullMixedPrec(A::AbstractMatrix; precisions::Vector{DataType})

Constructs a `FullMixedPrec` representation of the dense square matrix `A`.

Uses a base-2 recursive splitting scheme to partition the matrix into two dense
diagonal blocks and two rectangular off-diagonal blocks (`A21`, `A12`). To preserve
bounds during `Float16` conversion, per-block dynamic quantization handles elements
by detecting values exceeding `65504.0f0`, computing a scaling factor, and applying
clamping. This ensures high-magnitude structures safely avoid numerical overflow
across the memory layout.
"""
function FullMixedPrec(
    A::AbstractMatrix;
    precisions::Vector{DataType}
)
    FP16_MAX_VAL = 65504.0f0
    n = size(A, 1)
    @assert n == size(A, 2) "A must be square"

    if length(precisions) == 1 || n <= 1
        T_Base = precisions[1]
        local base_matrix
        local base_scale

        if T_Base == Float16
            alpha = maximum(abs, A)
            if alpha > FP16_MAX_VAL
                base_scale = Float32(alpha / FP16_MAX_VAL)
                base_matrix = similar(A, Float16, size(A))
                @. base_matrix = Float16(round(clamp(A / base_scale, -FP16_MAX_VAL, FP16_MAX_VAL)))
            else
                base_scale = nothing
                base_matrix = similar(A, Float16, size(A))
                base_matrix .= A
            end
        else
            base_matrix = similar(A, T_Base, size(A))
            base_matrix .= A
            base_scale = nothing
        end

        return FullMixedPrec{T_Base}(nothing, nothing, nothing, nothing, nothing, nothing, base_scale, base_matrix, (n, n))
    end

    mid = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))

    T_OffDiag = precisions[1]
    remaining_precisions = precisions[2:end]

    A11 = FullMixedPrec(view(A, 1:mid, 1:mid); precisions=remaining_precisions)
    A22 = FullMixedPrec(view(A, mid+1:n, mid+1:n); precisions=remaining_precisions)

    A21_view = view(A, mid+1:n, 1:mid)
    A12_view = view(A, 1:mid, mid+1:n)

    local A21_matrix
    local A21_scale = nothing
    if T_OffDiag == Float16
        alpha_A21 = maximum(abs, A21_view)
        if alpha_A21 > FP16_MAX_VAL
            A21_scale = Float32(alpha_A21 / FP16_MAX_VAL)
            A21_matrix = similar(A21_view, Float16, size(A21_view))
            @. A21_matrix = Float16(round(clamp(A21_view / A21_scale, -FP16_MAX_VAL, FP16_MAX_VAL)))
        else
            A21_matrix = similar(A21_view, Float16, size(A21_view))
            A21_matrix .= A21_view
        end
    else
        A21_matrix = similar(A, T_OffDiag, size(A21_view))
        A21_matrix .= A21_view
        A21_scale = nothing
    end

    local A12_matrix
    local A12_scale = nothing
    if T_OffDiag == Float16
        alpha_A12 = maximum(abs, A12_view)
        if alpha_A12 > FP16_MAX_VAL
            A12_scale = Float32(alpha_A12 / FP16_MAX_VAL)
            A12_matrix = similar(A12_view, Float16, size(A12_view))
            @. A12_matrix = Float16(round(clamp(A12_view / A12_scale, -FP16_MAX_VAL, FP16_MAX_VAL)))
        else
            A12_matrix = similar(A12_view, Float16, size(A12_view))
            A12_matrix .= A12_view
        end
    else
        A12_matrix = similar(A, T_OffDiag, size(A12_view))
        A12_matrix .= A12_view
        A12_scale = nothing
    end

    T_Final_Base = precisions[end]
    return FullMixedPrec{T_Final_Base}(A11, A22, A21_matrix, A12_matrix, A21_scale, A12_scale, nothing, nothing, (n, n))
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

    return sizeof(A.A11) + sizeof(A.A22) + sizeof(A.A21) + sizeof(A.A12)
end

# =============================================================================
# CUSOLVER nonpivoting base case
# =============================================================================

# """
#     getrf_nopiv!(A::StridedCuMatrix)

# Performs an in-place, nonpivoting LU factorization of `A` using CUSOLVER.
# Calls `cusolverDn<T>getrf` with a null pivot array (`CU_NULL`), which per the
# CUSOLVER documentation disables partial pivoting entirely. On return, the strictly
# lower triangle of `A` holds the unit-lower-triangular factor `L` (implicit unit
# diagonal) and the upper triangle holds `U`.
# """
for (bname, fname, elty) in ((:cusolverDnSgetrf_bufferSize, :cusolverDnSgetrf, :Float32),
                             (:cusolverDnDgetrf_bufferSize, :cusolverDnDgetrf, :Float64))
    @eval begin
        function getrf_nopiv!(A::StridedCuMatrix{$elty})
            m, n = size(A)
            lda = max(1, stride(A, 2))
            dh = CUSOLVER.dense_handle()

            lwork = Ref{Cint}(0)
            CUSOLVER.$bname(dh, m, n, A, lda, lwork)

            work = CuVector{$elty}(undef, lwork[])
            devinfo = CuVector{Cint}(undef, 1)

            CUSOLVER.$fname(dh, m, n, A, lda, work, CU_NULL, devinfo)
            return A
        end
    end
end

"""
    _getrf_dispatch!(A::CUDA.StridedCuArray)

Handles type-conversion and hardware routing for the nonpivoting LU base case.
`Float16` blocks are promoted to `Float32` for the CUSOLVER call and copied back,
mirroring the promotion strategy of `dispatch_trsm!`/`dispatch_trmm!`.
"""
function _getrf_dispatch!(A::CUDA.StridedCuArray)
    if eltype(A) == Float16
        A_temp = Float32.(A)
        getrf_nopiv!(A_temp)
        clamp!(A_temp, floatmin(Float16), floatmax(Float16))
        copy!(A, A_temp)
    else
        getrf_nopiv!(A)
    end
end

# =============================================================================
# Recursive general matrix multiplication update (Schur complement kernel)
# =============================================================================

"""
    _gemm_dispatch!(alpha::Number, A::CUDA.StridedCuArray, B::CUDA.StridedCuArray, beta::Number, C::CUDA.StridedCuArray)

Handles type-conversion and hardware routing for general (non-transposed) matrix
multiplications `C = alpha*A*B + beta*C`. Mirrors the `:GEMM` branch of
`_syrk_dispatch!`, but with a `'N','N'` operation pattern as required by the
LU Schur complement update `A22 = A22 - A21*A12`.
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

Internal implementation for nested recursive general matrix multiplication updates
(`C = alpha*A*B + beta*C` for square `C`). Recursively divides matrices into
sub-blocks and applies updates in-place, falling back to standard hardware routines
using the dispatch helper at the base case.
"""
function _recgemm_impl!(
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::AbstractMatrix,
    threshold::Int; parallel::Bool
)
    n = size(C, 1)
    if n <= threshold
        _gemm_dispatch!(alpha, A, B, beta, C)
        return
    end

    n1 = 2^floor(Int, log2(n)) ÷ 2
    k = size(A, 2)

    A1 = @view A[1:n1, 1:k]; A2 = @view A[n1+1:end, 1:k]
    B1 = @view B[1:k, 1:n1]; B2 = @view B[1:k, n1+1:end]
    C11 = @view C[1:n1, 1:n1];     C12 = @view C[1:n1, n1+1:end]
    C21 = @view C[n1+1:end, 1:n1]; C22 = @view C[n1+1:end, n1+1:end]

    _gemm_dispatch!(alpha, A2, B1, beta, C21)
    _gemm_dispatch!(alpha, A1, B2, beta, C12)

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

Internal implementation for nested recursive general matrix multiplication updates
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
    A1 = @view A[1:n1, :]; A2 = @view A[n1+1:end, :]
    B1 = @view B[:, 1:n1]; B2 = @view B[:, n1+1:end]

    _gemm_dispatch!(alpha, A2, B1, beta, C.A21)
    _gemm_dispatch!(alpha, A1, B2, beta, C.A12)

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

Performs an in-place, nested recursive block general matrix multiplication update
on a full dense mixed-precision matrix structure. Falls back to standard hardware
routines using the dispatch helper at the base case.
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

Performs an in-place, nested recursive block general matrix multiplication update
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
# Recursive nonpivoting block LU factorization
# =============================================================================

"""
    getrf_recursive!(A, block_size)

Performs an in-place, nonpivoting, nested recursive block LU factorization on the
matrix `A`. The recursion dynamically splits the matrix until the sub-block size is
less than or equal to `block_size`, at which point it falls back to the CUSOLVER
nonpivoting `getrf` base case.

Recursive formulation, with `A = [A11 A12; A21 A22] = [L11 0; L21 L22] * [U11 U12; 0 U22]`:

    1. A11 = L11*U11                     (recursive LU of the leading block)
    2. A12 <- inv(L11) * A12  (= U12)    (left, unit-lower TRSM)
    3. A21 <- A21 * inv(U11)  (= L21)    (right, non-unit upper TRSM)
    4. A22 <- A22 - A21*A12              (Schur complement update)
    5. A22 = L22*U22                     (recursive LU of the trailing block)

On return, the strictly lower triangle of `A` holds `L` (implicit unit diagonal)
and the upper triangle holds `U`.
"""
function getrf_recursive!(A, block_size)
    n = size(A, 1)

    if n <= block_size
        _getrf_dispatch!(A)
        return
    end

    n1 = 2^floor(Int, log2(n)) ÷ 2

    A11 = @view A[1:n1, 1:n1]
    A12 = @view A[1:n1, n1+1:end]
    A21 = @view A[n1+1:end, 1:n1]
    A22 = @view A[n1+1:end, n1+1:end]

    getrf_recursive!(A11, block_size)

    if (eltype(A11) == Float16)
        unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', A11, A12)
        unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', A11, A21)
    else
        trsm!('L', 'L', 'N', 'U', 1.0, A11, A12)
        trsm!('R', 'U', 'N', 'N', 1.0, A11, A21)
    end

    if (eltype(A21) == Float16)
        recgemm!(-1.0, A21, A12, 1.0, A22)
    else
        gemm!('N', 'N', -1.0, A21, A12, 1.0, A22)
    end

    getrf_recursive!(A22, block_size)
end

"""
    reconstruct_matrix(A::FullMixedPrec{T_Base})

Reconstructs a full dense matrix from the dense mixed-precision recursive block
structure `A`, applying any dynamic-quantization scales. Used primarily for
validation and returning to standard dense formats.
"""
function reconstruct_matrix(A::FullMixedPrec{T_Base}) where {T_Base}
    if A.BaseCase !== nothing
        if A.base_scale !== nothing
            return A.base_scale .* Float32.(A.BaseCase)
        end
        return copy(A.BaseCase)
    end

    C11 = reconstruct_matrix(A.A11)
    C22 = reconstruct_matrix(A.A22)
    C21 = A.A21_scale !== nothing ? A.A21_scale .* Float32.(A.A21) : A.A21
    C12 = A.A12_scale !== nothing ? A.A12_scale .* Float32.(A.A12) : A.A12
    n1 = size(C11, 1)
    n2 = size(C22, 1)
    n = n1 + n2

    T_out = promote_type(eltype(C11), eltype(C21), eltype(C12), eltype(C22))
    C_full = CuArray{T_out}(undef, n, n)
    C_full[1:n1, 1:n1] .= C11
    C_full[n1+1:n, 1:n1] .= C21
    C_full[1:n1, n1+1:n] .= C12
    C_full[n1+1:n, n1+1:n] .= C22

    return C_full
end

"""
    getrf_recursive!(A::FullMixedPrec)

Performs an in-place, nonpivoting, nested recursive block LU factorization on a
full dense mixed-precision matrix structure `A`. The recursion handles both
off-diagonal panel updates and falls back to the CUSOLVER nonpivoting routine at
the base case.

Because `FullMixedPrec` exposes the `A21`/`A12` and `A21_scale`/`A12_scale`
fields probed by `unified_rec_mixed`, the diagonal sub-structure `A.A11` is passed
directly to `unified_rectrxm!` as a triangular operand: the `uplo` argument selects
which stored off-diagonal block (and scale) participates in the solve.
"""
function getrf_recursive!(A::FullMixedPrec)
    if A.BaseCase !== nothing
        getrf_recursive!(A.BaseCase, 4096)
        return
    end

    getrf_recursive!(A.A11)

    unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', A.A11, A.A12)
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', A.A11, A.A21)

    prod_scale = (A.A21_scale !== nothing ? A.A21_scale : 1.0f0) *
                 (A.A12_scale !== nothing ? A.A12_scale : 1.0f0)
    recgemm!(-prod_scale, A.A21, A.A12, 1.0, A.A22)

    getrf_recursive!(A.A22)
end