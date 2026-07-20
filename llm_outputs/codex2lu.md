Great prompt. You’re right: what’s missing is a full (non-symmetric/non-triangular) hierarchical mixed-precision container plus a general recursive Schur-update kernel for LU.
Below is a drop-in style implementation that mirrors your current architecture (BaseCase fallback, recursive split, multiple dispatch) and uses CUSOLVER via lu!(A, NoPivot()) on CuArray at the base case.
using LinearAlgebra
using CUDA
include("wrappers.jl")
include("rectrxm.jl")   # unified_rectrxm!
# GEMM_SUB!/GEMM_ADD! already defined in your stack for GPU arrays

export FullMixedPrec, recgemm_sub!, getrf_recursive!

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

@inline _scale_or_one(s::Union{Nothing,Float32}) = s === nothing ? 1.0f0 : s

@inline function _split_mid(n::Int)
    # Same style as your existing constructors
    return isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))
end

function _convert_block_with_quantization(
    X::AbstractMatrix,
    T::DataType
)
    FP16_MAX_VAL = 65504.0f0
    if T == Float16
        α = maximum(abs, X)
        Y = similar(X, Float16, size(X))
        if α > FP16_MAX_VAL
            s = Float32(α / FP16_MAX_VAL)
            @. Y = Float16(round(clamp(X / s, -FP16_MAX_VAL, FP16_MAX_VAL)))
            return Y, s
        else
            Y .= X
            return Y, nothing
        end
    else
        if eltype(X) == T
            return X, nothing
        else
            Y = similar(X, T, size(X))
            Y .= X
            return Y, nothing
        end
    end
end

# Optional CPU fallback (if you ever run CPU tests)
function GEMM_SUB!(C::StridedMatrix, A, B, scale::Float32=1.0f0)
    transA = A isa Transpose ? 'T' : 'N'
    transB = B isa Transpose ? 'T' : 'N'
    A_mat = A isa Transpose ? parent(A) : A
    B_mat = B isa Transpose ? parent(B) : B
    T = eltype(C)
    gemm!(transA, transB, T(-scale), A_mat, B_mat, one(T), C)
end

# ------------------------------------------------------------------------------
# Full mixed-precision hierarchical structure
# ------------------------------------------------------------------------------

"""
    FullMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}

Recursive mixed-precision structure for full dense square matrices:
    [ A11  A12
      A21  A22 ]

- `A11`, `A22` are recursive `FullMixedPrec` blocks.
- `A12`, `A21` are dense off-diagonal blocks.
- Float16 blocks use dynamic quantization with per-block scales.
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

function FullMixedPrec(
    A::AbstractMatrix;
    precisions::Vector{DataType}
)
    @assert !isempty(precisions) "precisions cannot be empty"
    n = size(A, 1)
    @assert n == size(A, 2) "A must be square"

    # Base case
    if length(precisions) == 1 || n <= 1
        T_Base = precisions[1]
        base_matrix, base_scale = _convert_block_with_quantization(A, T_Base)
        return FullMixedPrec{T_Base}(
            nothing, nothing, nothing, nothing,
            nothing, nothing,
            base_scale, base_matrix,
            (n, n)
        )
    end

    # Recursive case
    mid = _split_mid(n)
    T_Off = precisions[1]
    remp = precisions[2:end]

    A11 = FullMixedPrec(view(A, 1:mid,       1:mid);       precisions=remp)
    A22 = FullMixedPrec(view(A, mid+1:n,     mid+1:n);     precisions=remp)

    A12_view = view(A, 1:mid,     mid+1:n)
    A21_view = view(A, mid+1:n,   1:mid)

    A12_mat, A12_scale = _convert_block_with_quantization(A12_view, T_Off)
    A21_mat, A21_scale = _convert_block_with_quantization(A21_view, T_Off)

    T_Final_Base = precisions[end]
    return FullMixedPrec{T_Final_Base}(
        A11, A22,
        A12_mat, A21_mat,
        A12_scale, A21_scale,
        nothing, nothing,
        (n, n)
    )
end

Base.size(A::FullMixedPrec) = A.sz

function Base.getindex(A::FullMixedPrec{T_Base}, i::Int, j::Int) where {T_Base}
    if A.BaseCase !== nothing
        return A.BaseCase[i, j]
    end
    mid = size(A.A11, 1)
    if i <= mid && j <= mid
        return A.A11[i, j]
    elseif i <= mid && j > mid
        return A.A12[i, j - mid]
    elseif i > mid && j <= mid
        return A.A21[i - mid, j]
    else
        return A.A22[i - mid, j - mid]
    end
end

# ------------------------------------------------------------------------------
# Recursive Schur update: C -= alpha * A * B
# ------------------------------------------------------------------------------

function _gemm_sub_into_block!(
    A::AbstractMatrix,
    B::AbstractMatrix,
    Cblk::AbstractMatrix,
    Cblk_scale::Union{Nothing,Float32},
    alpha::Float32
)
    γ = Float32(alpha / _scale_or_one(Cblk_scale))
    GEMM_SUB!(Cblk, A, B, γ)
    return Cblk
end

"""
    recgemm_sub!(A, B, C::AbstractMatrix, threshold=256; alpha=1f0)

Recursive dense update: C .= C .- alpha * A * B
"""
function recgemm_sub!(
    A::AbstractMatrix,
    B::AbstractMatrix,
    C::AbstractMatrix,
    threshold::Int=256;
    alpha::Float32=1.0f0
)
    n = size(C, 1)
    @assert n == size(C, 2)
    @assert size(A, 1) == n
    @assert size(B, 2) == n
    @assert size(A, 2) == size(B, 1)

    if n <= threshold
        GEMM_SUB!(C, A, B, alpha)
        return C
    end

    mid = _split_mid(n)

    A1 = @view A[1:mid, :]
    A2 = @view A[mid+1:n, :]

    B1 = @view B[:, 1:mid]
    B2 = @view B[:, mid+1:n]

    C11 = @view C[1:mid,     1:mid]
    C12 = @view C[1:mid,     mid+1:n]
    C21 = @view C[mid+1:n,   1:mid]
    C22 = @view C[mid+1:n,   mid+1:n]

    recgemm_sub!(A1, B1, C11, threshold; alpha=alpha)
    recgemm_sub!(A1, B2, C12, threshold; alpha=alpha)
    recgemm_sub!(A2, B1, C21, threshold; alpha=alpha)
    recgemm_sub!(A2, B2, C22, threshold; alpha=alpha)

    return C
end

"""
    recgemm_sub!(A, B, C::FullMixedPrec; alpha=1f0)

Recursive mixed-precision update into hierarchical FullMixedPrec:
C .= C .- alpha * A * B
"""
function recgemm_sub!(
    A::AbstractMatrix,
    B::AbstractMatrix,
    C::FullMixedPrec;
    alpha::Float32=1.0f0
)
    if C.BaseCase !== nothing
        γ = Float32(alpha / _scale_or_one(C.base_scale))
        GEMM_SUB!(C.BaseCase, A, B, γ)
        return C
    end

    n = size(C, 1)
    mid = size(C.A11, 1)

    A1 = @view A[1:mid, :]
    A2 = @view A[mid+1:n, :]

    B1 = @view B[:, 1:mid]
    B2 = @view B[:, mid+1:n]

    recgemm_sub!(A1, B1, C.A11; alpha=alpha)
    _gemm_sub_into_block!(A1, B2, C.A12, C.A12_scale, alpha)
    _gemm_sub_into_block!(A2, B1, C.A21, C.A21_scale, alpha)
    recgemm_sub!(A2, B2, C.A22; alpha=alpha)

    return C
end

# ------------------------------------------------------------------------------
# Nonpivoting base-case LU dispatch (CUSOLVER on CuArray)
# ------------------------------------------------------------------------------

function _getrf_nopiv_dispatch!(A::CUDA.StridedCuArray)
    # CUDA.jl dispatches to CUSOLVER
    lu!(A, NoPivot())
    return A
end

function _getrf_nopiv_dispatch!(A::StridedMatrix)
    lu!(A, NoPivot())
    return A
end

# ------------------------------------------------------------------------------
# Recursive LU (dense + mixed), in-place
# ------------------------------------------------------------------------------

"""
    getrf_recursive!(A::AbstractMatrix, block_size=256)

In-place recursive nonpivoting LU:
A = [A11 A12; A21 A22]
1) A11 = L11*U11
2) U12 = L11 \\ A12
3) L21 = A21 / U11
4) A22 -= L21*U12
5) recurse on A22
"""
function getrf_recursive!(A::AbstractMatrix, block_size::Int=256)
    n = size(A, 1)
    @assert n == size(A, 2)

    if n <= block_size
        _getrf_nopiv_dispatch!(A)
        return A
    end

    mid = _split_mid(n)

    A11 = @view A[1:mid,     1:mid]
    A12 = @view A[1:mid,     mid+1:n]
    A21 = @view A[mid+1:n,   1:mid]
    A22 = @view A[mid+1:n,   mid+1:n]

    # 1) Factor top-left
    getrf_recursive!(A11, block_size)

    # 2) U12 = L11^{-1} * A12   (L is unit-lower)
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', A11, A12)

    # 3) L21 = A21 * U11^{-1}   (U is upper non-unit)
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', A11, A21)

    # 4) Schur complement
    recgemm_sub!(A21, A12, A22, block_size)

    # 5) recurse
    getrf_recursive!(A22, block_size)

    return A
end

"""
    getrf_recursive!(A::FullMixedPrec)

In-place recursive nonpivoting LU over hierarchical mixed-precision storage.
Base case falls back to dense CUSOLVER/LAPACK path.
"""
function getrf_recursive!(A::FullMixedPrec)
    if A.BaseCase !== nothing
        getrf_recursive!(A.BaseCase, 4096)
        return A
    end

    # 1) Factor A11
    getrf_recursive!(A.A11)

    # 2) U12 = L11^{-1} * A12
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', A.A11, A.A12)

    # 3) L21 = A21 * U11^{-1}
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', A.A11, A.A21)

    # 4) A22 -= L21 * U12
    # If A12/A21 are quantized blocks, recover scale in alpha
    αschur = Float32(_scale_or_one(A.A21_scale) * _scale_or_one(A.A12_scale))
    recgemm_sub!(A.A21, A.A12, A.A22; alpha=αschur)

    # 5) recurse on A22
    getrf_recursive!(A.A22)

    return A
end

Notes (important)
This is nonpivoting LU; numerical failure is expected for small/zero pivots.
lu!(A, NoPivot()) on CuArray is the CUSOLVER-backed base case in CUDA.jl.
The recursive triangular solves use your existing unified_rectrxm! infrastructure exactly in the same style as your Cholesky flow.
If you want, I can also add a reconstruct_matrix(::FullMixedPrec) and a quick correctness harness (compare against lu(A, NoPivot()) residual).

