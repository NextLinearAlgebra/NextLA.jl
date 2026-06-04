export FullMixedPrec, reconstruct_matrix

# Full precision data structure utilizing a recursive data type.
# Modified from SymmMixedPrec: 
# - Removed uplo parameter as this is for full matrices.
# - Replaced OffDiag with explicit A21 and A12 block matrices.
# - Added corresponding A21_scale and A12_scale for quantization.
struct FullMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}
    A11::Union{FullMixedPrec{T_Base}, Nothing}
    A22::Union{FullMixedPrec{T_Base}, Nothing}

    A21::Union{AbstractMatrix, Nothing}
    A12::Union{AbstractMatrix, Nothing}

    # scaling for quantization
    A21_scale::Union{Float32, Nothing}
    A12_scale::Union{Float32, Nothing}
    
    base_scale::Union{Float32, Nothing}
    BaseCase::Union{AbstractMatrix{T_Base}, Nothing}
    sz::Tuple{Int, Int}
end

function FullMixedPrec(
    A::AbstractMatrix;
    precisions::Vector{DataType}
)
    FP16_MAX_VAL = 65504.0f0
    n = size(A, 1)
    m = size(A, 2)
    
    @assert n == m "A must be square for recursive factorization structure"

    # Base Case Condition
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

    # Recursive split point
    mid = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))

    T_OffDiag = precisions[1]
    remaining_precisions = precisions[2:end]

    # Recursively subdivide the diagonal blocks
    A11 = FullMixedPrec(view(A, 1:mid, 1:mid); precisions=remaining_precisions)
    A22 = FullMixedPrec(view(A, mid+1:n, mid+1:n); precisions=remaining_precisions)

    # Extract the views for A21 and A12
    view_A21 = view(A, mid+1:n, 1:mid)
    view_A12 = view(A, 1:mid, mid+1:n)

    local A21_matrix, A12_matrix
    local A21_scale = nothing
    local A12_scale = nothing

    # Handle A21 block
    if T_OffDiag == Float16
        alpha_A21 = maximum(abs, view_A21)
        if alpha_A21 > FP16_MAX_VAL
            A21_scale = Float32(alpha_A21 / FP16_MAX_VAL)
            A21_matrix = similar(view_A21, Float16, size(view_A21))
            @. A21_matrix = Float16(round(clamp(view_A21 / A21_scale, -FP16_MAX_VAL, FP16_MAX_VAL)))
        else
            A21_matrix = similar(view_A21, Float16, size(view_A21))
            A21_matrix .= view_A21
        end
    else
        A21_matrix = similar(A, T_OffDiag, size(view_A21))
        A21_matrix .= view_A21
    end

    # Handle A12 block
    if T_OffDiag == Float16
        alpha_A12 = maximum(abs, view_A12)
        if alpha_A12 > FP16_MAX_VAL
            A12_scale = Float32(alpha_A12 / FP16_MAX_VAL)
            A12_matrix = similar(view_A12, Float16, size(view_A12))
            @. A12_matrix = Float16(round(clamp(view_A12 / A12_scale, -FP16_MAX_VAL, FP16_MAX_VAL)))
        else
            A12_matrix = similar(view_A12, Float16, size(view_A12))
            A12_matrix .= view_A12
        end
    else
        A12_matrix = similar(A, T_OffDiag, size(view_A12))
        A12_matrix .= view_A12
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
    else # i <= mid && j > mid
        return A.A12[i, j - mid]
    end
end

# Helper function to copy the hierarchical matrix back into a flat GPU/CPU matrix
function reconstruct_matrix(A::FullMixedPrec{T_Base}) where {T_Base}
    if A.BaseCase !== nothing
        return copy(A.BaseCase)
    end
    
    C11 = reconstruct_matrix(A.A11)
    C22 = reconstruct_matrix(A.A22)
    C21 = A.A21
    C12 = A.A12
    
    n1, m1 = size(C11)
    n2, m2 = size(C22)
    n = n1 + n2

    # Allocate full matrix on the same device as the leaf blocks
    C_full = similar(C21, T_Base, n, n)
    
    C_full[1:n1, 1:m1] .= C11
    C_full[n1+1:n, 1:m1] .= C21
    C_full[n1+1:n, m1+1:n] .= C22
    C_full[1:n1, m1+1:n] .= C12

    return C_full
end
