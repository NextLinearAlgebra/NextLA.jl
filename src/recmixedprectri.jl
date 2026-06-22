"""
    TriMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}

A hierarchical, recursive mixed-precision data structure that maps to triangular matrices.
It partitions the matrix into two recursive diagonal blocks (`A11`, `A22`) and a single dense 
off-diagonal block (`OffDiag`), structured according to its `uplo` ('U' for upper, 'L' for lower) character.
"""
struct TriMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}
    A11::Union{TriMixedPrec{T_Base}, Nothing}
    A22::Union{TriMixedPrec{T_Base}, Nothing}
    OffDiag::Union{AbstractMatrix, Nothing}
    offDiag_scale::Union{Float32, Nothing}
    base_scale::Union{Float32, Nothing}
    BaseCase::Union{AbstractMatrix{T_Base}, Nothing}
    uplo::Char
    sz::Tuple{Int, Int}
end

"""
    TriMixedPrec(A::AbstractMatrix, uplo::Char; precisions::Vector{DataType})

Constructs a `TriMixedPrec` representation of the triangular matrix `A`.

Utilizes a base-2 recursive splitting scheme to divide the matrix into two diagonal triangular 
blocks and one off-diagonal rectangular block. The algorithm maps the given `precisions` to 
each depth level. For blocks evaluated as `Float16`, dynamic quantization is applied to safely 
scale and clamp any values that exceed the `Float16` maximum (`65504.0f0`), ensuring robust 
mixed-precision arithmetic without numerical overflow.
"""
function TriMixedPrec(
    A::AbstractMatrix,
    uplo::Char;
    precisions::Vector{DataType}
)
    FP16_MAX_VAL = 65504.0f0
    n = size(A, 1)
    @assert n == size(A, 2)

    if length(precisions) == 1
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
            if eltype(A) == T_Base
                base_matrix = A
            else
                base_matrix = similar(A, T_Base, size(A))
                base_matrix .= A
            end
            base_scale = nothing
        end
        
        return TriMixedPrec{T_Base}(nothing, nothing, nothing, nothing, base_scale, base_matrix, uplo, (n, n))
    end

    if isinteger(log2(n))
        mid = div(n, 2)
    else
        mid = 2 ^ floor(Int, log2(n))
    end
    
    T_OffDiag = precisions[1]
    remaining_precisions = precisions[2:end]

    A11 = TriMixedPrec(view(A, 1:mid, 1:mid), uplo; precisions=remaining_precisions)
    A22 = TriMixedPrec(view(A, mid+1:n, mid+1:n), uplo; precisions=remaining_precisions)

    local offDiag_matrix
    local offDiag_view
    local offDiag_scale = nothing
    if uplo == 'L'
        offDiag_view = view(A, mid+1:n, 1:mid)
    else
        offDiag_view = view(A, 1:mid, mid+1:n)
    end

    if T_OffDiag == Float16
        alpha_offDiag = maximum(abs, offDiag_view)
        if alpha_offDiag > FP16_MAX_VAL
            offDiag_scale = Float32(alpha_offDiag / FP16_MAX_VAL)
            offDiag_matrix = similar(offDiag_view, Float16, size(offDiag_view))
            @. offDiag_matrix = Float16(round(clamp(offDiag_view / offDiag_scale, -FP16_MAX_VAL, FP16_MAX_VAL)))
        else
            offDiag_matrix = similar(offDiag_view, Float16, size(offDiag_view))
            offDiag_matrix .= offDiag_view
        end
    else
        if eltype(offDiag_view) == T_OffDiag
            offDiag_matrix = offDiag_view
        else
            offDiag_matrix = similar(A, T_OffDiag, size(offDiag_view))
            offDiag_matrix .= offDiag_view
        end
        offDiag_scale = nothing
    end

    T_Final_Base = precisions[end]
    return TriMixedPrec{T_Final_Base}(A11, A22, offDiag_matrix, offDiag_scale, nothing, nothing, uplo, (n, n))
end

function Base.size(A::TriMixedPrec)
    return A.sz
end

function Base.getindex(A::TriMixedPrec{T_Base}, i::Int, j::Int) where {T_Base}
    if A.BaseCase !== nothing
        return A.BaseCase[i, j]
    end

    mid = size(A.A11, 1)

    if i <= mid && j <= mid
        return A.A11[i, j]
    elseif i > mid && j > mid
        return A.A22[i - mid, j - mid]
    elseif i > mid && j <= mid
        if A.uplo == 'L'
            return A.OffDiag[i - mid, j]
        else
            return zero(T_Base)
        end
    else
        if A.uplo == 'U'
            return A.OffDiag[i, j - mid]
        else
            return zero(T_Base)
        end
    end
end

function Base.sizeof(A::TriMixedPrec)
    if A.BaseCase !== nothing
        return sizeof(A.BaseCase)
    end

    return sizeof(A.A11) + sizeof(A.A22) + sizeof(A.OffDiag)
end

"""
    TriMixedPrec(A::SymmMixedPrec{T_Base})

Dynamically converts an existing `SymmMixedPrec` matrix structure into a `TriMixedPrec` format.
"""
function TriMixedPrec(A::SymmMixedPrec{T_Base}) where {T_Base}
    if A.BaseCase !== nothing
        return TriMixedPrec{T_Base}(
            nothing, nothing, nothing,
            nothing, A.base_scale, A.BaseCase,
            A.uplo, A.sz
        )
    end

    return TriMixedPrec{T_Base}(
        TriMixedPrec(A.A11),
        TriMixedPrec(A.A22),
        A.OffDiag,
        A.offDiag_scale,
        nothing,  
        nothing,  
        A.uplo,
        A.sz
    )
end