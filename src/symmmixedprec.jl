abstract type AbstractMixedPrec{T} <: AbstractMatrix{T} end


# mixed precision symmetric data structure utilizing a recursive data type 

struct SymmMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}
    A11::Union{SymmMixedPrec{T_Base}, Nothing}
    A22::Union{SymmMixedPrec{T_Base}, Nothing}

    OffDiag::Union{AbstractMatrix, Nothing}

    # scaling for quantization
    offDiag_scale::Union{Float32, Nothing}
    base_scale::Union{Float32, Nothing}

    BaseCase::Union{AbstractMatrix{T_Base}, Nothing}

    uplo::Char # 'U' or 'L' to know which triangle is stored
    sz::Tuple{Int, Int}
end


function SymmMixedPrec(
    A::AbstractMatrix,
    uplo::Char;
    precisions::Vector{DataType}
)
    FP16_MAX_VAL = 65504.0f0
    n = size(A, 1)
    @assert n == size(A, 2) "A must be square"

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
            base_matrix = similar(A, T_Base, size(A))
            base_matrix .= A
            base_scale = nothing
        end

        return SymmMixedPrec{T_Base}(nothing, nothing, nothing, nothing, base_scale, base_matrix, uplo, (n, n))
    end

    mid = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))

    T_OffDiag = precisions[1]
    remaining_precisions = precisions[2:end]

    A11 = SymmMixedPrec(view(A, 1:mid, 1:mid), uplo; precisions=remaining_precisions)
    A22 = SymmMixedPrec(view(A, mid+1:n, mid+1:n), uplo; precisions=remaining_precisions)

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
        offDiag_matrix = similar(A, T_OffDiag, size(offDiag_view))
        offDiag_matrix .= offDiag_view
        offDiag_scale = nothing
    end

    T_Final_Base = precisions[end]
    return SymmMixedPrec{T_Final_Base}(A11, A22, offDiag_matrix, offDiag_scale, nothing, nothing, uplo, (n, n))
end


function Base.size(A::SymmMixedPrec)
    return A.sz
end


function Base.transpose(A::SymmMixedPrec)
    return A
end


function Base.getindex(A::SymmMixedPrec{T_Base}, i::Int, j::Int) where {T_Base}
    if A.BaseCase !== nothing
        if (A.uplo == 'L' && i < j) || (A.uplo == 'U' && i > j)
             return A.BaseCase[j, i]
        else
             return A.BaseCase[i, j]
        end
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
            return A.OffDiag[j, i - mid]
        end

    else 
        if A.uplo == 'U'
            return A.OffDiag[i, j - mid]
        else 
            return A.OffDiag[j - mid, i]
        end
    end
end