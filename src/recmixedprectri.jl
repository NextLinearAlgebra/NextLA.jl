# mixed precision triangular data structure utilizing a recursive data type 

struct TriMixedPrec{T_Base} <: AbstractMatrix{T_Base}
    # T_Diag is the diagonal precision we seek ; T_OffDiag is the off diagonal precision we seek

    # A11 and A22 are the diagonal matrices - they are either another tri mixed precision matrix that will be further decomposed,
    # or, once reaching a certain depth/base case, they are Nothing 
    # OffDiag is the off diagonal matrix, which is either a matrix with the offdiag precision or Nothing in the case we are at the base case
    # Base is the base case triangular matrix in the case we are at the base case; or Nothing otherwise
    A11::Union{TriMixedPrec{T_Base}, Nothing}
    A22::Union{TriMixedPrec{T_Base}, Nothing}
    OffDiag::Union{AbstractMatrix, Nothing}
    offDiag_scale::Union{Float32, Nothing}
    base_scale::Union{Float32, Nothing}
    BaseCase::Union{AbstractMatrix{T_Base}, Nothing}
    uplo::Char # 'U' if upper tri, 'L' if lower
    sz::Tuple{Int, Int} # the size of the triangular matrix
end



# constructor for the triangular mixed precision
function TriMixedPrec(
    A::AbstractMatrix, 
    uplo::Char;
    precisions::Vector{DataType}
)
    # A : the original triangular matrix 
    # uplo : 'U' if the matrix is upper trianular and 'L' if it is lower triangular 
    # precision: a list of length the depth we want where each element corresponds to the precision we want at that depth's layer
    
    FP16_MAX_VAL = 65504.0f0
    n = size(A, 1) #A is square so m = n
    @assert n == size(A, 2) # A must be square

    # we have reached the base case as there is just one precision
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
        
        return TriMixedPrec{T_Base}(nothing, nothing, nothing, nothing, base_scale, base_matrix, uplo, (n, n)) # Updated constructor return
    end

    #otherwise we split the data structure recursively
    if isinteger(log2(n))
        mid = div(n, 2)
    else
        mid = 2 ^ floor(Int, log2(n))
    end
    
    T_OffDiag = precisions[1]
    remaining_precisions = precisions[2:end]

    # Recursively construct the diagonal blocks.
    A11 = TriMixedPrec(view(A, 1:mid, 1:mid), uplo; precisions=remaining_precisions)
    A22 = TriMixedPrec(view(A, mid+1:n, mid+1:n), uplo; precisions=remaining_precisions)

    #create off diag matrix with it's correct precision 
    local offDiag_matrix 
    local offDiag_view
    local offDiag_scale = nothing
    if uplo == 'L'
        offDiag_view = view(A, mid+1:n, 1:mid)
    else # uplo == 'U'
        offDiag_view = view(A, 1:mid, mid+1:n)
    end

    # quantization step 
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
    
    # we are at the base case
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
            return zero(T_Base) # The upper-right part of a lower-tri matrix is zero
        end
    else # i <= mid && j > mid
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



function Base.transpose(A::TriMixedPrec{T_Base}) where {T_Base}
    new_uplo = A.uplo == 'U' ? 'L' : 'U'

    if A.BaseCase !== nothing
        return TriMixedPrec{T_Base}(
            nothing,
            nothing,
            nothing,
            nothing,
            A.base_scale,
            copy(transpose(A.BaseCase)), # Transpose the base matrix
            new_uplo,
            A.sz
        )
    end

    return TriMixedPrec{T_Base}(
        transpose(A.A11),      
        transpose(A.A22),        
        copy(transpose(A.OffDiag)),    
        A.offDiag_scale,
        nothing,
        nothing,
        new_uplo,
        A.sz
    )
end


# constructor to convert a SymmMixedPrec into a TriMixedPrec
function TriMixedPrec(A::SymmMixedPrec{T_Base}) where {T_Base}
    
    if A.BaseCase !== nothing
        return TriMixedPrec{T_Base}(
            nothing, nothing, nothing, 
            nothing, A.base_scale, A.BaseCase, 
            A.uplo, A.sz
        )
    end

    return TriMixedPrec{T_Base}(
        TriMixedPrec(A.A11), # recursively convert A11
        TriMixedPrec(A.A22), # recursively convert A22
        A.OffDiag, 
        A.offDiag_scale,
        nothing,  
        nothing,  
        A.uplo,
        A.sz
    )
end