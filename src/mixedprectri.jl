# mixed precision triangular data structure utilizing a recursive data type 

struct TriMixedPrec{T_Diag, T_OffDiag} <: AbstractMatrix{T_Diag}
    # T_Diag is the diagonal precision we seek ; T_OffDiag is the off diagonal precision we seek

    # A11 and A22 are the diagonal matrices - they are either another tri mixed precision matrix that will be further decomposed,
    # or, once reaching a certain depth/base case, they are Nothing 
    # OffDiag is the off diagonal matrix, which is either a matrix with the offdiag precision or Nothing in the case we are at the base case
    # Base is the base case triangular matrix in the case we are at the base case; or Nothing otherwise
    A11::Union{TriMixedPrec{T_Diag, T_OffDiag}, Nothing}
    A22::Union{TriMixedPrec{T_Diag, T_OffDiag}, Nothing}
    OffDiag::Union{AbstractMatrix{T_OffDiag}, Nothing}
    Base::Union{Matrix{T_Diag}, Nothing}
    uplo::Char # 'U' if upper tri, 'L' if lower
    sz::Tuple{Int, Int} # the size of the triangular matrix
end

# constructor for the triangular mixed precision
function TriMixedPrec(
    A::AbstractMatrix, 
    uplo::Char; 
    T_Diag::Type, 
    T_OffDiag::Type, 
    threshold::Int
)
    # A : the original triangular matrix 
    # uplo : 'U' if the matrix is upper trianular and 'L' if it is lower triangular 
    # T_Diag : the float precision you want on the diagonal blocks (i.e. Float64)
    # T_OffDiag : the float precision you want on the off diagonal blocks (i.e. Float64)
    # threshold : after the threshold size we begin to switch from the OffDiag precision to the Diag precision 
    
    n = size(A, 1) #A is square so m = n
    @assert n == size(A, 2) # A must be square

    # we have reached 
    if n <= threshold
        local base
        if eltype(A) == T_Diag
            base = copy(A) # type is right, just copy.
        else
            base = Matrix{T_Diag}(A) # type is wrong, convert and copy.
        end
        return TriMixedPrec{T_Diag, T_OffDiag}(
            nothing, #A11 is nothing because we are on the base case
            nothing, #A22 is nothing because we are on the base case
            nothing, #off diag block is nothing because we are on the base case
            base,
            uplo,
            (n, n)
        )
    end
    #otherwise we split the data structure recursively
    if isinteger(log2(n))
        mid = div(n, 2)
    else
        mid = 2 ^ floor(Int, log2(n))
    end
    #split A11 and A22 with the same uplo, theshold, and precisions - these are tri mixed prec as well
    A11 = TriMixedPrec(view(A, 1:mid, 1:mid), uplo; T_Diag=T_Diag, T_OffDiag=T_OffDiag, threshold=threshold)
    A22 = TriMixedPrec(view(A, mid+1:n, mid+1:n), uplo; T_Diag=T_Diag, T_OffDiag=T_OffDiag, threshold=threshold)

    #create off diag matrix with it's correct precision 
    local offDiag
    if uplo == 'L'
        offdiag_view = view(A, mid+1:n, 1:mid)
        if eltype(offdiag_view) == T_OffDiag
            offDiag = copy(offdiag_view)
        else
            offDiag = Matrix{T_OffDiag}(offdiag_view)
        end
    else # uplo == 'U'
        offdiag_view = view(A, 1:mid, mid+1:n)
            if eltype(offdiag_view) == T_OffDiag
            offDiag = copy(offdiag_view)
        else
            offDiag = Matrix{T_OffDiag}(offdiag_view)
        end
    end

    return TriMixedPrec{T_Diag, T_OffDiag}(
        A11,
        A22,
        offDiag, 
        nothing,  # base is nothing because this is not a base case
        uplo,
        (n, n)
    )
end

function Base.size(A::TriMixedPrec)
    return A.sz
end