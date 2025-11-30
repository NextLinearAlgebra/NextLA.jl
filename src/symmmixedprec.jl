abstract type AbstractMixedPrec{T} <: AbstractMatrix{T} end

struct TransposedMixedPrec{T, M <: AbstractMixedPrec{T}} <: AbstractMixedPrec{T}
    parent::M
end

Base.transpose(A::AbstractMixedPrec) = TransposedMixedPrec(A)

Base.parent(A::TransposedMixedPrec) = A.parent
Base.size(A::TransposedMixedPrec) = reverse(size(parent(A)))

Base.getindex(A::TransposedMixedPrec, i::Int, j::Int) = parent(A)[j, i]

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
    loc::Int #for reconstructions, location
end

function SymmMixedPrec_prealloc(
    A::AbstractMatrix,
    precisions::Vector{DataType}
		)
	mem_alloc=[]
	n=size(A,1)
       @assert n == size(A, 2) "A must be square"
       mid=n
       mid_n = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))
       for current_precision in precisions[1:end-1]:
	       mid = isinteger(log2(mid)) ? div(mid, 2) : 2^floor(Int, log2(mid))
	       mem_alloc.pushfirst!(similar(A,current_precision,mid_n,mid))
       end
       mid = isinteger(log2(mid)) ? div(mid, 2) : 2^floor(Int, log2(mid))
       mem_alloc.pushfirst!(similar(A,current_precision,n,mid))
       return mem_alloc
end

function SymmMixedPrec(
    A::AbstractMatrix,
    uplo::Char;
    precisions::Vector{DataType},
    n_start::Int=0
)
    memspace = SymmMixedPrec_prealloc(A,precisions)
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
		base_matrix = view(memspace[length(precisions)],(1:size(A,1)).+n_start,size(A,2))
                #print("CLAMPING9")
		A./=base_scale
		clamp!(A,-FP16_MAX_VAL, FP16_MAX_VAL)
		base_matrix.=A
            else
                base_scale = nothing
                base_matrix = view(memspace[length(precisions)],(1:size(A,1)).+n_start,size(A,2))
                base_matrix .= A
            end
        else
            base_matrix = view(memspace[length(precisions)],(1:size(A,1)).+n_start,size(A,2))
            base_matrix .= A
            base_scale = nothing
        end

	return SymmMixedPrec{T_Base}(nothing, nothing, nothing, nothing, base_scale, base_matrix, uplo, (n, n),n_start)
    end

    mid = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))

    T_OffDiag = precisions[1]
    remaining_precisions = precisions[2:end]

    A11 = SymmMixedPrec(view(A, 1:mid, 1:mid), uplo; precisions=remaining_precisions)
    A22 = SymmMixedPrec(view(A, mid+1:n, mid+1:n), uplo; precisions=remaining_precisions,n_start=mid)

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
            # print("this is alpha off diag:", alpha_offDiag)
            offDiag_scale = Float32(alpha_offDiag / FP16_MAX_VAL)
            offDiag_matrix = view(memspace[length(precisions)],(1:size(offDiag_view,1)).+n_start,size(offDiag_view,2))
	    offDiag_matrix ./= offDiag_scale
	    clamp!(offDiag_matrix,-FP16_MAX_VAL, FP16_MAX_VAL)
            offDiag_matrix .= offDiag_view        
       else
            offDiag_matrix = view(memspace[length(precisions)],(1:size(offDiag_view,1)).+n_start,size(offDiag_view,2))
            offDiag_matrix .= offDiag_view
        end
    else
        offDiag_matrix = view(memspace[length(precisions)],(1:size(offDiag_view,1)).+n_start,size(offDiag_view,2))
        offDiag_matrix .= offDiag_view
        offDiag_scale = nothing
    end

    T_Final_Base = precisions[end]
    return SymmMixedPrec{T_Final_Base}(A11, A22, offDiag_matrix, offDiag_scale, nothing, nothing, uplo, (n, n),n_start)
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
