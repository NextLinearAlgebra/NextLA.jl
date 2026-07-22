"""
    FullMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}

A hierarchical, recursive mixed-precision data structure that maps to general full dense square matrices.
It recursively partitions the matrix into two square diagonal sub-blocks (`A11`, `A22`) and two 
corresponding off-diagonal rectangular blocks (`A12`, `A21`).
"""
abstract type AbstractMixedPrec{T} <: AbstractMatrix{T} end

struct TransposedMixedPrec{T, M <: AbstractMixedPrec{T}} <: AbstractMixedPrec{T}
    parent::M
end

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
    FullMixedPrec(A::AbstractMatrix; precisions::Vector{DataType})

Constructs a `FullMixedPrec` representation of the general square matrix `A`.

Uses a base-2 recursive splitting scheme to partition the matrix into smaller square diagonal blocks 
and rectangular off-diagonal views. To preserve bounds during `Float16` conversion, per-block dynamic 
quantization handles elements by detecting values exceeding `65504.0f0`, computing a scaling factor, 
and applying clamping. This ensures high-magnitude structures safely avoid numerical overflow across the memory layout.
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

    view_A12 = view(A, 1:mid, mid+1:n)
    view_A21 = view(A, mid+1:n, 1:mid)

    local mat_A12, mat_A21
    local scale_A12 = nothing
    local scale_A21 = nothing

    if T_OffDiag == Float16
        alpha_12 = maximum(abs, view_A12)
        if alpha_12 > FP16_MAX_VAL
            scale_A12 = Float32(alpha_12 / FP16_MAX_VAL)
            mat_A12 = similar(view_A12, Float16, size(view_A12))
            @. mat_A12 = Float16(round(clamp(view_A12 / scale_A12, -FP16_MAX_VAL, FP16_MAX_VAL)))
        else
            mat_A12 = similar(view_A12, Float16, size(view_A12))
            mat_A12 .= view_A12
        end

        alpha_21 = maximum(abs, view_A21)
        if alpha_21 > FP16_MAX_VAL
            scale_A21 = Float32(alpha_21 / FP16_MAX_VAL)
            mat_A21 = similar(view_A21, Float16, size(view_A21))
            @. mat_A21 = Float16(round(clamp(view_A21 / scale_A21, -FP16_MAX_VAL, FP16_MAX_VAL)))
        else
            mat_A21 = similar(view_A21, Float16, size(view_A21))
            mat_A21 .= view_A21
        end
    else
        mat_A12 = similar(A, T_OffDiag, size(view_A12))
        mat_A12 .= view_A12
        mat_A21 = similar(A, T_OffDiag, size(view_A21))
        mat_A21 .= view_A21
    end

    T_Final_Base = precisions[end]
    return FullMixedPrec{T_Final_Base}(A11, A22, mat_A12, mat_A21, scale_A12, scale_A21, nothing, nothing, (n, n))
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
    elseif i <= mid && j > mid
        return A.A12[i, j - mid]
    else
        return A.A21[i - mid, j]
    end
end

function Base.sizeof(A::FullMixedPrec)
    if A.BaseCase !== nothing
        return sizeof(A.BaseCase)
    end
    return sizeof(A.A11) + sizeof(A.A22) + sizeof(A.A12) + sizeof(A.A21)
end

"""
    TriMixedPrec(A::FullMixedPrec{T_Base}, uplo::Char) where {T_Base}

Dynamically converts an existing `FullMixedPrec` matrix structure into a `TriMixedPrec` format 
for a specified triangle ('U' or 'L'). Allows seamless interoperability with `unified_rectrxm!`.
"""
function TriMixedPrec(A::FullMixedPrec{T_Base}, uplo::Char) where {T_Base}
    if A.BaseCase !== nothing
        return TriMixedPrec{T_Base}(
            nothing, nothing, nothing,
            nothing, A.base_scale, A.BaseCase,
            uplo, A.sz
        )
    end

    offdiag_mat = (uplo == 'L') ? A.A21 : A.A12
    offdiag_sc  = (uplo == 'L') ? A.A21_scale : A.A12_scale

    return TriMixedPrec{T_Base}(
        TriMixedPrec(A.A11, uplo),
        TriMixedPrec(A.A22, uplo),
        offdiag_mat,
        offdiag_sc,
        nothing,
        nothing,
        uplo,
        A.sz
    )
end

"""
    reconstruct_matrix(A::FullMixedPrec{T_Base})

Reconstructs a full dense matrix from the general mixed-precision recursive block structure `A`. 
Used primarily for validation and returning to standard dense formats.
"""
function reconstruct_matrix(A::FullMixedPrec{T_Base}) where {T_Base}
    if A.BaseCase !== nothing
        return copy(A.BaseCase)
    end
    
    C11 = reconstruct_matrix(A.A11)
    C22 = reconstruct_matrix(A.A22)
    C12 = A.A12
    C21 = A.A21
    n1, m1 = size(C11)
    n2, m2 = size(C22)
    n = n1 + n2

    C_full = similar(C11, T_Base, n, n)
    C_full[1:n1, 1:m1] .= C11
    C_full[1:n1, m1+1:n] .= C12
    C_full[n1+1:n, 1:m1] .= C21
    C_full[n1+1:n, m1+1:n] .= C22

    return C_full
end

"""
    getrf!(A::CUDA.StridedCuArray)
    getrf!(A::AbstractMatrix)

Base case non-pivoting LU factorization wrapper routing to CUSOLVER or CPU fallback routines.
"""
function getrf!(A::CUDA.StridedCuArray)
    return CUSOLVER.getrf!(A)
end

function getrf!(A::AbstractMatrix)
    return LinearAlgebra.lu!(A, LinearAlgebra.NoPivot())
end

"""
    _gemm_dispatch!(transA::Char, transB::Char, alpha::Number, A, B, beta::Number, C)

Handles type-conversion, precision clamping, and hardware routing for general matrix multiplications.
"""
function _gemm_dispatch!(
    transA::Char, transB::Char,
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::AbstractMatrix
)
    TC = eltype(C)
    TA = eltype(A)
    TB = eltype(B)
    if TA == TB == TC && TC in (Float32, Float64)
        gemm!(transA, transB, TC(alpha), A, B, TC(beta), C)
    elseif TA == Float16 && TB == Float16 && TC in (Float16, Float32)
        gemmEx!(transA, transB, alpha, A, B, beta, C)
    else
        compute_type = (TC in (Float32, Float64)) ? TC : Float32
        A_final = (TA == compute_type) ? A : compute_type.(A)
        B_final = (TB == compute_type) ? B : compute_type.(B)
        C_temp  = (TC == compute_type) ? C : compute_type.(C)
        
        if compute_type in (Float32, Float64)
            gemm!(transA, transB, compute_type(alpha), A_final, B_final, compute_type(beta), C_temp)
        else 
            gemmEx!(transA, transB, alpha, A_final, B_final, beta, C_temp)
        end
        
        if C !== C_temp
            clamp!(C_temp, floatmin(TC), floatmax(TC))
            copy!(C, C_temp)
        end
    end
end

"""
    _recgemm_impl!(alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::AbstractMatrix, threshold::Int; parallel::Bool)

Internal implementation for nested recursive general matrix multiplication updates on dense views.
"""
function _recgemm_impl!(
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::AbstractMatrix,
    threshold::Int; parallel::Bool
)
    n, m = size(C)
    if n <= threshold || m <= threshold || size(A, 2) <= threshold
        _gemm_dispatch!('N', 'N', alpha, A, B, beta, C)
        return
    end

    n1 = 2^floor(Int, log2(n)) ÷ 2
    m1 = 2^floor(Int, log2(m)) ÷ 2

    A1 = @view A[1:n1, :]; A2 = @view A[n1+1:end, :]
    B1 = @view B[:, 1:m1]; B2 = @view B[:, m1+1:end]
    
    C11 = @view C[1:n1, 1:m1]; C12 = @view C[1:n1, m1+1:end]
    C21 = @view C[n1+1:end, 1:m1]; C22 = @view C[n1+1:end, m1+1:end]

    if parallel
        @sync begin
            @async _recgemm_impl!(alpha, A1, B1, beta, C11, threshold, parallel=false)
            @async _recgemm_impl!(alpha, A1, B2, beta, C12, threshold, parallel=false)
            @async _recgemm_impl!(alpha, A2, B1, beta, C21, threshold, parallel=false)
            @async _recgemm_impl!(alpha, A2, B2, beta, C22, threshold, parallel=false)
        end
    else
        _recgemm_impl!(alpha, A1, B1, beta, C11, threshold, parallel=false)
        _recgemm_impl!(alpha, A1, B2, beta, C12, threshold, parallel=false)
        _recgemm_impl!(alpha, A2, B1, beta, C21, threshold, parallel=false)
        _recgemm_impl!(alpha, A2, B2, beta, C22, threshold, parallel=false)
    end
end

"""
    _recgemm_impl!(alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::FullMixedPrec; parallel::Bool)

Internal implementation for nested recursive general matrix multiplication updates specifically for the `FullMixedPrec` block structure.
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

    _gemm_dispatch!('N', 'N', alpha, A1, B2, beta, C.A12)
    _gemm_dispatch!('N', 'N', alpha, A2, B1, beta, C.A21)

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

Performs an in-place, nested recursive block general matrix multiplication update on a full mixed-precision matrix structure.
Falls back to standard hardware routines using the dispatch helper at the base case.
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

Performs an in-place, nested recursive block general matrix multiplication update (`C = alpha*A*B + beta*C`).
Falls back to standard hardware routines using the dispatch helper at the specified base case threshold.
"""
function recgemm!(
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::AbstractMatrix, threshold::Int=256
)
    should_parallelize = size(C, 1) > PARALLEL_THRESHOLD
    _recgemm_impl!(alpha, A, B, beta, C, threshold, parallel=should_parallelize)
end

"""
    getrf_recursive!(A, block_size)

Performs an in-place, nested recursive non-pivoting block LU factorization on matrix `A`.
The recursion dynamically partitions the matrix until sub-blocks are less than or equal to `block_size`,
falling back to CUSOLVER/standard hardware routines at the base case.
"""
function getrf_recursive!(A, block_size)
    n = size(A, 1)

    if n <= block_size
        getrf!(A)
        return
    end

    n1 = 2^floor(Int, log2(n)) ÷ 2  
    
    A11 = @view A[1:n1, 1:n1]
    A12 = @view A[1:n1, n1+1:end]
    A21 = @view A[n1+1:end, 1:n1]
    A22 = @view A[n1+1:end, n1+1:end]

    # Step 1: Factorize A11 -> L11 * U11
    getrf_recursive!(A11, block_size)

    # Step 2: Solve L11 * U12 = A12 (Left side, Lower triangle, NoTrans, Unit diagonal)
    if (eltype(A11) == Float16)
        unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', A11, A12)
    else
        trsm!('L', 'L', 'N', 'U', 1.0, A11, A12)
    end

    # Step 3: Solve L21 * U11 = A21 (Right side, Upper triangle, NoTrans, Non-Unit diagonal)
    if (eltype(A11) == Float16)
        unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', A11, A21)
    else
        trsm!('R', 'U', 'N', 'N', 1.0, A11, A21)
    end

    # Step 4: Schur complement update A22 = A22 - L21 * U12
    if (eltype(A21) == Float16 || eltype(A12) == Float16)
        recgemm!(-1.0, A21, A12, 1.0, A22)
    else
        gemm!('N', 'N', -1.0, A21, A12, 1.0, A22)
    end
    
    # Step 5: Factorize Schur complement A22 -> L22 * U22
    getrf_recursive!(A22, block_size)
end

"""
    getrf_recursive!(A::FullMixedPrec)

Performs an in-place, nested recursive block LU factorization on a general square 
mixed-precision matrix structure `A`. Handles off-diagonal triangular solves and Schur 
complement updates across hierarchical precisions, falling back to CUSOLVER routines at the base case.
"""
function getrf_recursive!(A::FullMixedPrec)
    if A.BaseCase !== nothing
        getrf_recursive!(A.BaseCase, 4096)
        return
    end

    # Step 1: Factorize A11 -> L11 * U11
    getrf_recursive!(A.A11) 

    # Step 2: Solve L11 * U12 = A12 (Left side, Lower triangle, NoTrans, Unit diagonal)
    # Wrapping A.A11 in TriMixedPrec safely emphasizes its lower-triangular operator nature
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', TriMixedPrec(A.A11, 'L'), A.A12)

    # Step 3: Solve L21 * U11 = A21 (Right side, Upper triangle, NoTrans, Non-Unit diagonal)
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', TriMixedPrec(A.A11, 'U'), A.A21)

    # Step 4: Schur complement update A22 = A22 - L21 * U12 across hierarchical sub-blocks
    recgemm!(-1.0, A.A21, A.A12, 1.0, A.A22)

    # Step 5: Factorize A22 -> L22 * U22
    getrf_recursive!(A.A22)
end