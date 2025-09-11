using LinearAlgebra
using StochasticRounding
include("symmmixedprec.jl")
include("recmixedprectri.jl")
include("trsm.jl")
include("trmm.jl")
include("matmul.jl")
include("rectrxm.jl")
include("recsyrk.jl")
include("cholesky.jl")



function potrf!(A::CUDA.StridedCuArray{T}) where T
    if eltype(A) == Float16
        A_f32 = Float32.(A)
        CUSOLVER.potrf!('L', A_f32)
        A .= Float16.(A_f32)
    else
        CUSOLVER.potrf!('L', A)
    end
end

function potrf!(A::AMDGPU.StridedROCArray{T}) where T
    if eltype(A) == Float16
        A_f32 = Float32.(A)
        ROCSOLVER.potrf!('L', A_f32)
        A .= Float16.(A_f32)
    else
        ROCSOLVER.potrf!('L', A)
    end
end

function potrf!(A::oneAPI.oneDeviceArray{T}) where T
    if eltype(A) == Float16
        A_f32 = Float32.(A)
        oneMKL.potrf!('L', A_f32)
        A .= Float16.(A_f32)
    else
        oneMKL.potrf!('L', A)
    end
end

function potrf_recursive!(A, block_size)
    n = size(A, 1)

    # Print a message when entering the function to trace the recursion
    # println("[TRACE] potrf_recursive! called on $(n)x$(n) matrix of type $(eltype(A))")

    if n <= block_size
        potrf!(A)
        return
    end

    n1 = 2^floor(Int, log2(n)) ÷ 2  
    
    A11 = @view A[1:n1, 1:n1]
    A21 = @view A[n1+1:end, 1:n1]
    A22 = @view A[n1+1:end, n1+1:end]

    # Recursive POTRF on A11
    potrf_recursive!(A11, block_size)

    # TRSM: A21 = A21 * inv(L11ᵀ)
    if (eltype(A11) == Float16)
        unified_rectrxm!('R', 'L', 'T', 1.0, 'S', A11, A21)
    else
        CUBLAS.trsm!('R', 'L', 'T', 'N', 1.0, A11, A21)
    end

    # SYRK: A22 -= A21 * A21ᵀ
    if (eltype(A21) == Float16)
        recsyrk!(-1.0, A21, 1.0, A22)
    else
        CUBLAS.syrk!('L', 'N', -1.0, A21, 1.0, A22)
    end
    
    potrf_recursive!(A22, block_size)
end


function reconstruct_matrix(A::SymmMixedPrec{T_Base}) where {T_Base}
    if A.BaseCase !== nothing
        return copy(A.BaseCase)
    end
    
    C11 = reconstruct_matrix(A.A11)
    C22 = reconstruct_matrix(A.A22)
    C21 = A.OffDiag
    n1, m1 = size(C11)
    n2, m2 = size(C22)
    n = n1 + n2

    C_full = CuArray{T_Base}(undef, n, n)
    C_full[1:n1, 1:m1] .= C11
    C_full[n1+1:n, 1:m1] .= C21
    C_full[n1+1:n, m1+1:n] .= C22
    C_full[1:n1, m1+1:n] .= transpose(C21)

    return C_full
end

function potrf_recursive!(A:: SymmMixedPrec)
    if A.BaseCase !== nothing
        potrf_recursive!(A.BaseCase, 4096)
        return
    end

    # Recursive POTRF on A11
    potrf_recursive!(A.A11) 

    # TRSM: A21 = A21 * inv(L11ᵀ)
    unified_rectrxm!('R', 'L', 'T', 1.0, 'S', TriMixedPrec(A.A11), A.OffDiag)

    # SYRK: A22 -= A21 * A21ᵀ
    recsyrk!(-1.0, A.OffDiag, 1.0, A.A22)

    # Recursive POTRF on trailing block
    potrf_recursive!(A.A22)
end




#no nested recursion at all
function potrf_recursive_A!(A, block_size)
    n = size(A, 1)

    if n <= block_size
        # Base case: do regular Cholesky
        CUSOLVER.potrf!('L', A)
        return
    end

    # Recursive split
    n1 = 2^floor(Int, log2(n)) ÷ 2  # largest power-of-2 less than n


    # View subblocks
    A11 = @view A[1:n1, 1:n1]
    A21 = @view A[n1+1:end, 1:n1]
    A22 = @view A[n1+1:end, n1+1:end]

    # Recursive POTRF on A11
    potrf_recursive_A!(A11, block_size)

    # TRSM: A21 = A21 * inv(L11ᵀ)
    # L11 = Matrix(A11)
    # A21_mat = Matrix(A21)
    CUBLAS.trsm!('R', 'L', 'T', 'N', 1.0, A11, A21)
    # A21 .= A21_mat

    # SYRK: A22 -= A21 * A21ᵀ
    # A22_mat = Matrix(A22)
    # CUBLAS.gemm!('N', 'T', -1.0, A21, A21, 1.0, A22) #recursive syrk with mixed precision 
    CUBLAS.syrk!('L', 'N', -1.0, A21, 1.0, A22)
    # A22 .= A22_mat

    # Recursive POTRF on trailing block
    potrf_recursive_A!(A22, block_size)
end



# only nested recsyrk
function potrf_recursive_B!(A, block_size)
    n = size(A, 1)

    if n <= block_size
        # Base case: do regular Cholesky
        CUSOLVER.potrf!('L', A)
        return
    end

    # Recursive split
    n1 = 2^floor(Int, log2(n)) ÷ 2  # largest power-of-2 less than N

    # View subblocks
    A11 = @view A[1:n1, 1:n1]
    A21 = @view A[n1+1:end, 1:n1]
    A22 = @view A[n1+1:end, n1+1:end]

    # Recursive POTRF on A11
    potrf_recursive_B!(A11, block_size)

    # TRSM: A21 = A21 * inv(L11ᵀ)
    # L11 = Matrix(A11)
    # A21_mat = Matrix(A21)
    # unified_rectrxm!('R', 'L', 'T', 1.0, 'S', A11, A21)
    CUBLAS.trsm!('R', 'L', 'T', 'N', 1.0, A11, A21)
    # A21 .= A21_mat

    # SYRK: A22 -= A21 * A21ᵀ
    # A22_mat = Matrix(A22)
    # CUBLAS.gemm!('N', 'T', -1.0, A21, A21, 1.0, A22) #recursive syrk with mixed precision 
    # CUBLAS.syrk!('L', 'N', -1.0, A21, 1.0, A22)
    recsyrk!(-1.0, A21, 1.0, A22, 256) 
    # A22 .= A22_mat

    # Recursive POTRF on trailing block
    potrf_recursive_B!(A22, block_size)
end



# only nested rectrsm
function potrf_recursive_C!(A, block_size)
    n = size(A, 1)

    if n <= block_size
        # Base case: do regular Cholesky
        CUSOLVER.potrf!('L', A)
        return
    end

    # Recursive split
    n1 = 2^floor(Int, log2(n)) ÷ 2  # largest power-of-2 less than N

    # View subblocks
    A11 = @view A[1:n1, 1:n1]
    A21 = @view A[n1+1:end, 1:n1]
    A22 = @view A[n1+1:end, n1+1:end]

    # Recursive POTRF on A11
    potrf_recursive_C!(A11, block_size)

    # TRSM: A21 = A21 * inv(L11ᵀ)
    # L11 = Matrix(A11)
    # A21_mat = Matrix(A21)
    # CUBLAS.trsm!('R', 'L', 'T', 'N', 1.0, A11, A21)
    unified_rectrxm!('R', 'L', 'T', 1.0, 'S', A11, A21)
    # A21 .= A21_mat

    # SYRK: A22 -= A21 * A21ᵀ
    # A22_mat = Matrix(A22)
    # CUBLAS.gemm!('N', 'T', -1.0, A21, A21, 1.0, A22) #recursive syrk with mixed precision
    # recsyrk!(-1.0, A21, 1.0, A22, 256) 
    CUBLAS.syrk!('L', 'N', -1.0, A21, 1.0, A22)
    # A22 .= A22_mat

    # Recursive POTRF on trailing block
    potrf_recursive_C!(A22, block_size)
end



# full nested (both rectrsm and recsyrk)
function potrf_recursive_D!(A, block_size)
    n = size(A, 1)

    if n <= block_size
        # Base case: do regular Cholesky
        CUSOLVER.potrf!('L', A)
        return
    end

    # Recursive split
    n1 = 2^floor(Int, log2(n)) ÷ 2  # largest power-of-2 less than n

    # View subblocks
    A11 = @view A[1:n1, 1:n1]
    A21 = @view A[n1+1:end, 1:n1]
    A22 = @view A[n1+1:end, n1+1:end]

    # Recursive POTRF on A11
    potrf_recursive_D!(A11, block_size)

    # TRSM: A21 = A21 * inv(L11ᵀ)
    # L11 = Matrix(A11)
    # A21_mat = Matrix(A21)
    unified_rectrxm!('R', 'L', 'T', 1.0, 'S', A11, A21)
    # A21 .= A21_mat

    # SYRK: A22 -= A21 * A21ᵀ
    # A22_mat = Matrix(A22)
    # CUBLAS.gemm!('N', 'T', -1.0, A21, A21, 1.0, A22) #recursive syrk with mixed precision
    recsyrk!(-1.0, A21, 1.0, A22, 256) 
    # A22 .= A22_mat

    # Recursive POTRF on trailing block
    potrf_recursive_D!(A22, block_size)
end