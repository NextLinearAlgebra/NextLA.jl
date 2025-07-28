using LinearAlgebra
using LinearAlgebra.BLAS
include("rectrxm.jl")
include("symmmixedprec.jl")
include("recmixedprectri.jl")
include("recsyrk.jl")

function potrf_recursive!(A, block_size)
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
    potrf_recursive!(A11, block_size)

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
    potrf_recursive!(A22, block_size)
end


function potrf_recursive!(A:: SymmMixedPrec)
    if A.BaseCase !== nothing
        potrf_recursive_A!(A.BaseCase, 4096)
        return
    end

    # Recursive POTRF on A11
    potrf_recursive!(A.A11) 

    # TRSM: A21 = A21 * inv(L11ᵀ)
    # L11 = Matrix(A11)
    # A21_mat = Matrix(A21)
    # CUBLAS.trsm!('R', 'L', 'T', 'N', 1.0, A11, A21)
    L11 = TriMixedPrec(A.A11)
    unified_rectrxm!('R', 'L', 'T', 1.0, 'S', L11, A.OffDiag)
    
    # A21 .= A21_mat

    # SYRK: A22 -= A21 * A21ᵀ
    # A22_mat = Matrix(A22)
    recsyrk!(-1.0, A.OffDiag, 1.0, A.A22)
    # A22 .= A22_mat

    # Recursive POTRF on trailing block
    potrf_recursive!(A.A22)
end


# recursive mixed prec, without trsm
function potrf_recursive_T!(A:: SymmMixedPrec)
    if A.BaseCase !== nothing
        potrf_recursive_A!(A.BaseCase, 4096)
        return
    end

    # Recursive POTRF on A11
    potrf_recursive!(A.A11) 

    # TRSM: A21 = A21 * inv(L11ᵀ)
    # L11 = Matrix(A11)
    # A21_mat = Matrix(A21)
    # CUBLAS.trsm!('R', 'L', 'T', 'N', 1.0, A11, A21)
    L11 = TriMixedPrec(A.A11)
    # unified_rectrxm!('R', 'L', 'N', 1.0, 'S', L11, A.OffDiag)
    
    # A21 .= A21_mat

    # SYRK: A22 -= A21 * A21ᵀ
    # A22_mat = Matrix(A22)
    recsyrk!(-1.0, A.OffDiag, 1.0, A.A22)
    # A22 .= A22_mat

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