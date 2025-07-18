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
    n2 = n - n1

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


function potrf_recursive!(A:: SymmMixedPrec, block_size)
    if A.BaseCase !== nothing
        if eltype(A.BaseCase) == Float16 && A.base_scale !== nothing
            A_f32 = Float32.(A.BaseCase) .* A.base_scale
            CUSOLVER.potrf!('L', A_f32)
            A.BaseCase .= Float16.(clamp.(A_f32 ./ A.base_scale, floatmin(eltype(A.BaseCase)), floatmax(eltype(A.BaseCase))))
        else
            CUSOLVER.potrf!('L', A.BaseCase)
        end
        return
    end

    # Recursive POTRF on A11
    potrf_recursive!(A.A11, block_size) # will this be in place or return a tri mixed prec or an abstract matrix?

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
    potrf_recursive!(A.A22, block_size)
end


function test_potrf_recursive(n, block_size)
    A = randn(n, n)
    A = CuArray(A * A' + n * I)
    A_orig = copy(A)

    potrf_recursive!(A, block_size)

    L = tril(A)
    A_reconstructed = L * L'
    residual = norm(A_orig - A_reconstructed)
    rel_error = residual / norm(A_orig)

    println("Matrix size: $n × $n")
    println("Block cutoff size: $block_size")
    println("Relative error: $rel_error")
end




