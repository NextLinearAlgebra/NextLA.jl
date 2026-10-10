export lu_base!

@inline function get_l_vals!(A, k, p, row_start, col_start, I, temp)
    A[I+p, k] = A[I+p, k] / A[k, k]
end

@inline function swap_rows!(A, i, j, start_row, temp, I)
    I_idx = I + start_row - 1
    temp = A[i, I_idx]
    @inbounds A[i, I_idx] = A[j, I_idx]
    @inbounds A[j, I_idx] = temp
end

# col_k, offset and temp are recomputed in every block delimited by a
# @synchronize. On the CPU backend a @synchronize splits the kernel body into
# separate ndrange loops and only @localmem/@private survive the split, so a
# plain local assigned before it is out of scope afterwards — the kernel failed
# with `UndefVarError: col_k not defined`. Same fix as src/gemm/matmul.jl.
@kernel function lu_base_kernel!(A, ipiv, M, max_ind, col_start, col_stop, row_start, row_stop)
    I, J = @index(Global, NTuple)

    for k = row_start:row_stop
        col_k = k - row_start + col_start      
        offset = max(0, col_k)
        
        # 1. Find pivot (Thread 1, 1 does this to prevent race condition)
        if I == 1 && J == 1
            max_val = -one(real(eltype(A)))
            max_i = col_k
            for j = col_k:col_stop
                val = abs(A[j, k])
                if val > max_val
                    max_val = val
                    max_i = j
                end
            end 
            M[1] = max_val
            max_ind[1] = max_i
            ipiv[col_k] = max_i
        end
        @synchronize

        # 2. Swap rows in A
        col_k = k - row_start + col_start
        temp = zero(eltype(A))
        if I <= row_stop - row_start + 1 && J == 1
            swap_rows!(A, col_k, max_ind[1], row_start, temp, I)
        end
        @synchronize

        # 3. Compute L values (divide by pivot)
        col_k = k - row_start + col_start
        offset = max(0, col_k)
        temp = zero(eltype(A))
        if I <= col_stop - offset && J == 1
            get_l_vals!(A, k, offset, row_start, col_start, I, temp)
        end
        @synchronize

        # 4. Schur complement update for the trailing matrix
        offset = max(0, k - row_start + col_start)
        if I+col_start > offset && I+col_start <= col_stop && J+row_start <= row_stop && J+row_start > k
            A[I+col_start, J+row_start] = A[I+col_start, J+row_start] - A[I+col_start, k]*A[k, J+row_start]
        end
        @synchronize
    end
end

# LAPACK's ipiv -- row k was swapped with row ipiv[k], in order -- as a
# permutation p with A0[p, :] == L * U.
function _ipiv_to_perm(ipiv::AbstractVector{<:Integer})
    p = collect(1:length(ipiv))
    for k in eachindex(ipiv)
        p[k], p[ipiv[k]] = p[ipiv[k]], p[k]
    end
    return p
end

# LAPACK's info for a factored A: the first k with U[k, k] zero, or 0. A zero
# pivot is divided through, so the entries after it can be NaN; those count too.
function _lu_info(A::AbstractMatrix)
    d = Array(LinearAlgebra.diag(A))
    k = findfirst(x -> iszero(x) || !isfinite(x), d)
    return k === nothing ? 0 : k
end

"""
    lu_base!(A) -> (A, p, info)

LU with partial pivoting in one work-group, in place: `L` (unit diagonal) and `U` are packed
into `A`, and `p` is the row permutation as a vector, `A0[p, :] == L * U`. As LAPACK's getrf,
`info` is 0 or the first `k` with a zero `U[k, k]`, and nothing is thrown. On a GPU `n` must be
at most 32.
"""
function lu_base!(A::AbstractMatrix{T}) where T
    backend = KernelAbstractions.get_backend(A)
    n = size(A, 1)
    # One work-group of n x n threads synchronises over the whole matrix, so on a
    # device the 1024-thread limit caps n at 32.
    backend isa KernelAbstractions.CPU || n <= 32 ||
        throw(ArgumentError("lu_base! runs one n x n work-group, so n must be at most 32 on a GPU, got $n"))

    ipiv = KernelAbstractions.zeros(backend, Int, n)
    
    M = KernelAbstractions.zeros(backend, T, 1)
    max_ind = KernelAbstractions.zeros(backend, Int, 1)
    
    lu_base_kernel!(backend, (n, n))(
        A, ipiv, M, max_ind, 1, n, 1, n, 
        ndrange = (n, n)
    )
    KernelAbstractions.synchronize(backend)
    return A, _ipiv_to_perm(Array(ipiv)), _lu_info(A)
end
