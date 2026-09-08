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

@kernel function lu_base_kernel!(A, P, M, max_ind, col_start, col_stop, row_start, row_stop)
    I, J = @index(Global, NTuple)
    temp = zero(eltype(A))
    
    for k = row_start:row_stop
        col_k = k - row_start + col_start      
        offset = max(0, col_k)
        
        # 1. Find pivot (Thread 1, 1 does this to prevent race condition)
        if I == 1 && J == 1
            max_val = -1.0
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
        end
        @synchronize
        
        # 2. Swap rows in A and P
        if I <= row_stop - row_start + 1 && J == 1
            swap_rows!(A, col_k, max_ind[1], row_start, temp, I)
            swap_rows!(P, col_k, max_ind[1], row_start, temp, I)
        end
        @synchronize

        # 3. Compute L values (divide by pivot)
        if I <= col_stop - offset && J == 1
            get_l_vals!(A, k, offset, row_start, col_start, I, temp)
        end
        @synchronize
        
        # 4. Schur complement update for the trailing matrix
        if I+col_start > offset && I+col_start <= col_stop && J+row_start <= row_stop && J+row_start > k
            A[I+col_start, J+row_start] = A[I+col_start, J+row_start] - A[I+col_start, k]*A[k, J+row_start]
        end
        @synchronize
    end
end

@kernel function P_kernel_base!(P)
    I = @index(Global)
    P[I, I] = 1
end

function lu_base!(A::AbstractMatrix{T}) where T
    backend = KernelAbstractions.get_backend(A)
    n = size(A, 1)
    
    P = KernelAbstractions.zeros(backend, T, n, n)
    grid = min(n, 256)
    P_kernel_base!(backend, grid)(P, ndrange=n)
    
    M = KernelAbstractions.zeros(backend, T, 1)
    max_ind = KernelAbstractions.zeros(backend, Int, 1)
    
    lu_base_kernel!(backend, (n, n))(
        A, P, M, max_ind, 1, n, 1, n, 
        ndrange = (n, n)
    )
    KernelAbstractions.synchronize(backend)
    return A, P
end
