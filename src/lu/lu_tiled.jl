export tile_lu_factor!

@kernel function P_kernel_tiled!(P)
    I = @index(Global)
    P[I, I] = 1
end

@inline function swap_rows_tiled!(A, i, j, start_row, temp, I)
    I_idx = I + start_row - 1
    temp = A[i, I_idx]
    @inbounds A[i, I_idx] = A[j, I_idx]
    @inbounds A[j, I_idx] = temp
end

@inline function get_l_vals_tiled!(A, k, p, row_start, col_start, I, temp)
    A[I+p, k] = A[I+p, k] / A[k, k]
end

@kernel function lu_gpu_tiled!(A, P, M, max_ind, pivot, next, col_start, col_stop, row_start, row_stop)
    I, J = @index(Global, NTuple)
    temp = zero(eltype(A))
    
    for k = row_start:row_stop
        col_k = k - row_start + col_start      
        offset = max(next-1, col_k)
        
        # find max in col
        if pivot
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
            
            if I <= row_stop - row_start + 1 && J == 1
                swap_rows_tiled!(A, col_k, max_ind[1], row_start, temp, I)
                swap_rows_tiled!(P, col_k, max_ind[1], row_start, temp, I)
            end
            @synchronize
        end

        if I <= col_stop - offset && J == 1
            get_l_vals_tiled!(A, k, offset, row_start, col_start, I, temp)
        end
        
        @synchronize
        
        if I+col_start > offset && I+col_start <= col_stop && J+row_start <= row_stop && J+row_start > k
            A[I+col_start, J+row_start] = A[I+col_start, J+row_start] - A[I+col_start, k]*A[k, J+row_start]
        end
        @synchronize
    end
end

function dgetrf_tiled!(A, P, k, n, M, max_ind, backend)
    lu_gpu_tiled!(backend, (n, n))(A, P, M, max_ind, true, 1, (k-1)*n+1, k*n, (k-1)*n+1, k*n, ndrange = (n, n))
end

@kernel function L_inv_single_kernel!(L, L_inv, n, k_ind)
    I = @index(Global)
    k_tile = (k_ind-1)*n
    
    for i = 2:n
        if I <= i-1
            L_inv[i, I] = 0
        end
        @synchronize
        for k = 1:i-1 
            if I <= i-1
                L_inv[i, I] -= (L[i+k_tile, k+k_tile]*L_inv[k, I])
            end
            @synchronize
        end
    end
end

@kernel function dgessm_kernel_P!(A, P, k_ind, offset, n, temp)
    i, j, j_ind = @index(Global, NTuple)
    j_ind = offset+j_ind

    for s = 1:n
        temp += P[i+(k_ind-1)*n, s+(k_ind-1)*n] * A[s+(k_ind-1)*n, j+(j_ind-1)*n]
    end

    A[i+(k_ind-1)*n, j+(j_ind-1)*n] = temp
end

@kernel function dgessm_kernel_L_inv!(A, L_inv, k_ind, n, temp)
    i, j, j_ind = @index(Global, NTuple)
    j_ind = j_ind + k_ind

    for s = 1:n
        temp += L_inv[i,s] * A[s+(k_ind-1)*n, j+(j_ind-1)*n]
    end

    A[i+(k_ind-1)*n, j+(j_ind-1)*n] = temp
end

@kernel function single_dtstrf_lu_gpu!(A, next, k, tiles_col, row_start, row_stop)
    I, J = @index(Global, NTuple)
    temp = zero(eltype(A))
    n = @uniform @groupsize()[1]
    
    local_A = @localmem eltype(A) (n,n)
    
    for i = k+1:tiles_col
        col_start = (i-1)*n
        col_stop = (i)*n
    
        local_A[I, J] = A[I+col_start, J+row_start-1]
        @synchronize

        for k_inner = row_start:row_stop
            col_k = k_inner-row_start+col_start      
            offset = max(next-1, col_k)

            if J == 1
                local_A[I, k_inner-row_start+1] = local_A[I, k_inner-row_start+1]/A[k_inner, k_inner]
            end
            @synchronize

            if J+k_inner <= row_stop
                local_A[I, J+k_inner-row_start+1] = local_A[I, J+k_inner-row_start+1] - local_A[I, k_inner-row_start+1]*A[k_inner, J+k_inner]
            end
            @synchronize
        end

        A[I+col_start, J+row_start-1] = local_A[I, J]
        @synchronize
    end
end

@kernel function dssssm_kernel!(A, k, n, temp)
    i, j, j_ind, i_ind = @index(Global, NTuple)
    j_ind = j_ind + k
    i_ind = i_ind + k

    for s = 1:n
        temp -= A[i+(j_ind-1)*n, s+(k-1)*n] * A[s+(k-1)*n, j+(i_ind-1)*n]
    end

    A[i+(j_ind-1)*n, j+(i_ind-1)*n] += temp
end

function tile_lu_factor!(A::AbstractMatrix{T}, n::Int) where T
    backend = KernelAbstractions.get_backend(A)
    
    num_rows = size(A, 2) 
    num_cols = size(A, 1)
    
    M = KernelAbstractions.zeros(backend, T, 2)
    max_ind = KernelAbstractions.zeros(backend, Int, 2)
    L_for_gessm = KernelAbstractions.zeros(backend, T, n, n)
    P_kernel_tiled!(backend, min(n, 256))(L_for_gessm, ndrange=n)
    
    final_P = KernelAbstractions.zeros(backend, T, num_rows, num_rows)
    P_kernel_tiled!(backend, min(num_rows, 256))(final_P, ndrange=num_rows)

    @assert num_rows % n == 0 "The number of columns in the matrix is not divisible by n!"
    @assert num_cols % n == 0 "The number of rows in the matrix is not divisible by n!"

    tiles_row = num_rows ÷ n
    tiles_col = num_cols ÷ n

    temp = zero(T)

    for k = 1:min(tiles_row, tiles_col)
        # this modifies A and final_P
        dgetrf_tiled!(A, final_P, k, n, M, max_ind, backend)

        # this modifies L_for_gessm
        if n > 1
            L_inv_single_kernel!(backend, min(n-1, 256))(A, L_for_gessm, n, k, ndrange = (n-1))
        end
        
        # the two dgessm steps modify A
        if tiles_row - k > 0
            dgessm_kernel_P!(backend, 256)(A, final_P, k, k, n, temp, ndrange=(n, n, tiles_row-k))
            dgessm_kernel_L_inv!(backend, 256)(A, L_for_gessm, k, n, temp, ndrange=(n, n, tiles_row-k))
        end

        # propogates all changes from factoring tile kk to left and modifies A
        if k - 1 > 0
            dgessm_kernel_P!(backend, 256)(A, final_P, k, 0, n, temp, ndrange=(n, n, k-1))
        end
        
        # modifies A
        if tiles_col > k
            single_dtstrf_lu_gpu!(backend, (n, n))(A, 1, k, tiles_col, (k-1)*n+1, k*n, ndrange = (n, n))
        end

        # modifies A
        if tiles_col - k > 0 && tiles_row - k > 0
            dssssm_kernel!(backend, 256)(A, k, n, temp, ndrange=(n, n, tiles_col-k, tiles_row-k))
        end
    end
    
    return A, final_P
end
