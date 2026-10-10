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

# col_k, offset and temp are recomputed in each block a @synchronize delimits,
# for the reason documented in src/lu/lu_base.jl and src/gemm/matmul.jl: the CPU
# backend splits the body there and plain locals do not survive the split.
@kernel function lu_gpu_tiled!(A, ipiv, M, max_ind, pivot, next, col_start, col_stop, row_start, row_stop)
    I, J = @index(Global, NTuple)

    for k = row_start:row_stop
        
        # find max in col
        if pivot
            if I == 1 && J == 1
                max_val = -one(real(eltype(A)))
                max_i = k - row_start + col_start
                for j = (k - row_start + col_start):col_stop
                    val = abs(A[j, k])
                    if val > max_val
                        max_val = val
                        max_i = j
                    end
                end 
                M[1] = max_val
                max_ind[1] = max_i
                ipiv[k - row_start + 1] = max_i - col_start + 1
            end
            @synchronize

            if I <= row_stop - row_start + 1 && J == 1
                swap_rows_tiled!(A, k - row_start + col_start, max_ind[1], row_start, zero(eltype(A)), I)
            end
            @synchronize
        end

        if I <= col_stop - max(next-1, k - row_start + col_start) && J == 1
            get_l_vals_tiled!(A, k, max(next-1, k - row_start + col_start), row_start, col_start, I, zero(eltype(A)))
        end

        @synchronize

        if I+col_start > max(next-1, k - row_start + col_start) && I+col_start <= col_stop && J+row_start <= row_stop && J+row_start > k
            A[I+col_start, J+row_start] = A[I+col_start, J+row_start] - A[I+col_start, k]*A[k, J+row_start]
        end
        @synchronize
    end
end

# Factors diagonal tile k with pivoting inside the tile; ipiv records the tile's
# row swaps, local to its n rows.
function dgetrf_tiled!(A, ipiv, k, n, M, max_ind, backend)
    lu_gpu_tiled!(backend, (n, n))(A, ipiv, M, max_ind, true, 1, (k-1)*n+1, k*n, (k-1)*n+1, k*n, ndrange = (n, n))
end

@kernel function L_inv_single_kernel!(L, L_inv, n, k_ind)
    I = @index(Global)
    
    for i = 2:n
        if I <= i-1
            L_inv[i, I] = 0
        end
        @synchronize
        for k = 1:i-1 
            if I <= i-1
                L_inv[i, I] -= (L[i+(k_ind-1)*n, k+(k_ind-1)*n]*L_inv[k, I])
            end
            @synchronize
        end
    end
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
        local_A[I, J] = A[I+(i-1)*n, J+row_start-1]
        @synchronize

        for k_inner = row_start:row_stop
            if J == 1
                local_A[I, k_inner-row_start+1] = local_A[I, k_inner-row_start+1]/A[k_inner, k_inner]
            end
            @synchronize

            if J+k_inner <= row_stop
                local_A[I, J+k_inner-row_start+1] = local_A[I, J+k_inner-row_start+1] - local_A[I, k_inner-row_start+1]*A[k_inner, J+k_inner]
            end
            @synchronize
        end

        A[I+(i-1)*n, J+row_start-1] = local_A[I, J]
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

"""
    tile_lu_factor!(A, n) -> (A, p, info)

Tiled LU with partial pivoting within each `n x n` tile, in place, on a GPU: `L` (unit
diagonal) and `U` are packed into `A`, and `p` is the row permutation as a vector,
`A0[p, :] == L * U`. A tile's row swaps reach the rest of its tile row as a gather of those
rows. As LAPACK's getrf, `info` is 0 or the first `k` with a zero `U[k, k]`, and nothing is
thrown. The tile `n` must divide both dimensions and be at most 32.
"""
function tile_lu_factor!(A::AbstractMatrix{T}, n::Int) where T
    backend = KernelAbstractions.get_backend(A)

    # This factorisation is correct on CUDA and wrong on the CPU backend — not
    # an error, wrong numbers: relative residuals of 0.16-0.35 at n=16..64
    # (tile 4, diagonally dominant input), against 1e-16 on CUDA for the same
    # inputs. The kernels rely on @synchronize
    # semantics that KernelAbstractions' CPU emulation does not reproduce here.
    # Refusing is deliberate: silently returning a plausible but wrong
    # factorisation is the worse failure. See KNOWN_ISSUES.md.
    if backend isa KernelAbstractions.CPU
        throw(ArgumentError(
            "tile_lu_factor! is not correct on the CPU backend; use lu_base! " *
            "or LinearAlgebra.lu. See KNOWN_ISSUES.md."))
    end
    # Each tile is factored by one n x n work-group, which the 1024-thread limit
    # caps at n = 32.
    n <= 32 || throw(ArgumentError("tile_lu_factor! runs n x n work-groups, so the tile n must be at most 32, got $n"))
    
    num_rows = size(A, 2) 
    num_cols = size(A, 1)
    
    M = KernelAbstractions.zeros(backend, T, 2)
    max_ind = KernelAbstractions.zeros(backend, Int, 2)
    L_for_gessm = KernelAbstractions.zeros(backend, T, n, n)
    P_kernel_tiled!(backend, min(n, 256))(L_for_gessm, ndrange=n)
    
    ipiv = KernelAbstractions.zeros(backend, Int, n)
    p = collect(1:num_cols)

    num_rows % n == 0 || throw(ArgumentError("tile_lu_factor!: the tile size $n does not divide the $num_rows columns"))
    num_cols % n == 0 || throw(ArgumentError("tile_lu_factor!: the tile size $n does not divide the $num_cols rows"))

    tiles_row = num_rows ÷ n
    tiles_col = num_cols ÷ n

    temp = zero(T)

    for k = 1:min(tiles_row, tiles_col)
        # this modifies A and ipiv
        dgetrf_tiled!(A, ipiv, k, n, M, max_ind, backend)

        # The tile's swaps as a permutation of its rows. They reach the rest of
        # the tile row as a gather: A[rows[perm], cols] is copied before the
        # assignment, so no work-item reads a row another one has overwritten,
        # where a permutation-matrix product read and wrote A in place.
        perm = _ipiv_to_perm(Array(ipiv))
        rows = (k-1)*n .+ (1:n)
        p[rows] = p[rows[perm]]

        # this modifies L_for_gessm
        if n > 1
            L_inv_single_kernel!(backend, min(n-1, 256))(A, L_for_gessm, n, k, ndrange = (n-1))
        end
        
        # the two dgessm steps modify A
        if tiles_row - k > 0
            cols = (k*n + 1):num_rows
            A[rows, cols] .= A[rows[perm], cols]
            dgessm_kernel_L_inv!(backend, 256)(A, L_for_gessm, k, n, temp, ndrange=(n, n, tiles_row-k))
        end

        # propagates the tile's row swaps to the tiles on its left
        if k - 1 > 0
            cols = 1:((k-1)*n)
            A[rows, cols] .= A[rows[perm], cols]
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
    
    return A, p, _lu_info(A)
end
