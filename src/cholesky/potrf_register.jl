# Register-blocked Cholesky of one diagonal block, the left-looking sibling of
# chol_kernel_shared! in potrf_blocked.jl.
#
# Source: vicki-development, src/potrf_left.jl -- the live sixth of it. The rest
# of that file is three superseded versions of this kernel commented out (a
# double-buffered variant and two earlier thread mappings) and
# cholesky_lower_left_profiled!, a timing harness returning CUDA.@elapsed
# figures. Its driver is not taken either: it is cholesky_blocked! with
# CUBLAS.gemm!/CUBLAS.trsm! hardcoded, so this kernel is wired into that driver
# instead and one driver serves both kernels.
#
# Where chol_kernel_shared! stages the whole block in shared memory, this one
# keeps each thread's four columns in registers and shares only the active
# column and the next diagonal entry. That is the point of it: shared memory
# holds N + 1 elements rather than 64 * 65, and the square root for column k+1
# is computed during the elimination for column k rather than after it.
#
# The thread mapping reads as warps but needs no warp: 768 threads are cut into
# 24 chunks of 32, and the barriers are @synchronize. Chunks 0-15 take the first
# eight strips two at a time, one chunk per 32-row half of the block; chunks
# 16-23 take the last eight strips, whose columns only reach rows 33-64 in a
# lower triangle, so one chunk covers each. Nothing here reads another lane's
# register, so a 64-wide wavefront changes the work split and not the answer.
#
# cpu=false is the author's and is kept, for the reason chol_kernel_shared!
# carries it: @private registers survive the CPU backend's ndrange split but the
# hand-rolled row and strip indices do not.

# 24 chunks of 32 rows: two chunks per strip over a 64-row block. The launch has
# to put all of them in one workgroup, so this is a workgroup size as well as an
# ndrange.
const CHOL_REG_THREADS = 768
# Columns per thread, and so the depth of the @private register file.
const CHOL_REG_STRIP = 4
# Rows one chunk covers; CHOL_BLOCK is two of these.
const CHOL_REG_ROWS = 32
# Strips whose rows are split across two chunks: the first half of the strips,
# eight of the sixteen a 64-wide block holds.
const CHOL_REG_SPLIT = CHOL_BLOCK ÷ (2 * CHOL_REG_STRIP)

@kernel cpu=false inbounds=true unsafe_indices=false function chol_kernel_register!(
        A, ::Val{N}) where N
    tx = @index(Global, Linear)

    chunk = (tx - 1) ÷ CHOL_REG_ROWS
    lane = (tx - 1) % CHOL_REG_ROWS

    # Which strip of CHOL_REG_STRIP columns this thread owns, and which row of
    # it. The first half of the chunks splits a strip's rows in two; the second
    # half needs only the bottom rows, the top of those columns being above the
    # diagonal.
    if chunk < 2 * CHOL_REG_SPLIT
        strip_idx = chunk ÷ 2
        is_bottom = (chunk % 2) == 1
        my_row = lane + 1 + (is_bottom ? CHOL_REG_ROWS : 0)
    else
        strip_idx = chunk - CHOL_REG_SPLIT
        my_row = lane + 1 + CHOL_REG_ROWS
    end

    col_start = strip_idx * CHOL_REG_STRIP + 1

    # Global -> registers. A column past the block's width reads as zero so the
    # tail block of a matrix that is not a multiple of CHOL_BLOCK still works.
    my_vals = @private eltype(A) (CHOL_REG_STRIP,)
    if my_row <= N
        @unroll for i in 1:CHOL_REG_STRIP
            c = col_start + (i - 1)
            @inbounds my_vals[i] = c <= N ? A[my_row, c] : zero(eltype(A))
        end
    end

    # The whole of shared memory: the active column, and the diagonal entry the
    # next iteration divides by.
    tile = @localmem eltype(A) N
    diag_val_next = @localmem eltype(A) 1

    @synchronize

    # Prime the pipeline with the first diagonal; every later square root is
    # taken inside the loop, one iteration ahead of its use.
    if tx == 1
        @inbounds my_vals[1] = sqrt(my_vals[1])
        @inbounds diag_val_next[1] = my_vals[1]
    end

    @synchronize

    @unroll 4 for k in 1:N
        local_idx = (k - col_start) + 1
        diag_val = @inbounds diag_val_next[1]

        # The owners of column k scale it by the diagonal and publish it.
        if k >= col_start && k < (col_start + CHOL_REG_STRIP)
            curr_val = @inbounds my_vals[local_idx]
            if my_row == k
                @inbounds tile[k] = curr_val
            elseif my_row > k && my_row <= N
                curr_val /= diag_val
                @inbounds my_vals[local_idx] = curr_val
                @inbounds tile[my_row] = curr_val
            end
        end

        @synchronize

        # Column k - CHOL_REG_STRIP took its last update at iteration k - 1, so
        # it is final now and can go back to global memory while the rest of the
        # block is still being worked on.
        c_write = k - CHOL_REG_STRIP
        if c_write >= col_start && c_write < col_start + CHOL_REG_STRIP &&
           my_row <= N && my_row >= c_write
            @inbounds A[my_row, c_write] = my_vals[c_write - col_start + 1]
        end

        # A[r, c] -= L[r, k] * L[c, k], entirely in registers.
        if my_row > k && my_row <= N && k < (col_start + CHOL_REG_STRIP)
            L_rk = @inbounds tile[my_row]
            @unroll for i in 1:CHOL_REG_STRIP
                c = col_start + (i - 1)
                if c > k && my_row >= c
                    L_ck = @inbounds tile[c]
                    @inbounds my_vals[i] -= L_ck * L_rk
                end
            end
        end

        # The square root for the next column, taken now so the division at the
        # top of the next iteration does not wait on it.
        if k < N
            k_next = k + 1
            if k_next >= col_start && k_next < (col_start + CHOL_REG_STRIP) &&
               my_row == k_next
                local_idx_next = (k_next - col_start) + 1
                @inbounds my_vals[local_idx_next] = sqrt(my_vals[local_idx_next])
                @inbounds diag_val_next[1] = my_vals[local_idx_next]
            end
        end

        @synchronize
    end

    # The last CHOL_REG_STRIP columns never reach the in-loop write above, the
    # loop having run out of iterations to trail behind.
    if my_row <= N && col_start + CHOL_REG_STRIP - 1 > N - CHOL_REG_STRIP
        @unroll for i in 1:CHOL_REG_STRIP
            c = col_start + (i - 1)
            if c <= N && my_row >= c && c > N - CHOL_REG_STRIP
                @inbounds A[my_row, c] = my_vals[i]
            end
        end
    end
end
