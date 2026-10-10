export cholesky_blocked!

# Right-looking blocked Cholesky: the panel form of the factorisation, driving a
# shared-memory kernel on each diagonal block.
#
# Source: vicki-development @ 694d019, src/potrf.jl -- the live third of it; the
# other two thirds are an earlier version of the same kernel, commented out.
# src/potrf_left.jl carries a near-identical driver differing only in which
# kernel it calls, so one driver is merged rather than two.
#
# This is the blocked counterpart to potrf_recursive!: same factorisation, a
# panel sweep instead of a subdivision. For a single diagonal block, cholesky_lower!
# in potrf_kernel.jl is the portable choice; this is for a matrix large enough
# that the trailing update dominates.

# The branch tiles a 64-wide block into shared memory with one column of padding,
# so that column c lands at (c-1)*CHOL_STRIDE and consecutive rows fall in
# different banks. Named apart from the branch's BLOCK_SIZE/PAD/STRIDE, which are
# too generic to hold at module scope.
const CHOL_BLOCK = 64
const CHOL_STRIDE = CHOL_BLOCK + 1

# Shared-memory Cholesky of one diagonal block, verbatim in structure.
#
# `cpu=false` is the author's and is kept: the body indexes a flat @localmem tile
# with a hand-computed stride and strides the work across a fixed thread count,
# neither of which survives the CPU backend's ndrange split. cholesky_blocked!
# refuses on the CPU rather than let that fail obscurely.
@kernel cpu=false inbounds=true unsafe_indices=false function chol_kernel_shared!(
        A, ::Val{N}, ::Val{NT}) where {N, NT}
    tx = @index(Global, Linear)

    tile = @localmem eltype(A) (CHOL_BLOCK * CHOL_STRIDE)

    total_elements = N * N
    idx = tx
    while idx <= total_elements
        c = div(idx - 1, N) + 1
        r = rem(idx - 1, N) + 1
        @inbounds tile[(c - 1) * CHOL_STRIDE + r] = A[r, c]
        idx += NT
    end

    @synchronize

    for k in 1:N
        diag_idx = (k - 1) * CHOL_STRIDE + k
        if tx == 1
            @inbounds tile[diag_idx] = sqrt(tile[diag_idx])
        end

        @synchronize

        # Read after the barrier, so every thread sees the root the first wrote.
        dval = @inbounds tile[diag_idx]
        idx = k + tx
        while idx <= N
            @inbounds tile[(k - 1) * CHOL_STRIDE + idx] /= dval
            idx += NT
        end

        @synchronize

        len = Int32(N - k)
        if len > 0
            limit = len * len
            t_idx = Int32(tx) - Int32(1)
            stride = Int32(NT)
            col_offset = div(t_idx, len)
            row_offset = rem(t_idx, len)
            stride_c = div(stride, len)
            stride_r = rem(stride, len)

            while t_idx < limit
                c = col_offset + Int32(k + 1)
                r = row_offset + Int32(k + 1)
                @inbounds tile[(c - 1) * CHOL_STRIDE + r] -=
                    tile[(k - 1) * CHOL_STRIDE + r] * tile[(k - 1) * CHOL_STRIDE + c]
                t_idx += stride
                col_offset += stride_c
                row_offset += stride_r
                if row_offset >= len
                    row_offset -= len
                    col_offset += Int32(1)
                end
            end
        end

        @synchronize
    end

    idx = tx
    while idx <= total_elements
        c = div(idx - 1, N) + 1
        r = rem(idx - 1, N) + 1
        if r >= c
            @inbounds A[r, c] = tile[(c - 1) * CHOL_STRIDE + r]
        end
        idx += NT
    end
end

"""
    cholesky_blocked!(A; block = 64, threads = 512, kernel = :shared) -> A

In-place right-looking blocked Cholesky, `A = L * Lᵀ`, writing `L` into the lower
triangle of `A` and leaving the strict upper triangle untouched.

Each pass factors one diagonal block with a kernel, updates the panel below it
with a triangular solve, and applies the trailing rank-`block` update. `block`
must not exceed 64, the width both kernels are sized for.

`kernel` picks how a diagonal block is factored, and only that -- the panel
solve and the trailing update are the same either way:

  - `:shared` stages the block in shared memory (`chol_kernel_shared!`), and
    takes `threads` as given.
  - `:register` keeps each thread's four columns in registers and shares only
    the active column (`chol_kernel_register!`). Its thread mapping is fixed, so
    it ignores `threads` and launches `CHOL_REG_THREADS` of them in one
    workgroup; the backend has to allow a workgroup that large.

GPU only: the kernel is declared `cpu=false` on the branch this came from, and
the CPU backend cannot run it. Use [`potrf!`](@ref) or [`potrf_recursive!`](@ref)
there.

!!! warning
    No positive-definiteness check. The kernel takes `sqrt` of each diagonal
    entry as it goes, so an indefinite matrix yields `NaN` rather than an error
    or an `info` code — unlike `potrf!`, which reports through `info`.
"""
function cholesky_blocked!(A::AbstractMatrix; block::Integer = CHOL_BLOCK,
                           threads::Integer = 512, kernel::Symbol = :shared)
    backend = KernelAbstractions.get_backend(A)
    backend isa KernelAbstractions.CPU && throw(ArgumentError(
        "cholesky_blocked! is GPU-only; its kernel is declared cpu=false. " *
        "Use potrf! or potrf_recursive! on the CPU backend."))
    size(A, 1) == size(A, 2) ||
        throw(DimensionMismatch("cholesky_blocked!: A must be square"))
    1 <= block <= CHOL_BLOCK || throw(ArgumentError(
        "block must be between 1 and $CHOL_BLOCK; the shared tile is sized for that"))
    kernel in (:shared, :register) || throw(ArgumentError(
        "kernel must be :shared or :register, got :$kernel"))

    N = size(A, 1)
    T = eltype(A)
    for k in 1:block:N
        k_end = min(k + block - 1, N)
        blk = k_end - k + 1

        if k > 1
            # A_panel -= L_prev_cols * L_prev_topᵀ. The branch called
            # CUBLAS.gemm!('N','T',...); the output-first gemm! here takes the
            # transpose on the argument instead, which reaches the same call.
            L_prev_cols = view(A, k:N, 1:(k - 1))
            L_prev_top = view(A, k:k_end, 1:(k - 1))
            gemm!(view(A, k:N, k:k_end), L_prev_cols, transpose(L_prev_top),
                  -one(T), one(T))
        end

        A_diag = view(A, k:k_end, k:k_end)
        if kernel === :shared
            chol_kernel_shared!(backend, threads)(
                A_diag, Val(blk), Val(Int(threads)); ndrange = threads)
        else
            # One workgroup, and the whole of it: the mapping cuts exactly
            # CHOL_REG_THREADS threads into chunks and every barrier in the
            # kernel is a workgroup barrier.
            chol_kernel_register!(backend, CHOL_REG_THREADS)(
                A_diag, Val(blk); ndrange = CHOL_REG_THREADS)
        end

        if k_end < N
            # L21 = A21 * L11⁻ᵀ
            unified_rectrxm!('R', 'L', 'T', 'N', one(T), 'S',
                             A_diag, view(A, (k_end + 1):N, k:k_end))
        end
    end
    KernelAbstractions.synchronize(backend)
    return A
end
