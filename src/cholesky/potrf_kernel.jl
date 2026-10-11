export cholesky_lower!

# Named MAX_CHOL_THREADS rather than the branch's MAX_THREADS: that name is
# module scope here, too generic to claim, and potrf_left.jl -- a candidate for
# a later port -- defines its own MAX_THREADS with a different meaning.
const MAX_CHOL_THREADS = 512

# Unblocked right-looking Cholesky in a single workgroup, the Cholesky
# counterpart to lu_base!. One thread does the square root and the column
# scaling; the rest share the trailing rank-1 update, with a barrier on either
# side. Suited to a diagonal block, not a whole large matrix -- use
# potrf_recursive! for that, which drives this at its leaves.
#
# Structure is verbatim, including the single @index read at the top. That is
# worth a note: a plain local assigned before a @synchronize does not survive
# the CPU backend's split of the body into separate ndrange loops, which has
# bitten this repo four times. @index is not a plain local and does survive --
# checked by running the unported kernel on the CPU, where it factors to 1.6e-16.
@kernel function chol_kernel_lower!(A, N, ops_per_thread)
    tx = @index(Global, Linear)

    for k in 1:N
        if tx == 1
            @inbounds A[k, k] = sqrt(A[k, k])
            for j in (k + 1):N
                @inbounds A[j, k] /= A[k, k]
            end
        end

        @synchronize

        istart = (k + 1) + (tx - 1) * ops_per_thread
        iend = min(N, istart + ops_per_thread - 1)

        for i in istart:iend
            for j in i:N
                @inbounds A[j, i] -= A[j, k] * A[i, k]
            end
        end

        @synchronize
    end

    istart = (tx - 1) * ops_per_thread + 1
    iend = min(N, istart + ops_per_thread - 1)

    for i in istart:iend
        for j in (i + 1):N
            @inbounds A[i, j] = zero(eltype(A))
        end
    end
end

"""
    cholesky_lower!(A) -> A

Factor the symmetric positive definite matrix `A` in place as `A = L * Lᵀ`,
overwriting the lower triangle with `L` and explicitly zeroing the upper
triangle.

Unblocked and confined to a single workgroup, so it is intended for a diagonal
block rather than a full matrix; [`potrf_recursive!`](@ref) uses it that way.
For a one-shot factorization of a general matrix prefer [`potrf!`](@ref), which
dispatches to the vendor library.

Unlike [`potrf!`](@ref) this zeroes the opposite triangle, and it reports
nothing about definiteness — a non-positive-definite input yields `NaN`s from
the square root rather than an `info` code.
"""
function cholesky_lower!(A)
    N = size(A, 1)
    N == size(A, 2) || throw(DimensionMismatch("cholesky_lower!: A must be square"))
    backend = get_backend(A)

    num_threads = min(N, MAX_CHOL_THREADS)
    ops_per_thread = cld(N, num_threads)

    chol_kernel_lower!(backend, num_threads)(A, N, ops_per_thread; ndrange = num_threads)
    KernelAbstractions.synchronize(backend)
    return A
end
