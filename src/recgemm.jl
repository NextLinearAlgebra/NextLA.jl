export recgemm!
include("wrappers.jl")

function _gemm_dispatch!(alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::AbstractMatrix)
    TA, TB, TC = eltype(A), eltype(B), eltype(C)
    
    if TA == TB == TC && TC in (Float32, Float64)
        gemm!('N', 'N', TC(alpha), A, B, TC(beta), C)
    elseif TA == Float16 && TB == Float16 && TC in (Float16, Float32)
        gemmEx!('N', 'N', alpha, A, B, beta, C)
    else
        A_final = (TA == TC) ? A : TC.(A)
        B_final = (TB == TC) ? B : TC.(B)
        if TC in (Float32, Float64)
            gemm!('N', 'N', TC(alpha), A_final, B_final, TC(beta), C)
        else
            gemmEx!('N', 'N', alpha, A_final, B_final, beta, C)
        end
    end
end

function recgemm!(alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::AbstractMatrix)
    _gemm_dispatch!(alpha, A, B, beta, C)
end

# Pieces of the inner dimension `k` that each C block accumulates over. The m/n splits are fixed by
# C's structure (`size(C.A11)`); the k split is free, so it is chosen with the same power-of-2 rule.
# For square operands this is the original (1:mid, mid+1:n) split.
_k_ranges(k::Integer) = k <= 1 ? (1:k,) : (mk = _split_dim(k); (1:mk, mk+1:k))

# Cblk <- alpha * A[rows, :] * B[:, cols] + beta * Cblk, accumulated over the k pieces
# (first piece applies beta, the rest accumulate with beta = 1).
function _acc!(alpha, A, B, beta, Cblk::FullMixedPrec, rows, cols, kr)
    for (p, r) in enumerate(kr)
        recgemm!(alpha, view(A, rows, r), view(B, r, cols), p == 1 ? beta : 1.0, Cblk; parallel=false)
    end
end

function _acc!(alpha, A, B, beta, Cblk::AbstractMatrix, rows, cols, kr)
    for (p, r) in enumerate(kr)
        _gemm_dispatch!(alpha, view(A, rows, r), view(B, r, cols), p == 1 ? beta : 1.0, Cblk)
    end
end

# C (m x n) <- alpha * A (m x k) * B (k x n) + beta * C. `A`, `B` and `C` may be rectangular
# (e.g. tall-and-skinny): C's own block structure fixes the m and n splits, k is split independently.
function recgemm!(alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::FullMixedPrec; parallel::Bool=(size(C, 1) > 512))
    if C.BaseCase !== nothing
        _gemm_dispatch!(alpha, A, B, beta, C.BaseCase)
        return
    end

    nr, nc = size(C)
    @assert size(A, 1) == nr && size(B, 2) == nc && size(A, 2) == size(B, 1) "recgemm!: incompatible sizes A=$(size(A)), B=$(size(B)), C=$(size(C))"
    mr, mc = size(C.A11)      # row / column split of C (equal for square C)
    kr = _k_ranges(size(A, 2))

    top, bot = 1:mr, mr+1:nr
    left, right = 1:mc, mc+1:nc

    if parallel
        @sync begin
            @async _acc!(alpha, A, B, beta, C.A11, top, left,  kr)
            @async _acc!(alpha, A, B, beta, C.A12, top, right, kr)
            @async _acc!(alpha, A, B, beta, C.A21, bot, left,  kr)
            @async _acc!(alpha, A, B, beta, C.A22, bot, right, kr)
        end
    else
        # Sequential updates
        _acc!(alpha, A, B, beta, C.A11, top, left,  kr)
        _acc!(alpha, A, B, beta, C.A12, top, right, kr)
        _acc!(alpha, A, B, beta, C.A21, bot, left,  kr)
        _acc!(alpha, A, B, beta, C.A22, bot, right, kr)
    end
end
