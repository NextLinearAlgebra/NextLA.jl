# Triangular solve (func = 'S') and triangular multiply (func = 'M') with a
# mixed-precision container A, without densifying A. Source: vicki-development
# @ 694d019, by Vicki <vickicar@mit.edu>, where it sat in src/trsm/rectrxm.jl.
#
# How the recursion works
# -----------------------
# A container is a 2x2 block tree. A node is either a leaf (one dense block,
# `Base`) or two diagonal children A11, A22 and one off-diagonal block, stored
# narrow with its own scale. B is cut where A is cut: into top/bottom halves B1,
# B2 for side = 'L' (op(A) * X = B), left/right halves for side = 'R'
# (X * op(A) = B). For a lower-triangular solve from the left:
#
#     [A11  0 ] [X1]   [B1]       X1 = A11 \ B1          (recurse)
#     [A21 A22] [X2] = [B2]   =>  B2 = B2 - A21 * X1     (off-diagonal update)
#                                 X2 = A22 \ B2          (recurse)
#
# Every other case has the same three steps: recurse into one half, update the
# other half with the off-diagonal block, recurse into the other half. Only two
# things change between cases, and _rec_mixed! below decides both:
#   - which half goes first, and
#   - which half the off-diagonal update reads and which it writes.
#
# Precision
# ---------
# An off-diagonal block is stored narrow because its precision matters least,
# so its update runs at that narrow type (_offdiag_update!). A leaf is a
# diagonal block, whose precision matters, so it runs at the wider of its own
# type and B's (the leaf branch of _rec_mixed!). B, which accumulates the
# result, is never rounded.

const _RecMixedPrec = Union{FullMixedPrec, SymmMixedPrec, TriMixedPrec}
const _RecMixedPrecOrTransposed = Union{_RecMixedPrec, TransposedMixedPrec{<:Any, <:_RecMixedPrec}}

# ---------------------------------------------------------------------------
# Reading one node of the tree. A transposed container is read through its
# parent: its children are the parent's children transposed, and its lower
# off-diagonal block is the parent's upper one, transposed.
# ---------------------------------------------------------------------------

# The off-diagonal block on the `uplo` side of a node, and its scale.
_offdiag(A::FullMixedPrec, uplo::Char) = uplo == 'L' ? (A.A21, A.A21_scale) : (A.A12, A.A12_scale)
_offdiag(A::Union{SymmMixedPrec, TriMixedPrec}, ::Char) = (A.OffDiag, A.OffDiag_scale)

_is_leaf(A::_RecMixedPrec) = A.Base !== nothing
_is_leaf(A::TransposedMixedPrec) = _is_leaf(parent(A))

# A leaf's dense block and its scale.
_leaf_block(A::_RecMixedPrec) = (A.Base, A.Base_scale)
function _leaf_block(A::TransposedMixedPrec)
    block, scale = _leaf_block(parent(A))
    return (transpose(block), scale)
end

# A split node's diagonal children, its off-diagonal block on the `uplo` side
# with that block's scale, and the size of the first child (where B is cut).
function _children(A::_RecMixedPrec, uplo::Char)
    offdiag, scale = _offdiag(A, uplo)
    return (A.A11, A.A22, offdiag, scale, size(A.A11, 1))
end
function _children(A::TransposedMixedPrec, uplo::Char)
    P = parent(A)
    offdiag, scale = _offdiag(P, uplo == 'L' ? 'U' : 'L')
    return (TransposedMixedPrec(P.A11), TransposedMixedPrec(P.A22), transpose(offdiag), scale, size(P.A11, 1))
end

# ---------------------------------------------------------------------------
# The off-diagonal update
# ---------------------------------------------------------------------------

# C = alpha * block * X + beta * C on the left, or alpha * X * block + beta * C
# on the right: the off-diagonal block multiplies X from the side A is on.
function _side_gemm!(side::Char, alpha, block, X, beta, C)
    if side == 'L'
        return _gemm_dispatch!(alpha, block, X, beta, C)
    else
        return _gemm_dispatch!(alpha, X, block, beta, C)
    end
end

# The off-diagonal update: target = target - scale * block * source for a solve,
# target = target + scale * block * source for a multiply (source * block on the
# right). `scale` is the block's scale, which turns its stored values back into
# real ones.
function _offdiag_update!(func::Char, side::Char, target::AbstractMatrix, block::AbstractMatrix,
                          source::AbstractMatrix, scale)
    sign = func == 'S' ? -1 : 1
    T_block, T_target = eltype(block), eltype(target)

    # The vendor GEMM takes a transposed operand only in its own type, so a
    # transposed block that is about to be mixed with another type is copied --
    # at its storage type, which is small.
    if T_block != T_target && block isa Union{Transpose, Adjoint}
        block = copy(block)
    end

    # Case 1: the block is at least as wide as target. Nothing is narrowed; the
    # product goes straight into target.
    if T_block == T_target || promote_type(T_block, T_target) == T_block
        return _side_gemm!(side, sign * scale, block, source, one(T_target), target)
    end

    # Case 2: the block is narrower than target, the usual case. The product
    # runs at the block's type: source is rounded to it the way the containers
    # store a block, with a scale of its own when its values pass floatmax, which
    # joins the block's in alpha. That adds an error of the order the block's
    # storage already carries. A non-finite source, left by an overflowed multiply
    # before it, converts as it is. The sum accumulates in at least Float32 --
    # gemmEx does this for Float16 -- and is added into target, which is never
    # rounded.
    source_narrow, source_scale =
        all(isfinite, source) ? quantize(source, T_block) : (T_block.(source), 1f0)
    T_accumulate = sizeof(T_block) < 4 ? Float32 : T_block
    alpha = T_accumulate(sign * Float64(scale) * Float64(source_scale))
    if T_accumulate == T_target
        _side_gemm!(side, alpha, block, source_narrow, one(T_target), target)
    else
        partial = similar(target, T_accumulate, size(target))
        _side_gemm!(side, alpha, block, source_narrow, zero(T_accumulate), partial)
        target .+= partial
    end
    return target
end

# ---------------------------------------------------------------------------
# The recursion
# ---------------------------------------------------------------------------

function _rec_mixed!(func::Char, side::Char, uplo::Char, diag::Char, A::_RecMixedPrecOrTransposed,
                     B::AbstractMatrix, threshold::Integer)
    if _is_leaf(A)
        # The leaf: one dense diagonal block, solved or multiplied by the dense
        # kernels. Its precision matters, so the work runs in the wider of the
        # block's type and B's and neither is rounded; the kernels take one element
        # type, so whichever of the two is narrower is widened for the call. The
        # block holds its values divided by its scale, so a solve then divides by
        # the scale and a multiply multiplies by it, before the result goes back
        # into B, so a narrow B only sees the final values.
        block, scale = _leaf_block(A)
        T_block, T_B = eltype(block), eltype(B)
        T_work = promote_type(T_block, T_B)
        block_work = T_block == T_work ? block : T_work.(block)
        B_work = T_B == T_work ? B : T_work.(B)
        unified_rec(func, side, uplo, block_work, size(block_work, 1), B_work, threshold;
                    unitdiag=(diag == 'U'))
        if !isone(scale)
            func == 'S' ? (B_work ./= scale) : (B_work .*= scale)
        end
        B_work === B || copy!(B, B_work)
        return B
    end

    A11, A22, offdiag, scale, mid = _children(A, uplo)

    # Cut B where A is cut: rows for side 'L', columns for side 'R'.
    if side == 'L'
        n = size(B, 1)
        B1, B2 = view(B, 1:mid, :), view(B, mid+1:n, :)
    else
        n = size(B, 2)
        B1, B2 = view(B, :, 1:mid), view(B, :, mid+1:n)
    end

    # A solve must finish the half that does not depend on the other one first; a
    # multiply must finish the half that the other one is added into first, before
    # that half is overwritten. Worked out case by case:
    #
    #   func  side  uplo   first half
    #   'S'   'L'   'L'    top     X1 = A11 \ B1 needs nothing else
    #   'S'   'L'   'U'    bottom  X2 = A22 \ B2 needs nothing else
    #   'S'   'R'   'L'    right   X2 = B2 / A22 needs nothing else
    #   'S'   'R'   'U'    left    X1 = B1 / A11 needs nothing else
    #   'M'   any   any    the opposite half to the solve with the same side and uplo
    #
    # "Top" and "left" are both the A11 half, so the rule is: the A11 half goes
    # first exactly when (it is a solve) == (side and uplo agree).
    if (func == 'S') == ((side == 'L') == (uplo == 'L'))
        A_first, B_first, A_second, B_second = A11, B1, A22, B2
    else
        A_first, B_first, A_second, B_second = A22, B2, A11, B1
    end

    # Step 1: the first half.
    _rec_mixed!(func, side, uplo, diag, A_first, B_first, threshold)

    # Step 2: the off-diagonal update. A solve takes the half it has just solved
    # out of the half still to come. A multiply adds the half not yet multiplied
    # (still the original values) into the half already done.
    if func == 'S'
        _offdiag_update!(func, side, B_second, offdiag, B_first, scale)
    else
        _offdiag_update!(func, side, B_first, offdiag, B_second, scale)
    end

    # Step 3: the second half.
    _rec_mixed!(func, side, uplo, diag, A_second, B_second, threshold)
    return B
end

"""
    unified_rectrxm!(side, uplo, trans, diag, alpha, func, A, B) -> B
    unified_rectrxm!(side, uplo, trans, alpha, func, A, B) -> B

`unified_rectrxm!` for a mixed-precision container `A`: solves `op(A) X = alpha B` for `func =
'S'`, or computes `B = alpha op(A) B` for `func = 'M'`, on side `side` (`B * op(A)` for
`'R'`), in place in `B`. `op` is the identity for `trans = 'N'` and the transpose for `'T'` or
`'C'`; `uplo` names the triangle of `A` that is read, and must be `A.uplo` for a
`SymmMixedPrec` or `TriMixedPrec`, which store only that triangle. `diag = 'U'` takes the
diagonal as ones, and the form without `diag` takes `diag = 'N'`. `A` is a `FullMixedPrec`,
`SymmMixedPrec` or `TriMixedPrec`; for its transpose pass `trans = 'T'`, not `transpose(A)`.
Any other `side`, `uplo`, `trans`, `diag` or `func` throws an `ArgumentError`.
"""
function unified_rectrxm!(
        side::Char, uplo::Char, trans::Char, diag::Char, alpha::Number, func::Char,
        A::_RecMixedPrec, B::AbstractMatrix
    )
    side in ('L', 'R') || throw(ArgumentError("side must be 'L' or 'R', got '$side'"))
    uplo in ('L', 'U') || throw(ArgumentError("uplo must be 'L' or 'U', got '$uplo'"))
    trans in ('N', 'T', 'C') || throw(ArgumentError("trans must be 'N', 'T' or 'C', got '$trans'"))
    diag in ('N', 'U') || throw(ArgumentError("diag must be 'N' or 'U', got '$diag'"))
    func in ('S', 'M') || throw(ArgumentError("func must be 'S' or 'M', got '$func'"))
    A isa FullMixedPrec || uplo == A.uplo ||
        throw(ArgumentError("uplo must be A.uplo, '$(A.uplo)', for a $(nameof(typeof(A))), got '$uplo'"))
    op_A, op_uplo = trans == 'N' ? (A, uplo) : (TransposedMixedPrec(A), uplo == 'L' ? 'U' : 'L')
    func == 'S' && (B .= alpha .* B)
    _rec_mixed!(func, side, op_uplo, diag, op_A, B, 256)
    func == 'M' && (B .= alpha .* B)
    return B
end

function unified_rectrxm!(
        side::Char, uplo::Char, trans::Char, alpha::Number, func::Char,
        A::_RecMixedPrec, B::AbstractMatrix
    )
    return unified_rectrxm!(side, uplo, trans, 'N', alpha, func, A, B)
end
