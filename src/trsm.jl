export LeftLowerTRSM!, LeftUpperTRSM!, RightLowerTRSM!, RightUpperTRSM!, trsm_batched!

@inline _trsm_default_nb(::Type) = Val(32)
@inline _trsm_batch_size(A::AbstractMatrix) = 1
@inline _trsm_batch_size(A::AbstractArray{<:Any,3}) = size(A, 3)
@inline _trsm_batch_view(A::AbstractMatrix, b::Int) = A
@inline _trsm_batch_view(A::AbstractArray{<:Any,3}, b::Int) = view(A, :, :, b)

@inline _trsm_diag_block(A::AbstractMatrix, cols) = view(A, :, cols)
@inline _trsm_diag_block(A::AbstractArray{<:Any,3}, cols) = view(A, :, cols, :)

@inline _trsm_A_block(A::AbstractMatrix, rows, cols) = view(A, rows, cols)
@inline _trsm_A_block(A::AbstractArray{<:Any,3}, rows, cols) = view(A, rows, cols, :)

@inline _trsm_row_block(B::AbstractMatrix, rows) = view(B, rows, :)
@inline _trsm_row_block(B::AbstractArray{<:Any,3}, rows) = view(B, rows, :, :)

@inline _trsm_col_block(B::AbstractMatrix, cols) = view(B, :, cols)
@inline _trsm_col_block(B::AbstractArray{<:Any,3}, cols) = view(B, :, cols, :)

@inline _trsm_getindex(A::AbstractMatrix, i::Int, j::Int, b::Int) = @inbounds A[i, j]
@inline _trsm_getindex(A::AbstractArray{<:Any,3}, i::Int, j::Int, b::Int) = @inbounds A[i, j, b]

@inline _trsm_setindex!(A::AbstractMatrix, value, i::Int, j::Int, b::Int) = (@inbounds A[i, j] = value; nothing)
@inline _trsm_setindex!(A::AbstractArray{<:Any,3}, value, i::Int, j::Int, b::Int) = (@inbounds A[i, j, b] = value; nothing)

@inline function _trsm_matmul!(C::AbstractMatrix, A, B, alpha, beta)
    LinearAlgebra.mul!(C, A, B, alpha, beta) # dispatch to vendor `gemm`
    return C
end

@inline function _trsm_matmul!(C::AbstractMatrix, A::AbstractArray{<:Any,3}, B, alpha, beta)
    size(A, 3) == 1 || throw(DimensionMismatch("trsm: expected a singleton batch"))
    LinearAlgebra.mul!(C, view(A, :, :, 1), B, alpha, beta)
    return C
end

@inline function _trsm_matmul!(C::AbstractMatrix, A, B::AbstractArray{<:Any,3}, alpha, beta)
    size(B, 3) == 1 || throw(DimensionMismatch("trsm: expected a singleton batch"))
    LinearAlgebra.mul!(C, A, view(B, :, :, 1), alpha, beta)
    return C
end

@inline function _trsm_matmul!(C::AbstractMatrix, A::AbstractArray{<:Any,3}, B::AbstractArray{<:Any,3}, alpha, beta)
    size(A, 3) == 1 || throw(DimensionMismatch("trsm: expected a singleton left batch"))
    size(B, 3) == 1 || throw(DimensionMismatch("trsm: expected a singleton right batch"))
    LinearAlgebra.mul!(C, view(A, :, :, 1), view(B, :, :, 1), alpha, beta)
    return C
end

@inline function _trsm_matmul!(C::AbstractArray{<:Any,3}, A, B, alpha, beta)
    gemm_batched!('N', 'N', alpha, A, B, beta, C)
    return C
end

function _trsm_validate(side::Val{SIDE}, A, B) where {SIDE}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("trsm: A must be square"))
    _trsm_batch_size(A) == _trsm_batch_size(B) || throw(DimensionMismatch("trsm: batch sizes must match"))
    size(B, SIDE === :left ? 1 : 2) == n || throw(DimensionMismatch("trsm: incompatible A and B"))
    return n, _trsm_batch_size(B)
end

@inline function _trsm_update!(A,
    B,
    side::Val{SIDE},
    uplo::Val{UPLO},
    k0::Int,
    k1::Int,
    n::Int) where {SIDE,UPLO}
    alpha = -one(eltype(B))
    beta = one(eltype(B))
    if SIDE === :left && UPLO === :lower
        k1 == n && return B
        _trsm_matmul!(_trsm_row_block(B, k1+1:n),
            _trsm_A_block(A, k1+1:n, k0:k1),
            _trsm_row_block(B, k0:k1),
            alpha, beta)
    elseif SIDE === :left
        k0 == 1 && return B
        _trsm_matmul!(_trsm_row_block(B, 1:k0-1),
            _trsm_A_block(A, 1:k0-1, k0:k1),
            _trsm_row_block(B, k0:k1),
            alpha, beta)
    elseif UPLO === :lower
        k0 == 1 && return B
        _trsm_matmul!(_trsm_col_block(B, 1:k0-1),
            _trsm_col_block(B, k0:k1),
            _trsm_A_block(A, k0:k1, 1:k0-1),
            alpha, beta)
    else
        k1 == n && return B
        _trsm_matmul!(_trsm_col_block(B, k1+1:n),
            _trsm_col_block(B, k0:k1),
            _trsm_A_block(A, k0:k1, k1+1:n),
            alpha, beta)
    end
    return B
end

@inline _trsm_side_char(::Val{:left}) = 'L'
@inline _trsm_side_char(::Val{:right}) = 'R'
@inline _trsm_uplo_char(::Val{:lower}) = 'L'
@inline _trsm_uplo_char(::Val{:upper}) = 'U'
@inline _trsm_last_block_start(n::Int, nb::Int) = ((n - 1) ÷ nb) * nb + 1

function _trsm_cpu_fallback!(A, B, side, uplo)
    # blas fallback on host batches.
    for b in 1:_trsm_batch_size(A)
        A_blas = _trsm_batch_view(A, b)
        A_blas = A_blas isa Union{Transpose,Adjoint} ? copy(A_blas) : A_blas
        BLAS.trsm!(_trsm_side_char(side), _trsm_uplo_char(uplo), 'N', 'N',
            one(eltype(A)), A_blas, _trsm_batch_view(B, b))
    end
    return B
end

@kernel unsafe_indices = true function _trsm_invert_kernel_2d!(A::AbstractMatrix,
    Ainv::AbstractMatrix,
    k0::Int,
    block_n::Int,
    ::Val{UPLO},
    ::Val{NB}) where {UPLO,NB}
    row = @index(Local, Linear)
    tile = @localmem eltype(Ainv) (NB + 1, NB)

    # stage one diagonal block with tail padding.
    @unroll for col in 1:NB
        if row <= block_n && col <= block_n
            value = UPLO === :lower ? (col <= row ? A[k0 + row - 1, k0 + col - 1] : zero(eltype(A))) :
                    (row <= col ? A[k0 + row - 1, k0 + col - 1] : zero(eltype(A)))
            @inbounds tile[row, col] = value
        else
            @inbounds tile[row, col] = row == col ? one(eltype(A)) : zero(eltype(A))
        end
    end
    @synchronize

    # shared-memory triangular inversion.
    @unroll for k in 1:NB
        if row == k
            @inbounds inv_pivot = inv(tile[k, k])
            @unroll for col in 1:NB
                @inbounds tile[k, col] = (col == k ? one(eltype(A)) : tile[k, col]) * inv_pivot
            end
        end
        @synchronize

        if row != k
            @inbounds aik = tile[row, k]
            @unroll for col in 1:NB
                if col == k
                    @inbounds tile[row, col] = -aik * tile[k, col]
                else
                    @inbounds tile[row, col] -= aik * tile[k, col]
                end
            end
        end
        @synchronize
    end

    @unroll for col in 1:NB
        @inbounds Ainv[row, col] = tile[row, col]
    end
end

@kernel unsafe_indices = true function _trsm_invert_kernel_3d!(A::AbstractArray{<:Any,3},
    Ainv::AbstractArray{<:Any,3},
    k0::Int,
    block_n::Int,
    ::Val{UPLO},
    ::Val{NB}) where {UPLO,NB}
    row = @index(Local, Linear)
    group = @index(Group, NTuple)
    batch = group[end]
    tile = @localmem eltype(Ainv) (NB + 1, NB)

    # stage one diagonal block per batch entry.
    @unroll for col in 1:NB
        if row <= block_n && col <= block_n
            value = UPLO === :lower ? (col <= row ? A[k0 + row - 1, k0 + col - 1, batch] : zero(eltype(A))) :
                    (row <= col ? A[k0 + row - 1, k0 + col - 1, batch] : zero(eltype(A)))
            @inbounds tile[row, col] = value
        else
            @inbounds tile[row, col] = row == col ? one(eltype(A)) : zero(eltype(A))
        end
    end
    @synchronize

    # shared-memory triangular inversion.
    @unroll for k in 1:NB
        if row == k
            @inbounds inv_pivot = inv(tile[k, k])
            @unroll for col in 1:NB
                @inbounds tile[k, col] = (col == k ? one(eltype(A)) : tile[k, col]) * inv_pivot
            end
        end
        @synchronize

        if row != k
            @inbounds aik = tile[row, k]
            @unroll for col in 1:NB
                if col == k
                    @inbounds tile[row, col] = -aik * tile[k, col]
                else
                    @inbounds tile[row, col] -= aik * tile[k, col]
                end
            end
        end
        @synchronize
    end

    @unroll for col in 1:NB
        @inbounds Ainv[row, col, batch] = tile[row, col]
    end
end

function _trsm_device_inverse_block!(Ainv::AbstractMatrix, A::AbstractMatrix, uplo,
    k0::Int,
    block_n::Int,
    ::Val{NB},
    batch::Int) where {NB}
    NB in (32, 64) || throw(ArgumentError("device TRSM inversion supports only NB=32 or NB=64, got NB=$NB"))
    _trsm_invert_kernel_2d!(get_backend(A), (NB, 1, 1))(A, Ainv, k0, block_n, uplo, Val(NB); ndrange=(NB, 1, 1))
    return Ainv
end

function _trsm_device_inverse_block!(Ainv::AbstractArray{<:Any,3}, A::AbstractArray{<:Any,3}, uplo,
    k0::Int,
    block_n::Int,
    ::Val{NB},
    batch::Int) where {NB}
    NB in (32, 64) || throw(ArgumentError("device TRSM inversion supports only NB=32 or NB=64, got NB=$NB"))
    _trsm_invert_kernel_3d!(get_backend(A), (NB, 1, 1))(A, Ainv, k0, block_n, uplo, Val(NB); ndrange=(NB, 1, batch))
    return Ainv
end

function _trsm_apply_inverse_block!(Ainv,
    Btmp,
    B,
    side::Val{SIDE},
    k0::Int,
    block_n::Int) where {SIDE}
    cols = 1:block_n
    block = k0:k0+block_n-1
    # apply the inverted diagonal block.
    if SIDE === :left
        _trsm_matmul!(Btmp,
            _trsm_diag_block(Ainv, cols),
            _trsm_row_block(B, block),
            one(eltype(Btmp)),
            zero(eltype(Btmp)))
        copyto!(_trsm_row_block(B, block), _trsm_row_block(Btmp, cols))
    else
        _trsm_matmul!(Btmp,
            _trsm_col_block(B, block),
            _trsm_A_block(Ainv, cols, :),
            one(eltype(Btmp)),
            zero(eltype(Btmp)))
        copyto!(_trsm_col_block(B, block), _trsm_col_block(Btmp, cols))
    end
    return B
end

function _trsm_blocked!(A, B, side::Val{SIDE}, uplo::Val{UPLO},
    nb::Val{NB}) where {SIDE,UPLO,NB}
    n, batch = _trsm_validate(side, A, B)
    n == 0 && return B

    is_batched = ndims(A) == 3
    Ainv = is_batched ? similar(A, NB, NB, batch) : similar(A, NB, NB)
    Btmp = is_batched ?
           (SIDE === :left ? similar(B, NB, size(B, 2), batch) : similar(B, size(B, 1), NB, batch)) :
           (SIDE === :left ? similar(B, NB, size(B, 2)) : similar(B, size(B, 1), NB))
    iter = if (SIDE === :left && UPLO === :lower) || (SIDE === :right && UPLO === :upper)
        1:NB:n
    else
        _trsm_last_block_start(n, NB):-NB:1
    end

    # blocked diagonal solve and trailing update.
    for k0 in iter
        block_n = min(NB, n - k0 + 1)
        _trsm_device_inverse_block!(Ainv, A, uplo, k0, block_n, nb, batch)
        _trsm_apply_inverse_block!(Ainv, Btmp, B, side, k0, block_n)
        _trsm_update!(A, B, side, uplo, k0, k0 + block_n - 1, n)
    end
    return B
end

function _trsm!(A, B, side::Val{SIDE}, uplo::Val{UPLO}) where {SIDE,UPLO}
    if get_backend(A) isa KernelAbstractions.CPU
        _trsm_validate(side, A, B)
        return _trsm_cpu_fallback!(A, B, side, uplo)
    end
    return _trsm_blocked!(A, B, side, uplo, _trsm_default_nb(eltype(B)))
end

"""
    LeftLowerTRSM!(A, B)

Solve a left-sided lower-triangular system in place.

This function overwrites `B` with the solution `X` of

`A * X = B`

where `A` is interpreted as lower triangular. `A` must be square. For matrix
inputs, `B` must have the same number of rows as `A`. For three-dimensional
inputs, slices `A[:, :, b]` and `B[:, :, b]` are solved independently.

On CPU backends this dispatches to BLAS `trsm!`. On GPU backends NextLA uses a
blocked algorithm: invert each diagonal block in shared memory, apply that
inverse with GEMM, and update the remaining block rows with GEMM or batched
GEMM.
"""
LeftLowerTRSM!(A, B) = _trsm!(A, B, Val(:left), Val(:lower))

"""
    LeftUpperTRSM!(A, B)

Solve a left-sided upper-triangular system in place.

This function overwrites `B` with the solution `X` of

`A * X = B`

where `A` is interpreted as upper triangular. Dimension and batch semantics are
the same as for [`LeftLowerTRSM!`](@ref).
"""
LeftUpperTRSM!(A, B) = _trsm!(A, B, Val(:left), Val(:upper))

"""
    RightLowerTRSM!(A, B)

Solve a right-sided lower-triangular system in place.

This function overwrites `B` with the solution `X` of

`X * A = B`

where `A` is interpreted as lower triangular. `A` must be square. For matrix
inputs, `B` must have the same number of columns as `A`. For three-dimensional
inputs, corresponding batch slices are solved independently.
"""
RightLowerTRSM!(A, B) = _trsm!(A, B, Val(:right), Val(:lower))

"""
    RightUpperTRSM!(A, B)

Solve a right-sided upper-triangular system in place.

This function overwrites `B` with the solution `X` of

`X * A = B`

where `A` is interpreted as upper triangular. Dimension and batch semantics are
the same as for [`RightLowerTRSM!`](@ref).
"""
RightUpperTRSM!(A, B) = _trsm!(A, B, Val(:right), Val(:upper))

@inline _trsm_opA(A, transa::Char) = transa == 'N' ? A : transa == 'T' ? Transpose(A) : Adjoint(A)
@inline _trsm_opA(A::AbstractArray{<:Any,3}, transa::Char) =
    transa == 'N' ? A :
    transa == 'T' ? permutedims(A, (2, 1, 3)) :
    permutedims(conj.(A), (2, 1, 3))
@inline _trsm_effective_uplo(uplo::Char, transa::Char) = transa == 'N' ? uplo : (uplo == 'L' ? 'U' : 'L')
@inline _trsm_dense_opA(opA, backend) = backend isa KernelAbstractions.CPU ? opA : copy(opA)

"""
    trsm!(side, uplo, transa, diag, A, B, alpha=one(eltype(A)))

Solve a triangular matrix equation in place and return `B`.

`side` is `'L'` for `op(A) * X = alpha * B` and `'R'` for
`X * op(A) = alpha * B`. `uplo` selects the stored triangle and `transa`
selects no transpose (`'N'`), transpose (`'T'`), or conjugate transpose (`'C'`).
Only non-unit diagonal matrices (`diag = 'N'`) are currently supported.

`A` must be square in its leading two dimensions. If `side == 'L'`, `size(B, 1)`
must match `size(A, 1)`. If `side == 'R'`, `size(B, 2)` must match `size(A, 1)`.

If `alpha != one(eltype(A))`, `B` is scaled before the solve. The input `A` is
not modified.

For three-dimensional arrays, corresponding slices of `A` and `B` are solved
independently using the batched path. GPU backends use a type-selected blocked
kernel configuration internally; those compile-time tuning parameters are not
part of the public API.
"""
function trsm!(side, uplo, transa, diag, A, B, alpha=one(eltype(A)))
    transa in ('N', 'T', 'C') || throw(ArgumentError("trsm: transa must be 'N', 'T', or 'C'"))
    diag == 'N' || throw(ArgumentError("trsm: only diag='N' is supported"))
    alpha != one(eltype(A)) && (B .*= alpha)

    backend = KernelAbstractions.get_backend(B)
    opA = transa == 'N' ? A : _trsm_dense_opA(_trsm_opA(A, transa), backend)
    effective_uplo = _trsm_effective_uplo(uplo, transa)
    if side == 'L' && effective_uplo == 'L'
        LeftLowerTRSM!(opA, B)
    elseif side == 'L' && effective_uplo == 'U'
        LeftUpperTRSM!(opA, B)
    elseif side == 'R' && effective_uplo == 'L'
        RightLowerTRSM!(opA, B)
    elseif side == 'R' && effective_uplo == 'U'
        RightUpperTRSM!(opA, B)
    else
        error("Unsupported combination of side='$side', uplo='$uplo'")
    end
end

"""
    trsm_batched!(side, uplo, transa, diag, A, B, alpha=one(eltype(A)))

Solve a batch of triangular systems in place.

`A` and `B` must be three-dimensional arrays with equal batch sizes. Each slice
`B[:, :, b]` is overwritten by the solution of the corresponding triangular
system defined by `A[:, :, b]`. The meaning of `side`, `uplo`, `transa`, `diag`,
and `alpha` is the same as for [`trsm`](@ref).

On GPU backends the batched implementation inverts each diagonal block once per
matrix, applies that inverse with batched GEMM, and updates the remaining work
with batched GEMM as well.
"""
trsm_batched!(side, uplo, transa, diag, A::AbstractArray{<:Any,3}, B::AbstractArray{<:Any,3}, alpha=one(eltype(A))) =
    trsm!(side, uplo, transa, diag, A, B, alpha)

function trsm_batched!(side,
                       uplo,
                       transa,
                       diag,
                       A::AbstractVector{<:AbstractMatrix},
                       B::AbstractVector{<:AbstractMatrix},
                       alpha=one(eltype(first(A))))
    length(A) == length(B) || throw(DimensionMismatch("trsm_batched!: matrix batches must have matching lengths"))
    for bid in eachindex(A, B)
        trsm!(side, uplo, transa, diag, A[bid], B[bid], alpha)
    end
    return B
end
