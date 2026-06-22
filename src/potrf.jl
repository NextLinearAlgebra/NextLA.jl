export potrf!

#helpers 
@inline _potrf_default_config(::Type) = (Val(64), Val(64), Val(8), Val(16))

@inline _potrf_batch_size(A::AbstractMatrix) = 1
@inline _potrf_batch_size(A::AbstractArray{<:Any,3}) = size(A, 3)

@inline _potrf_kernel_batch(::AbstractMatrix, group) = 1
@inline _potrf_kernel_batch(::AbstractArray{<:Any,3}, group) = group[end]

@inline _potrf_batch_view(A::AbstractMatrix, b::Int) = A
@inline _potrf_batch_view(A::AbstractArray{<:Any,3}, b::Int) = view(A, :, :, b)

@inline _potrf_getindex(A::AbstractMatrix, i::Int, j::Int, b::Int) = @inbounds A[i, j]
@inline _potrf_getindex(A::AbstractArray{<:Any,3}, i::Int, j::Int, b::Int) = @inbounds A[i, j, b]

@inline _potrf_setindex!(A::AbstractMatrix, value, i::Int, j::Int, b::Int) = (@inbounds A[i, j] = value; nothing)
@inline _potrf_setindex!(A::AbstractArray{<:Any,3}, value, i::Int, j::Int, b::Int) = (@inbounds A[i, j, b] = value; nothing)

@inline _potrf_status_buffer(A::AbstractMatrix) = zeros(Int32, 1)
@inline _potrf_status_buffer(A::AbstractArray{<:Any,3}) = zeros(Int32, size(A, 3))

@inline _potrf_tile_view(A::AbstractMatrix, first::Int, last::Int) = view(A, first:last, first:last)
@inline _potrf_tile_view(A::AbstractArray{<:Any,3}, first::Int, last::Int) = view(A, first:last, first:last, :)

@inline _potrf_panel_view(::Val{:L}, A::AbstractMatrix, k0::Int, k1::Int, n::Int) = view(A, k1+1:n, k0:k1)
@inline _potrf_panel_view(::Val{:U}, A::AbstractMatrix, k0::Int, k1::Int, n::Int) = view(A, k0:k1, k1+1:n)
@inline _potrf_panel_view(::Val{:L}, A::AbstractArray{<:Any,3}, k0::Int, k1::Int, n::Int) = view(A, k1+1:n, k0:k1, :)
@inline _potrf_panel_view(::Val{:U}, A::AbstractArray{<:Any,3}, k0::Int, k1::Int, n::Int) = view(A, k0:k1, k1+1:n, :)

@inline _potrf_trailing_view(A::AbstractMatrix, k1::Int, n::Int) = view(A, k1+1:n, k1+1:n)
@inline _potrf_trailing_view(A::AbstractArray{<:Any,3}, k1::Int, n::Int) = view(A, k1+1:n, k1+1:n, :)

@inline function _potrf_syrk_dispatch!(trailing::AbstractMatrix, panel, uplo::Char, trans::Char, alpha)
    syrk!(uplo, trans, -alpha, panel, alpha, trailing)
    return trailing
end

@inline function _potrf_syrk_dispatch!(trailing::AbstractArray{<:Any,3}, panel, uplo::Char, trans::Char, alpha)
    syrk_batched!(uplo, trans, -alpha, panel, alpha, trailing)
    return trailing
end

@inline function _potrf_validate!(A, status)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("potrf!: A must be square in its first two dimensions"))
    length(status) == _potrf_batch_size(A) ||
        throw(DimensionMismatch("potrf!: status length must match the batch size"))
    return n
end

@inline function _potrf_translate_status!(host_status::AbstractVector{Int32}, k0::Int)
    translated = false
    @inbounds for batch in eachindex(host_status)
        if host_status[batch] != 0
            host_status[batch] += Int32(k0 - 1)
            translated = true
        end
    end
    return translated
end

@inline function _potrf_translate_status!(host_status::AbstractVector{Int32}, active_before, k0::Int)
    translated = false
    @inbounds for batch in eachindex(host_status)
        if active_before[batch] && host_status[batch] != 0
            host_status[batch] += Int32(k0 - 1)
            translated = true
        end
    end
    return translated
end

@inline function _potrf_throw_status!(status_host::AbstractVector{Int32})
    failed_batch = findfirst(!iszero, status_host)
    failed_batch === nothing || throw(PosDefException(status_host[failed_batch]))
    return nothing
end

function _potrf_cpu_fallback!(uplo::Char, A, status::AbstractVector{Int32})
    fill!(status, Int32(0))
    for b in 1:_potrf_batch_size(A)
        _, info = LAPACK.potrf!(uplo, _potrf_batch_view(A, b))
        status[b] = Int32(info)
    end
    return A
end

@inline function _load_diagonal_block!(Ls,
                                       A,
                                       k0::Int,
                                       batch::Int,
                                       tile_n::Int,
                                       tid,
                                       nthreads,
                                       uplo::Val{UPLO},
                                       ::Val{IB}) where {UPLO,IB}
    # stage one diagonal tile.
    @inbounds for j in tid:nthreads:(tile_n * tile_n)
        row = ((j - 1) % tile_n) + 1
        col = ((j - 1) ÷ tile_n) + 1

        value = UPLO === :L ?
                _potrf_getindex(A, k0 + row - 1, k0 + col - 1, batch) :
                _potrf_getindex(A, k0 + col - 1, k0 + row - 1, batch)
        lset!(
            Ls,
            value,
            row,
            col,
            Val(IB),
        )
    end

    return nothing
end

@inline function _store_diagonal_block!(A,
                                        Ls,
                                        k0::Int,
                                        batch::Int,
                                        tile_n::Int,
                                        tid,
                                        nthreads,
                                        uplo::Val{UPLO},
                                        ::Val{IB}) where {UPLO,IB}
    # write the factored triangle back.
    @inbounds for j in tid:nthreads:(tile_n * tile_n)
        row = ((j - 1) % tile_n) + 1
        col = ((j - 1) ÷ tile_n) + 1

        if row >= col
            if UPLO === :L
                _potrf_setindex!(A, lget(Ls, row, col, Val(IB)), k0 + row - 1, k0 + col - 1, batch)
            else
                _potrf_setindex!(A, lget(Ls, row, col, Val(IB)), k0 + col - 1, k0 + row - 1, batch)
            end
        end
    end

    return nothing
end

@inline function _trsm_panel!(A,
                              Ls,
                              status,
                              k0::Int,
                              batch::Int,
                              tid,
                              nthreads,
                              ::Val{IB},
                              ::Val{OB},
                              ::Val{MR},
                              ::Val{RB},
                              uplo::Val{UPLO}) where {IB,OB,MR,RB,UPLO}
    @inbounds status[batch] == 0 || return nothing

    # panel solve under the micro-tile.
    kt = k0 + IB
    kt > OB && return nothing

    for trailing_row_base in kt:MR:OB
        trailing_rows = min(MR, OB - trailing_row_base + 1)

        for row_offset in tid:nthreads:trailing_rows
            global_row = trailing_row_base + row_offset - 1

            @unroll for panel_col in 1:IB
                row_panel_entry = (
                    UPLO === :L ?
                    _potrf_getindex(A, global_row, k0 + panel_col - 1, batch) :
                    _potrf_getindex(A, k0 + panel_col - 1, global_row, batch)
                ) / lget(Ls, panel_col, panel_col, Val(IB))
                if UPLO === :L
                    _potrf_setindex!(A, row_panel_entry, global_row, k0 + panel_col - 1, batch)
                else
                    _potrf_setindex!(A, row_panel_entry, k0 + panel_col - 1, global_row, batch)
                end

                for update_col_base in (panel_col + 1):RB:IB
                    @unroll for i in 1:RB
                        update_col = update_col_base + i - 1
                        if update_col <= IB
                            updated_entry = (
                                UPLO === :L ?
                                _potrf_getindex(A, global_row, k0 + update_col - 1, batch) :
                                _potrf_getindex(A, k0 + update_col - 1, global_row, batch)
                            ) - row_panel_entry * lget(Ls, update_col, panel_col, Val(IB))
                            if UPLO === :L
                                _potrf_setindex!(A, updated_entry, global_row, k0 + update_col - 1, batch)
                            else
                                _potrf_setindex!(A, updated_entry, k0 + update_col - 1, global_row, batch)
                            end
                        end
                    end
                end
            end
        end
    end

    return nothing
end

@kernel unsafe_indices = true function _potrf_fused_tile_kernel!(A,
                                                                 status,
                                                                 k0::Int,
    uplo::Val{UPLO},
                                                                 ::Val{OB},
                                                                 ::Val{IB},
                                                                 ::Val{RB},
                                                                 ::Val{MR},
                                                                 ::Val{W}) where {UPLO,OB,IB,RB,MR,W}
    tid = @index(Local, Linear)
    group = @index(Group, NTuple)
    batch = _potrf_kernel_batch(A, group)
    nthreads = @uniform @groupsize()[1]

    # fuse potf2 and panel trsm while the diagonal micro-panel remains in local memory.
    Ls = @localmem eltype(A) (IB, IB)

    if @inbounds status[batch] == 0
        # stage, factor, store, and immediately consume the diagonal micro-panel.
        _load_diagonal_block!(Ls, A, k0, batch, IB, tid, nthreads, uplo, Val(IB))
        @synchronize()

        _potf2(Ls, status, batch, k0, tid, nthreads, Val(IB), Val(RB), Val(W))
        @synchronize()

        _store_diagonal_block!(A, Ls, k0, batch, IB, tid, nthreads, uplo, Val(IB))
        _trsm_panel!(A, Ls, status, k0, batch, tid, nthreads, Val(IB), Val(OB), Val(MR), Val(RB), uplo)
    end
end

@kernel unsafe_indices = true function _potf2_partial_kernel!(A,
                                                              status,
                                                              base_k::Int,
                                                              tile_n::Int,
                                                              uplo::Val{UPLO},
                                                              ::Val{IB},
                                                              ::Val{RB},
                                                              ::Val{W}) where {UPLO,IB,RB,W}
    tid = @index(Local, Linear)
    group = @index(Group, NTuple)
    batch = _potrf_kernel_batch(A, group)
    nthreads = @uniform @groupsize()[1]
    Ls = @localmem eltype(A) (IB, IB)

    if @inbounds status[batch] == 0
        _load_diagonal_block!(Ls, A, 1, batch, tile_n, tid, nthreads, uplo, Val(IB))
        @synchronize()

        _potf2_partial(Ls, status, batch, base_k, tile_n, tid, nthreads, Val(IB), Val(RB), Val(W))
        @synchronize()

        _store_diagonal_block!(A, Ls, 1, batch, tile_n, tid, nthreads, uplo, Val(IB))
    end
end

function _potrf_blocked!(uplo::Char,
                         A,
                         status::AbstractVector{Int32},
                         ::Val{OB},
                         ::Val{IB},
                         ::Val{RB},
                         ::Val{MR}) where {OB,IB,RB,MR}

    @assert IB % RB == 0
    
    backend = KernelAbstractions.get_backend(A)
    backend isa KernelAbstractions.CPU && return _potrf_cpu_fallback!(uplo, A, status)
    W = _val_parameter(SUBGROUP_SIZE(typeof(backend)))
    
    # --- Guard Checks & Initializations ---    
    n = _potrf_validate!(A, status)
    trans = eltype(A) <: Real ? 'T' : 'C'
    alpha = one(eltype(A))
    host_status = _potrf_status_buffer(A)
    batch_count = ndims(A) == 2 ? 1 : size(A, 3)

    uplo_val = uplo == 'L' ? Val(:L) :
               uplo == 'U' ? Val(:U) :
               throw(ArgumentError("potrf!: uplo must be 'L' or 'U'"))

    fill!(status, Int32(0))

    # Pre-instantiate the kernels outside the loop
    fused_tile_kernel = _potrf_fused_tile_kernel!(backend, (512, 1, 1))
    partial_kernel     = _potf2_partial_kernel!(backend, (256, 1, 1))

    @inbounds for k0 in 1:OB:n
        # outer blocked sweep.
        k1 = min(k0 + OB - 1, n)
        diag_tile = _potrf_tile_view(A, k0, k1)
        outer_n = size(diag_tile, 1)
        active_before = iszero.(host_status)

        fused_n = (outer_n ÷ IB) * IB
        partial_n = outer_n - fused_n

        #  1: factor the Diagonal Block (_potrf_factor_outer_tile!)
        if fused_n > 0
            # fused leading micro-tiles.
            fused_tile = _potrf_tile_view(diag_tile, 1, fused_n)
            
            # Inlined: _potrf_fused_tile!
            @unroll for k0_fused in 1:IB:fused_n
                fused_tile_kernel(fused_tile, status, k0_fused, uplo_val, Val(fused_n), Val(IB), Val(RB), Val(MR), Val(W); ndrange=(512, 1, batch_count))
                KernelAbstractions.synchronize(backend)
                copyto!(host_status, status)
                
                all(x -> !iszero(x), host_status) && break

                kt = k0_fused + IB
                if kt <= fused_n
                    panel = _potrf_panel_view(uplo_val, fused_tile, k0_fused, kt - 1, fused_n)
                    trailing = _potrf_trailing_view(fused_tile, kt - 1, fused_n)
                    _potrf_syrk_dispatch!(trailing, panel, uplo == 'L' ? 'L' : 'U', uplo == 'L' ? 'N' : 'T', alpha)
                end
            end
            
            copyto!(host_status, status)
            if all(x -> !iszero(x), host_status)
                _potrf_translate_status!(host_status, active_before, k0) && copyto!(status, host_status)
                return A
            end
        end

        if partial_n > 0
            if fused_n > 0
                # update the diagonal tile tail.
                # Inlined: _potrf_update_trailing! (Internal Block Update)
                fused_tile = _potrf_tile_view(diag_tile, 1, fused_n)
                panel = _potrf_panel_view(uplo_val, diag_tile, 1, fused_n, outer_n)
                trailing = _potrf_trailing_view(diag_tile, fused_n, outer_n)
                if uplo == 'L'
                    trsm!('R', 'L', trans, 'N', fused_tile, panel, alpha)
                    _potrf_syrk_dispatch!(trailing, panel, 'L', 'N', alpha)
                else
                    trsm!('L', 'U', trans, 'N', fused_tile, panel, alpha)
                    _potrf_syrk_dispatch!(trailing, panel, 'U', trans, alpha)
                end
            end

            # runtime tail micro-tile.
            partial_tile = _potrf_tile_view(diag_tile, fused_n + 1, outer_n)
            partial_kernel(partial_tile, status, fused_n + 1, partial_n, uplo_val, Val(IB), Val(RB), Val(W); ndrange=(256, 1, batch_count))
            KernelAbstractions.synchronize(backend)
            copyto!(host_status, status)
        end

        # 2: status translation & error
        _potrf_translate_status!(host_status, active_before, k0) && copyto!(status, host_status)
        all(!iszero, host_status) && return A

        k1 == n && continue

        # 3: update trailing matrix
        panel = _potrf_panel_view(uplo_val, A, k0, k1, n)
        trailing = _potrf_trailing_view(A, k1, n)
        # update the global trailing matrix.
        if uplo == 'L'
            trsm!('R', 'L', trans, 'N', diag_tile, panel, alpha)
            _potrf_syrk_dispatch!(trailing, panel, 'L', 'N', alpha)
        else
            trsm!('L', 'U', trans, 'N', diag_tile, panel, alpha)
            _potrf_syrk_dispatch!(trailing, panel, 'U', trans, alpha)
        end
    end

    copyto!(status, host_status)
    return A
end

function _potrf_throwing!(uplo::Char,
                A,
                ::Val{OB},
                ::Val{IB},
                ::Val{RB},
                ::Val{MR}) where {OB,IB,RB,MR}
    status = similar(A, Int32, _potrf_batch_size(A))
    _potrf_blocked!(uplo, A, status, Val(OB), Val(IB), Val(RB), Val(MR))
    _potrf_throw_status!(Array(status))
    return A
end

function _potrf_checked!(uplo::Char,
                A,
                ::Val{OB},
                ::Val{IB},
                ::Val{RB},
                ::Val{MR};
                check::Bool = true) where {OB,IB,RB,MR}
    if check
        return _potrf_throwing!(uplo, A, Val(OB), Val(IB), Val(RB), Val(MR))
    end

    status = similar(A, Int32, _potrf_batch_size(A))
    _potrf_blocked!(uplo, A, status, Val(OB), Val(IB), Val(RB), Val(MR))
    return A, status
end

function potrf!(uplo::Char,
                A::AbstractMatrix{T},
                ::Val{OB},
                ::Val{IB},
                ::Val{RB},
                ::Val{MR};
                check::Bool = true) where {T,OB,IB,RB,MR}
    return _potrf_checked!(uplo, A, Val(OB), Val(IB), Val(RB), Val(MR); check)
end

function potrf!(uplo::Char,
                A::AbstractArray{T,3},
                ::Val{OB},
                ::Val{IB},
                ::Val{RB},
                ::Val{MR};
                check::Bool = true) where {T,OB,IB,RB,MR}
    return _potrf_checked!(uplo, A, Val(OB), Val(IB), Val(RB), Val(MR); check)
end

function potrf_batched!(uplo::Char,
                A::AbstractArray{T,3},
                status::AbstractVector{Int32},
                ::Val{OB},
                ::Val{IB},
                ::Val{RB},
                ::Val{MR}) where {T,OB,IB,RB,MR}
    return _potrf_blocked!(uplo, A, status, Val(OB), Val(IB), Val(RB), Val(MR))
end

function potrf_batched!(uplo::Char,
                A::AbstractArray{T,3},
                ::Val{OB},
                ::Val{IB},
                ::Val{RB},
                ::Val{MR};
                check::Bool = true) where {T,OB,IB,RB,MR}
    return _potrf_checked!(uplo, A, Val(OB), Val(IB), Val(RB), Val(MR); check)
end

"""
    potrf!(uplo, A; check=true)

Compute an in-place Cholesky factorization of the square matrix `A`.

If `uplo == 'L'`, this computes a lower-triangular factor `L` such that

`A ≈ L * L'`

and stores `L` in the lower triangle of `A`. If `uplo == 'U'`, this computes an
upper-triangular factor `U` such that

`A ≈ U' * U`

and stores `U` in the upper triangle of `A`.

By default this method throws `PosDefException` if the factorization encounters
a non-positive pivot. Pass `check=false` to suppress the exception and return
`(A, status)`, where `status[1]` is `0` on success or the 1-based index of the
first failing diagonal entry.

GPU backends use a blocked factorization with type-selected compile-time tile
sizes. Diagonal micro-panels are factored in shared memory, panel solves are
applied immediately, and the trailing matrix is updated with BLAS-like kernels.
Those tuning constants are internal and are not part of the public API.

At the moment the GPU implementation does not support `Float16` end to end.
The blocked update path depends on `syrk`, and the required `Float16` `syrk`
backend is not available here, so `potrf!` currently fails for `Float16`.
"""
function potrf!(uplo::Char, A::AbstractMatrix{T}; check::Bool = true) where {T}
     ob, ib, rb, mr = _potrf_default_config(T)
    return potrf!(uplo, A,  ob, ib, rb, mr; check)
end

function potrf!(uplo::Char, A::AbstractArray{T,3}; check::Bool = true) where {T}
     ob, ib, rb, mr = _potrf_default_config(T)
    return potrf!(uplo, A,  ob, ib, rb, mr; check)
end

"""
    potrf_batched!(uplo, A; check=true)

Compute an in-place batched Cholesky factorization of `A`.

`A` must be a three-dimensional array whose slices `A[:, :, b]` are factored
independently.

By default this method throws `PosDefException` if any batch entry encounters a
non-positive pivot. Pass `check=false` to suppress the exception and return
`(A, status)`, where `status[b]` is `0` on success or the 1-based index of the
first non-positive pivot in batch entry `b`.

On GPU backends the batched implementation reuses the same blocked algorithm as
the matrix case, but launches fused POTF2/TRSM kernels and trailing updates over
the batch dimension.

As in the matrix case, the current GPU implementation does not support
`Float16` end to end because the blocked trailing update depends on `syrk`,
and the required `Float16` `syrk` backend is not available here.
"""
function potrf_batched!(uplo::Char, A::AbstractArray{T,3}; check::Bool = true) where {T}
     ob, ib, rb, mr = _potrf_default_config(T)
    return potrf_batched!(uplo, A,  ob, ib, rb, mr; check)
end

function potrf_batched!(uplo::Char, A::AbstractArray{T,3}, status::AbstractVector{Int32}) where {T}
     ob, ib, rb, mr = _potrf_default_config(T)
    return potrf_batched!(uplo, A, status,  ob, ib, rb, mr)
end
