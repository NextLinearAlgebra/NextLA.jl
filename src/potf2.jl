@inline function swz_row(row::Int, col::Int, ::Val{IB}) where {IB}
    return (xor(row - 1, col - 1) & (IB - 1)) + 1
end

# swizzle shared-memory to prevent bank conflicts
@inline function lget(Ls, row::Int, col::Int, ::Val{IB}) where {IB}
    @inbounds return Ls[swz_row(row, col, Val(IB)), col]
end

@inline function lset!(Ls, val, row::Int, col::Int, ::Val{IB}) where {IB}
    @inbounds Ls[swz_row(row, col, Val(IB)), col] = val
    return nothing
end

"""
    _potf2(Ls, status, status_index, base_k, tid, nthreads, Val(IB), Val(RB), Val(W))

Internal unblocked Cholesky factorization on one `IB x IB` tile already resident
in local/shared memory `Ls`.

This routine is intentionally inlineable so fused tile `potrf` kernels can call it
directly after staging the diagonal subtile into shared memory.
"""
@inline function _potf2(Ls,
                        status,
                        status_index::Int,
                        base_k::Int,
                        tid,
                        nthreads,
                        ::Val{IB},
                        ::Val{RB},
                        ::Val{W}) where {IB,RB,W}
    subgroup_lane = ((tid - 1) % W) + 1
    subgroup_in_workgroup = ((tid - 1) ÷ W) + 1
    nsubgroups = nthreads ÷ W

    T = @uniform eltype(Ls)
    zero0 = @uniform zero(T)

    # column sweep inside one shared tile.
    j = 1
    while j <= IB
        # factor one column
        raw_diag = lget(Ls, j, j, Val(IB))

        if raw_diag <= zero0
            if tid == 1
                @inbounds status[status_index] = Int32(base_k + j - 1)
            end

            return nothing
        end

        diag = sqrt(raw_diag)

        if tid == 1
            lset!(Ls, diag, j, j, Val(IB))
        end

        row = j + tid
        while row <= IB
            x = lget(Ls, row, j, Val(IB))
            lset!(Ls, x / diag, row, j, Val(IB))
            row += nthreads
        end

        @synchronize()

        cbase = j + 1 + (subgroup_in_workgroup - 1) * RB

        # apply its rank-1 update cooperatively.
        while cbase <= IB
            row = cbase + subgroup_lane - 1

            while row <= IB
                xj = lget(Ls, row, j, Val(IB))

                @unroll for i in 1:RB
                    ck = cbase + i - 1

                    pivot = ifelse(
                        ck <= IB,
                        lget(Ls, ck, j, Val(IB)),
                        zero0,
                    )

                    if ck <= IB && row >= ck
                        old = lget(Ls, row, ck, Val(IB))
                        lset!(Ls, old - xj * pivot, row, ck, Val(IB))
                    end
                end

                row += W
            end

            cbase += nsubgroups * RB
        end

        @synchronize()

        j += 1
    end

    return nothing
end

"""
    _potf2_partial(Ls, status, status_index, base_k, tile_n, tid, nthreads, Val(IB), Val(RB), Val(W))

Internal unblocked Cholesky factorization on a runtime-sized `tile_n x tile_n`
tile resident in an `IB x IB` local/shared-memory allocation.
"""
@inline function _potf2_partial(Ls,
                                status,
                                status_index::Int,
                                base_k::Int,
                                tile_n::Int,
                                tid,
                                nthreads,
                                ::Val{IB},
                                ::Val{RB},
                                ::Val{W}) where {IB,RB,W}
    subgroup_lane = ((tid - 1) % W) + 1
    subgroup_in_workgroup = ((tid - 1) ÷ W) + 1
    nsubgroups = nthreads ÷ W

    T = @uniform eltype(Ls)
    zero0 = @uniform zero(T)

    # runtime tail column sweep.
    j = 1
    while j <= tile_n
        # guard the runtime tail 
        raw_diag = lget(Ls, j, j, Val(IB))

        if raw_diag <= zero0
            if tid == 1
                @inbounds status[status_index] = Int32(base_k + j - 1)
            end

            return nothing
        end

        diag = sqrt(raw_diag)

        if tid == 1
            lset!(Ls, diag, j, j, Val(IB))
        end

        row = j + tid
        while row <= tile_n
            x = lget(Ls, row, j, Val(IB))
            lset!(Ls, x / diag, row, j, Val(IB))
            row += nthreads
        end

        @synchronize()

        cbase = j + 1 + (subgroup_in_workgroup - 1) * RB

        while cbase <= tile_n
            row = cbase + subgroup_lane - 1

            while row <= tile_n
                xj = lget(Ls, row, j, Val(IB))

                @unroll for i in 1:RB
                    ck = cbase + i - 1

                    pivot = ifelse(
                        ck <= tile_n,
                        lget(Ls, ck, j, Val(IB)),
                        zero0,
                    )

                    if ck <= tile_n && row >= ck
                        old = lget(Ls, row, ck, Val(IB))
                        lset!(Ls, old - xj * pivot, row, ck, Val(IB))
                    end
                end

                row += W
            end

            cbase += nsubgroups * RB
        end

        @synchronize()

        j += 1
    end

    return nothing
end
