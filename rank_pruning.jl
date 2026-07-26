function _launch_prune_columns!(U::AbstractArray{T,3},
                                V,
                                ranks,
                                error_sq,
                                active_columns::Int,
                                maxrank::Int,
                                tol_sq::Float64,
                                relative::Bool,
                                mode,
) where {T}
    count = size(U, 3)
    count == size(V, 3) == length(ranks) == length(error_sq) ||
        throw(DimensionMismatch("factor batches, ranks, and errors must have equal lengths"))
    size(U, 2) == size(V, 2) ||
        throw(DimensionMismatch("U and V must have the same factor capacity"))
    0 <= active_columns <= size(U, 2) ||
        throw(ArgumentError("active_columns must fit within the factor capacity"))
    0 <= maxrank <= active_columns ||
        throw(ArgumentError("maxrank must satisfy 0 <= maxrank <= active_columns"))
    tol_sq >= 0 || throw(ArgumentError("tol_sq must be nonnegative"))

    count == 0 && return U
    if active_columns == 0
        fill!(ranks, zero(eltype(ranks)))
        fill!(U, zero(T))
        fill!(V, zero(eltype(V)))
        return U
    end

    backend = get_backend(U)
    W = unwrap(SUBGROUP_SIZE(typeof(backend)))
    nthreads = W * min(active_columns, 8)
    _prune_columns_kernel(backend, nthreads)(
        U, V, ranks, error_sq, tol_sq, relative, maxrank,
        mode, Val{active_columns}(), Val{W}();
        ndrange=(nthreads * count,), workgroupsize=nthreads,
    )
    return U
end

"""
    prune_orthogonal_columns!(
        Q, V, ranks, error_sq, active_columns, maxrank, tol_sq, relative,
    ) -> Q

Fused exact-coordinate pruning for `Q * V'` when the active columns of `Q` are
orthonormal. `error_sq` contains any already-incurred squared error on entry
and is overwritten by that base error plus the energy discarded here.

No standalone column-energy array is materialized: norms, selection,
compaction, rank output, and zero padding remain one fused launch.
"""
function prune_orthogonal_columns!(Q::AbstractArray{T,3},
                                   V,
                                   ranks,
                                   error_sq,
                                   active_columns::Int,
                                   maxrank::Int,
                                   tol_sq::Float64,
                                   relative::Bool,
) where {T}
    return _launch_prune_columns!(
        Q, V, ranks, error_sq, active_columns, maxrank, tol_sq,
        relative, Val(:orthogonal),
    )
end

"""
    prune_cholqr_coordinates!(
        Q, V, ranks, discarded_coefficient_sq,
        active_columns, maxrank, rank_tol_sq,
    ) -> Q

Fused numerical-rank selection after shifted CholQR2. Near-null CholQR
coordinates need not themselves be orthonormal, so the reported discarded
coefficient energy is a rank-revelation diagnostic rather than a claim of exact
matrix approximation error.
"""
function prune_cholqr_coordinates!(Q::AbstractArray{T,3},
                                   V,
                                   ranks,
                                   discarded_coefficient_sq,
                                   active_columns::Int,
                                   maxrank::Int,
                                   rank_tol_sq::Float64,
                                   relative::Bool=true,
) where {T}
    return _launch_prune_columns!(
        Q, V, ranks, discarded_coefficient_sq, active_columns, maxrank,
        rank_tol_sq, relative, Val(:cholqr_rank),
    )
end

"""
    mixed_cholqr2_compress!(
        ws, ranks, error_sq, maxrank, rank_tol_sq, relative=true,
    ) -> ws

Factor-producing shifted CholQR2 followed by fused coordinate-energy rank
selection. This is the reusable tall-skinny panel operation required by the
orthogonal merge. `rank_tol_sq` is intended for numerical-rank revelation;
the final merge pruning remains responsible for the application tolerance.
"""
function mixed_cholqr2_compress!(ws::CholQR2FactorWorkspace,
                                 ranks,
                                 error_sq,
                                 maxrank::Int,
                                 rank_tol_sq::Float64,
                                 relative::Bool=true,
)
    mixed_cholqr2_factor!(ws)
    active_columns = size(ws.Q, 2)
    prune_cholqr_coordinates!(ws.Q,
                              ws.V, ranks, error_sq, active_columns, maxrank,
                              rank_tol_sq, relative,
    )
    return ws
end

"""
    _prune_columns_kernel(...)

One workgroup handles one factor pair. The policy mode is a compile-time `Val`
distinguishing exact orthogonal-coordinate pruning from the shifted-CholQR
numerical-rank diagnostic.
"""
@kernel function _prune_columns_kernel(U::AbstractArray{T,3},
                                       V::AbstractArray{T,3},
                                       ranks::AbstractVector,
                                       error_sq::AbstractVector{Terr},
                                       tol_sq::Float64,
                                       relative::Bool,
                                       maxrank::Int,
                                       ::Val{Mode},
                                       ::Val{S},
                                       ::Val{W},
) where {T,Terr,Mode,S,W}
    ob = @index(Group, Linear)
    tid = @index(Local, Linear)
    nthreads = @uniform @groupsize()[1]

    energies = @localmem Float64 (S,)
    dropped_flag = @localmem Int32 (S,)
    order = @localmem Int32 (S,)
    move_dst = @localmem Int32 (S,)
    rank_moves = @localmem Int32 (2,)

    # Column energies are intentionally fused into pruning. A separate reusable
    # norm primitive would add a launch and a global S×batch intermediate.
    for col in tid:nthreads:S
        acc = 0.0
        for row in axes(V, 1)
            @inbounds acc += _abs2_f64(V[row, col, ob])
        end
        @inbounds energies[col] = acc
    end

    @synchronize

    if tid == 1
        total = 0.0
        @inbounds for i in 1:S
            total += energies[i]
            dropped_flag[i] = Int32(0)
            order[i] = Int32(i)
        end

        # Stable insertion sort gives deterministic source-index tie breaking.
        @inbounds for i in 2:S
            key = order[i]
            key_energy = energies[Int(key)]
            j = i - 1
            while j >= 1
                previous = order[j]
                previous_energy = energies[Int(previous)]
                (
                    previous_energy < key_energy ||
                    (previous_energy == key_energy && previous < key)
                ) && break
                order[j + 1] = previous
                j -= 1
            end
            order[j + 1] = key
        end

        supplied_error = Float64(real(@inbounds error_sq[ob]))
        reference_sq = total
        base_error = supplied_error
        budget = 0.0

        base_error = Mode === :orthogonal ? max(base_error, 0.0) : 0.0
        target = relative ? tol_sq * reference_sq : tol_sq
        budget = target - base_error

        retained = S
        dropped = 0.0
        next_drop = 1
        if budget >= 0.0
            while next_drop <= S
                col = @inbounds order[next_drop]
                energy = @inbounds energies[Int(col)]
                if energy <= budget
                    @inbounds dropped_flag[Int(col)] = Int32(1)
                    budget -= energy
                    dropped += energy
                    retained -= 1
                    next_drop += 1
                else
                    break
                end
            end
        end

        # The hard capacity is authoritative. Any tolerance violation remains
        # visible in the achieved error written below.
        while retained > maxrank
            col = @inbounds order[next_drop]
            @inbounds dropped_flag[Int(col)] = Int32(1)
            dropped += @inbounds energies[Int(col)]
            next_drop += 1
            retained -= 1
        end

        # Pair the leftmost hole with the rightmost retained source. This is a
        # deterministic minimum-move in-place compaction map.
        left = 1
        right = S
        nmoves = 0
        while true
            @inbounds while left <= S && dropped_flag[left] == 0
                left += 1
            end
            @inbounds while right >= 1 && dropped_flag[right] != 0
                right -= 1
            end
            left < right || break
            nmoves += 1
            @inbounds move_dst[nmoves] = Int32(left)
            @inbounds order[nmoves] = Int32(right)
            left += 1
            right -= 1
        end

        @inbounds ranks[ob] = eltype(ranks)(retained)
        @inbounds error_sq[ob] = Terr(base_error + dropped)
        @inbounds rank_moves[1] = Int32(retained)
        @inbounds rank_moves[2] = Int32(nmoves)
    end

    @synchronize

    lane = (tid - 1) % W + 1
    subgroup = (tid - 1) ÷ W + 1
    nsubgroups = nthreads ÷ W
    nmoves = Int(@inbounds rank_moves[2])
    move = subgroup
    while move <= nmoves
        dst = Int(@inbounds move_dst[move])
        src = Int(@inbounds order[move])
        row = lane
        while row <= size(U, 1)
            @inbounds U[row, dst, ob] = U[row, src, ob]
            row += W
        end
        row = lane
        while row <= size(V, 1)
            @inbounds V[row, dst, ob] = V[row, src, ob]
            row += W
        end
        move += nsubgroups
    end

    @synchronize

    lane = (tid - 1) % W + 1
    subgroup = (tid - 1) ÷ W + 1
    nsubgroups = nthreads ÷ W
    retained = Int(@inbounds rank_moves[1])
    col = retained + 1 + (subgroup - 1)
    while col <= size(U, 2)
        row = lane
        while row <= size(U, 1)
            @inbounds U[row, col, ob] = zero(T)
            row += W
        end
        row = lane
        while row <= size(V, 1)
            @inbounds V[row, col, ob] = zero(T)
            row += W
        end
        col += nsubgroups
    end
end
