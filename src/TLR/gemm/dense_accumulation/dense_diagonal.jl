@inline diag_tile_ref(A::TLRMatrix, k::Int) = (diag_tile_view(A, k), 'N')
@inline diag_tile_ref(A::TransposeTLRMatrix{<:Any,<:TLRMatrix}, k::Int) =
    (diag_tile_view(parent(A), k), 'T')

"""Nonzero-execution-rank `(i,k,r)` tiles between `OA`'s off-diagonal factors
and `DB`'s diagonal, paired with each tile's intermediate (`S = V'D`) element
count. Shared by the workspace-bound queries and `tlr_offdiag_times_diag!`'s
packing loop, so the two can never enumerate different tile sets."""
function _tlr_right_cross_items(OA, DB)
    qm, qk = grid_size(OA)
    items = Tuple{Int,Int,Int}[]
    @inbounds for k in 1:min(qk, ndiag_tiles(DB)), i in 1:qm
        r = compressed_ftlr_storage_rank(OA, i, k)
        r == 0 || push!(items, (i, k, r))
    end

    sizes = [r * length(tile_axis_range(DB, k, 2)) for (_, k, r) in items]
    return items, sizes
end

"""Mirror of [`_tlr_right_cross_items`](@ref): nonzero-execution-rank `(k,j,r)`
tiles between `DA`'s diagonal and `OB`'s off-diagonal factors, paired with
each tile's intermediate (`W = DU`) element count."""
function _tlr_left_cross_items(OB, DA)
    qk, qn = grid_size(OB)
    items = Tuple{Int,Int,Int}[]
    @inbounds for k in 1:min(qk, ndiag_tiles(DA)), j in 1:qn
        r = compressed_ftlr_storage_rank(OB, k, j)
        r == 0 || push!(items, (k, j, r))
    end

    sizes = [length(tile_axis_range(DA, k, 1)) * r for (k, _, r) in items]
    return items, sizes
end

"""Workspace element counts for the two off-diagonal/dense-diagonal cross terms."""
function _tlr_diagonal_intermediate_sizes(A::TLRMatrix, B::TLRMatrix,
                                          transA::Char, transB::Char)
    DA = transA == 'T' ? transpose(A) : A
    DB = transB == 'T' ? transpose(B) : B
    OA = offdiagonal(DA)
    OB = offdiagonal(DB)
    _, qk = grid_size(OA)
    qkB, _ = grid_size(OB)

    qk == qkB || throw(DimensionMismatch("TLR contraction grids do not match"))

    _, right = _tlr_right_cross_items(OA, DB)
    _, left = _tlr_left_cross_items(OB, DA)
    return right, left
end

function gemm_minimum_workspace_bytes(A::TLRMatrix, B::TLRMatrix;
                                      transA::Char='N', transB::Char='N')
    compressed = gemm_minimum_workspace_bytes(
        offdiagonal(A), offdiagonal(B); transA, transB)
    right, left = _tlr_diagonal_intermediate_sizes(A, B, transA, transB)
    one_cross = max(isempty(right) ? 0 : maximum(right),
                    isempty(left) ? 0 : maximum(left)) * sizeof(eltype(A))
    return max(compressed, one_cross)
end

@inline function _tlr_aligned_intermediate_elements(sizes, ::Type{T}) where {T}
    alignment = gemm_alignment_quantum(T)
    total = 0
    @inbounds for count in sizes
        total = cld(total, alignment) * alignment + count
    end
    return total
end

function gemm_maximum_workspace_bytes(A::TLRMatrix, B::TLRMatrix;
                                      transA::Char='N', transB::Char='N')
    compressed = gemm_maximum_workspace_bytes(
        offdiagonal(A), offdiagonal(B); transA, transB)
    right, left = _tlr_diagonal_intermediate_sizes(A, B, transA, transB)
    T = eltype(A)
    all_cross = max(_tlr_aligned_intermediate_elements(right, T),
                    _tlr_aligned_intermediate_elements(left, T)) * sizeof(T)
    return max(compressed, all_cross)
end

"""
    _tlr_pack_cross_items!(mode, arena, capacity, T, items, sizes, push_item!)

Pack dense-diagonal cross terms into 16-byte-aligned arena passes. `sizes`
contains each intermediate's elements; `push_item!` carves that intermediate
and appends its two `GroupedGemmTask`s.
"""
function _tlr_pack_cross_items!(mode, arena, capacity::Int, ::Type{T}, items, sizes,
                                push_item!) where {T}
    if !isempty(sizes)
        needed = maximum(sizes) * sizeof(T)
        capacity >= needed || throw(ArgumentError(
            "workspace has $capacity bytes; a dense-diagonal update requires at least $needed bytes"))
    end

    first_item = 1
    while first_item <= length(items)
        arena_reset!(arena)
        stage1 = GroupedGemmTask[]
        stage2 = GroupedGemmTask[]

        # aligned items for this pass
        used = 0
        item = first_item
        while item <= length(items)
            count = sizes[item]
            aligned_used = cld(used, gemm_alignment_quantum(T)) * gemm_alignment_quantum(T)
            aligned_used + count <= length(arena.storage) || break
            arena.cursor = firstindex(arena.storage) + aligned_used
            push_item!(stage1, stage2, item)
            used = aligned_used + count
            item += 1
        end

        precision_gemm_grouped!(stage1, mode)
        precision_gemm_grouped!(stage2, mode)
        first_item = item
    end
    return nothing
end

# O_A D_B:  S_ik = V_A_ik' D_B_kk  (stage 1),  C[rows,cols] += α U_A_ik S_ik  (stage 2).
function tlr_offdiag_times_diag!(C, OA, DB, alpha, mode, arena, capacity::Int)
    T = eltype(OA)
    items, sizes = _tlr_right_cross_items(OA, DB)
    backend = get_backend(OA)

    # cross-term batches
    _tlr_pack_cross_items!(mode, arena, capacity, T, items, sizes,
        (stage1, stage2, idx) -> begin
            i, k, r = items[idx]
            cols = tile_axis_range(DB, k, 2)
            S = workspace_array!(arena, backend, T, r, length(cols))
            V = compressed_ftlr_storage_inner(OA, i, k)
            D, opD = diag_tile_ref(DB, k)
            push!(stage1, GroupedGemmTask(
                'T', opD, one(T), V, D, zero(T), S))
            U = compressed_ftlr_storage_outer(OA, i, k)
            rows = tile_axis_range(OA, i, 1)
            push!(stage2, GroupedGemmTask(
                'N', 'N', alpha, U, S, one(alpha), view(C, rows, cols)))
        end)
    return C
end

# D_A O_B:  W_kj = D_A_kk U_B_kj  (stage 1),  C[rows,cols] += α W_kj V_B_kj'  (stage 2).
function tlr_diag_times_offdiag!(C, DA, OB, alpha, mode, arena, capacity::Int)
    T = eltype(OB)
    items, sizes = _tlr_left_cross_items(OB, DA)
    backend = get_backend(OB)

    # cross-term batches
    _tlr_pack_cross_items!(mode, arena, capacity, T, items, sizes,
        (stage1, stage2, idx) -> begin
            k, j, r = items[idx]
            rows = tile_axis_range(DA, k, 1)
            W = workspace_array!(arena, backend, T, length(rows), r)
            U = compressed_ftlr_storage_outer(OB, k, j)
            D, opD = diag_tile_ref(DA, k)
            push!(stage1, GroupedGemmTask(
                opD, 'N', one(T), D, U, zero(T), W))
            V = compressed_ftlr_storage_inner(OB, k, j)
            cols = tile_axis_range(OB, j, 2)
            push!(stage2, GroupedGemmTask(
                'N', 'T', alpha, W, V, one(alpha), view(C, rows, cols)))
        end)
    return C
end

function tlr_diag_times_diag!(C, DA, DB, alpha, mode)
    n = min(ndiag_tiles(DA), ndiag_tiles(DB))
    tasks = GroupedGemmTask[]
    sizehint!(tasks, n)

    # diagonal product tasks
    @inbounds for k in 1:n
        Atile, opA = diag_tile_ref(DA, k)
        Btile, opB = diag_tile_ref(DB, k)
        rows = tile_axis_range(DA, k, 1)
        cols = tile_axis_range(DB, k, 2)
        push!(tasks, GroupedGemmTask(
            opA, opB, alpha, Atile, Btile, one(alpha),
            view(C, rows, cols)))
    end

    isempty(tasks) || precision_gemm_grouped!(tasks, mode)
    return C
end

"""Add the block-diagonal part of `op(A::TLRMatrix) * op(B::Matrix)` to `C`.

The off-diagonal part is handled by the compressed two-stage lowering. Each
dense diagonal tile contributes to one disjoint output row block, so these
updates need no numerical workspace and can be submitted as one grouped GEMM.
"""
function tlr_diag_times_dense!(C, DA, B, alpha, mode)
    tasks = GroupedGemmTask[]
    sizehint!(tasks, ndiag_tiles(DA))
    output_cols = 1:size(C, 2)

    # diagonal-row updates
    @inbounds for k in 1:ndiag_tiles(DA)
        D, opD = diag_tile_ref(DA, k)
        rows = tile_axis_range(DA, k, 1)
        inner = tile_axis_range(DA, k, 2)
        Bdense, opB = dense_block(B, inner, output_cols)
        push!(tasks, GroupedGemmTask(
            opD, opB, alpha, D, Bdense, one(alpha),
            view(C, rows, output_cols)))
    end

    isempty(tasks) || precision_gemm_grouped!(tasks, mode)
    return C
end

"""Add the block-diagonal part of `op(A::Matrix) * op(B::TLRMatrix)` to `C`.

This is the mirror of [`tlr_diag_times_dense!`](@ref): each diagonal tile
updates one disjoint output column block after the compressed FoldLeft pass.
"""
function dense_times_tlr_diag!(C, A, DB, alpha, mode)
    tasks = GroupedGemmTask[]
    sizehint!(tasks, ndiag_tiles(DB))
    output_rows = 1:size(C, 1)

    # diagonal-column updates
    @inbounds for k in 1:ndiag_tiles(DB)
        D, opD = diag_tile_ref(DB, k)
        inner = tile_axis_range(DB, k, 1)
        cols = tile_axis_range(DB, k, 2)
        Adense, opA = dense_block(A, output_rows, inner)
        push!(tasks, GroupedGemmTask(
            opA, opD, alpha, Adense, D, one(alpha),
            view(C, output_rows, cols)))
    end

    isempty(tasks) || precision_gemm_grouped!(tasks, mode)
    return C
end
