"""Nonzero-execution-rank `(i,k,r)` tiles between `OA`'s off-diagonal factors
and `DB`'s diagonal, paired with each tile's intermediate (`S = V'D`) element
count. Shared by the workspace-bound queries and `_tlr_offdiag_times_diag!`'s
packing loop, so the two can never enumerate different tile sets."""
function _tlr_right_cross_items(OA, DB)
    qm, qk = grid_size(OA)
    items = Tuple{Int,Int,Int}[]
    @inbounds for k in 1:min(qk, ndiag_tiles(DB)), i in 1:qm
        r = _compressed_ftlr_storage_rank(OA, i, k)
        r == 0 || push!(items, (i, k, r))
    end
    sizes = [r * length(_tile_axis_range(DB, k, 2)) for (_, k, r) in items]
    return items, sizes
end

"""Mirror of [`_tlr_right_cross_items`](@ref): nonzero-execution-rank `(k,j,r)`
tiles between `DA`'s diagonal and `OB`'s off-diagonal factors, paired with
each tile's intermediate (`W = DU`) element count."""
function _tlr_left_cross_items(OB, DA)
    qk, qn = grid_size(OB)
    items = Tuple{Int,Int,Int}[]
    @inbounds for k in 1:min(qk, ndiag_tiles(DA)), j in 1:qn
        r = _compressed_ftlr_storage_rank(OB, k, j)
        r == 0 || push!(items, (k, j, r))
    end
    sizes = [length(_tile_axis_range(DA, k, 1)) * r for (k, _, r) in items]
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

function _tlr_check_cross_workspace(sizes, capacity::Int, ::Type{T}) where {T}
    isempty(sizes) && return nothing
    needed = maximum(sizes) * sizeof(T)
    capacity >= needed || throw(ArgumentError(
        "workspace has $capacity bytes; a dense-diagonal update requires at least $needed bytes"))
    return nothing
end

"""
    _tlr_pack_cross_items!(mode, arena, capacity, T, items, sizes, push_item!)

Shared arena-packing driver for the two dense-diagonal cross terms
(`O_A D_B` and `D_A O_B`). Chunks `items` (with per-item intermediate element
counts `sizes`) into passes that fit `arena`, 16-byte-aligning each
intermediate's offset within a pass, and submits each pass as one grouped
stage-1 call and one grouped stage-2 call. `push_item!(stage1, stage2, idx)`
builds and pushes item `items[idx]`'s stage-1/stage-2 `GroupedGemmTask`s,
including carving its own intermediate out of `arena` (already positioned at
that item's aligned offset when `push_item!` is called).
"""
function _tlr_pack_cross_items!(mode, arena, capacity::Int, ::Type{T}, items, sizes,
                                push_item!) where {T}
    _tlr_check_cross_workspace(sizes, capacity, T)
    first_item = 1
    while first_item <= length(items)
        _arena_reset!(arena)
        stage1 = GroupedGemmTask[]
        stage2 = GroupedGemmTask[]
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
function _tlr_offdiag_times_diag!(C, OA, DB, alpha, mode, arena, capacity::Int)
    T = eltype(OA)
    items, sizes = _tlr_right_cross_items(OA, DB)
    backend = get_backend(OA)
    _tlr_pack_cross_items!(mode, arena, capacity, T, items, sizes,
        (stage1, stage2, idx) -> begin
            i, k, r = items[idx]
            cols = _tile_axis_range(DB, k, 2)
            S = _workspace_array!(arena, backend, T, r, length(cols))
            V = compressed_ftlr_storage_inner(OA, i, k)
            D = _diag_tile_ref(DB, k)
            push!(stage1, GroupedGemmTask(
                'T', _dense_op(D), one(T), V, _dense_data(D), zero(T), S))
            U = compressed_ftlr_storage_outer(OA, i, k)
            rows = _tile_axis_range(OA, i, 1)
            push!(stage2, GroupedGemmTask(
                'N', 'N', alpha, U, S, one(alpha), view(C, rows, cols)))
        end)
    return C
end

# D_A O_B:  W_kj = D_A_kk U_B_kj  (stage 1),  C[rows,cols] += α W_kj V_B_kj'  (stage 2).
function _tlr_diag_times_offdiag!(C, DA, OB, alpha, mode, arena, capacity::Int)
    T = eltype(OB)
    items, sizes = _tlr_left_cross_items(OB, DA)
    backend = get_backend(OB)
    _tlr_pack_cross_items!(mode, arena, capacity, T, items, sizes,
        (stage1, stage2, idx) -> begin
            k, j, r = items[idx]
            rows = _tile_axis_range(DA, k, 1)
            W = _workspace_array!(arena, backend, T, length(rows), r)
            U = compressed_ftlr_storage_outer(OB, k, j)
            D = _diag_tile_ref(DA, k)
            push!(stage1, GroupedGemmTask(
                _dense_op(D), 'N', one(T), _dense_data(D), U, zero(T), W))
            V = compressed_ftlr_storage_inner(OB, k, j)
            cols = _tile_axis_range(OB, j, 2)
            push!(stage2, GroupedGemmTask(
                'N', 'T', alpha, W, V, one(alpha), view(C, rows, cols)))
        end)
    return C
end

function _tlr_diag_times_diag!(C, DA, DB, alpha, mode)
    n = min(ndiag_tiles(DA), ndiag_tiles(DB))
    tasks = GroupedGemmTask[]
    sizehint!(tasks, n)
    @inbounds for k in 1:n
        Aref = _diag_tile_ref(DA, k)
        Bref = _diag_tile_ref(DB, k)
        rows = _tile_axis_range(DA, k, 1)
        cols = _tile_axis_range(DB, k, 2)
        push!(tasks, GroupedGemmTask(
            _dense_op(Aref), _dense_op(Bref), alpha,
            _dense_data(Aref), _dense_data(Bref), one(alpha),
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
function _tlr_diag_times_dense!(C, DA, B, alpha, mode)
    tasks = GroupedGemmTask[]
    sizehint!(tasks, ndiag_tiles(DA))
    output_cols = 1:size(C, 2)
    @inbounds for k in 1:ndiag_tiles(DA)
        D = _diag_tile_ref(DA, k)
        rows = _tile_axis_range(DA, k, 1)
        inner = _tile_axis_range(DA, k, 2)
        Bdense, opB = _dense_block(B, inner, output_cols)
        push!(tasks, GroupedGemmTask(
            _dense_op(D), opB, alpha, _dense_data(D), Bdense, one(alpha),
            view(C, rows, output_cols)))
    end
    isempty(tasks) || precision_gemm_grouped!(tasks, mode)
    return C
end

"""Add the block-diagonal part of `op(A::Matrix) * op(B::TLRMatrix)` to `C`.

This is the mirror of [`_tlr_diag_times_dense!`](@ref): each diagonal tile
updates one disjoint output column block after the compressed FoldLeft pass.
"""
function _dense_times_tlr_diag!(C, A, DB, alpha, mode)
    tasks = GroupedGemmTask[]
    sizehint!(tasks, ndiag_tiles(DB))
    output_rows = 1:size(C, 1)
    @inbounds for k in 1:ndiag_tiles(DB)
        D = _diag_tile_ref(DB, k)
        inner = _tile_axis_range(DB, k, 1)
        cols = _tile_axis_range(DB, k, 2)
        Adense, opA = _dense_block(A, output_rows, inner)
        push!(tasks, GroupedGemmTask(
            opA, _dense_op(D), alpha, Adense, _dense_data(D), one(alpha),
            view(C, output_rows, cols)))
    end
    isempty(tasks) || precision_gemm_grouped!(tasks, mode)
    return C
end

function _tlr_add_diagonal_terms!(C, A::TLRMatrix, B::TLRMatrix, workspace,
                                  alpha, transA::Char, transB::Char, mode)
    _, arena, capacity = _prepare_dense_result_workspace(A, workspace)
    DA = transA == 'T' ? transpose(A) : A
    DB = transB == 'T' ? transpose(B) : B
    OA = offdiagonal(DA)
    OB = offdiagonal(DB)
    _tlr_offdiag_times_diag!(C, OA, DB, alpha, mode, arena, capacity)
    _tlr_diag_times_offdiag!(C, DA, OB, alpha, mode, arena, capacity)
    _tlr_diag_times_diag!(C, DA, DB, alpha, mode)
    return C
end
