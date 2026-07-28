# Complete TLR × dense and dense × TLR products. Each low-rank tile needs only one
# operand-typed intermediate; the reduction over TLR tiles accumulates directly in C.

function _tlr_dense_gemm!(C, A::LogicalTLROperand{<:Any,<:PaddedFTLRMatrix{<:Any,T}},
                          B::LogicalDenseOperand, alpha, beta, budget::Int,
                          compute, arena=nothing) where {T}
    _scale_output!(C, beta)
    r = maxrank(A)
    (isempty(C) || r == 0) && return C
    mt, kt = grid_size(A)
    n = size(B, 2)
    batch_width = clamp(div(budget, max(r * sizeof(T), 1)), 1, n)
    _arena_reset!(arena)
    work = _workspace_array!(arena, get_backend(A), T, r, batch_width)

    @inbounds for i in 1:mt, cols in Iterators.partition(1:n, batch_width)
        rows = _tile_axis_range(A, i, 1)
        Tview = view(work, :, 1:length(cols))
        Cview = view(C, rows, cols)
        for k in 1:kt
            inner = _tile_axis_range(A, k, 2)
            U, V = logical_tile_factors(A, i, k)
            Bd, opB = _dense_block(B, inner, cols)
            precision_gemm!('T', opB, one(T), V, Bd, zero(T), Tview, compute)
            precision_gemm!('N', 'N', alpha, U, Tview, one(alpha), Cview, compute)
        end
    end
    return C
end

function _dense_tlr_gemm!(C, A::LogicalDenseOperand,
                          B::LogicalTLROperand{<:Any,<:PaddedFTLRMatrix{<:Any,T}},
                          alpha, beta, budget::Int, compute, arena=nothing) where {T}
    _scale_output!(C, beta)
    r = maxrank(B)
    (isempty(C) || r == 0) && return C
    kt, nt = grid_size(B)
    m = size(A, 1)
    height = clamp(div(budget, max(r * sizeof(T), 1)), 1, m)
    _arena_reset!(arena)
    work = _workspace_array!(arena, get_backend(B), T, height, r)

    @inbounds for j in 1:nt, rows in Iterators.partition(1:m, height)
        cols = _tile_axis_range(B, j, 2)
        Tview = view(work, 1:length(rows), :)
        Cview = view(C, rows, cols)
        for k in 1:kt
            inner = _tile_axis_range(B, k, 1)
            Ad, opA = _dense_block(A, rows, inner)
            U, V = logical_tile_factors(B, k, j)
            precision_gemm!(opA, 'N', one(T), Ad, U, zero(T), Tview, compute)
            precision_gemm!('N', 'T', alpha, Tview, V, one(alpha), Cview, compute)
        end
    end
    return C
end
