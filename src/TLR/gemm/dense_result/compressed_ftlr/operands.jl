# Exact-rank factor access under logical N/T. A transpose swaps factor roles,
# tile coordinates, and the two complementary packing orders.
@inline _compressed_ftlr_logical_coords(
    ::LogicalTLROperand{:N,<:CompressedFTLRMatrix}, i::Int, j::Int) = (i, j)
@inline _compressed_ftlr_logical_coords(
    ::LogicalTLROperand{:T,<:CompressedFTLRMatrix}, i::Int, j::Int) = (j, i)

@inline _compressed_ftlr_rank(
    A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    _compressed_ftlr_rank(physical(A), i, j)
@inline _compressed_ftlr_rank(
    A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    _compressed_ftlr_rank(physical(A), j, i)
@inline _compressed_ftlr_execution_rank(
    A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    _compressed_ftlr_execution_rank(physical(A), i, j)
@inline _compressed_ftlr_execution_rank(
    A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    _compressed_ftlr_execution_rank(physical(A), j, i)

@inline compressed_ftlr_outer(
    A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_outer(physical(A), i, j)
@inline compressed_ftlr_outer(
    A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_inner(physical(A), j, i)
@inline compressed_ftlr_inner(
    A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_inner(physical(A), i, j)
@inline compressed_ftlr_inner(
    A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_outer(physical(A), j, i)
@inline compressed_ftlr_execution_outer(
    A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_execution_outer(physical(A), i, j)
@inline compressed_ftlr_execution_outer(
    A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_execution_inner(physical(A), j, i)
@inline compressed_ftlr_execution_inner(
    A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_execution_inner(physical(A), i, j)
@inline compressed_ftlr_execution_inner(
    A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_execution_outer(physical(A), j, i)

@inline logical_tile_factors(
    A::LogicalTLROperand{<:Any,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    (compressed_ftlr_outer(A, i, j), compressed_ftlr_inner(A, i, j))

@inline compressed_ftlr_outer_order(
    A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}) =
    compressed_ftlr_outer_order(physical(A))
@inline compressed_ftlr_outer_order(
    A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}) =
    _transpose_order(compressed_ftlr_inner_order(physical(A)))
@inline compressed_ftlr_inner_order(
    A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}) =
    compressed_ftlr_inner_order(physical(A))
@inline compressed_ftlr_inner_order(
    A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}) =
    _transpose_order(compressed_ftlr_outer_order(physical(A)))

@inline _compressed_ftlr_outer_storage(A::CompressedFTLRMatrix) = A.outer
@inline _compressed_ftlr_inner_storage(A::CompressedFTLRMatrix) = A.inner
@inline _compressed_ftlr_parent(A::CompressedFTLRMatrix) = A
@inline _compressed_ftlr_outer_storage(
    A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}) = physical(A).outer
@inline _compressed_ftlr_outer_storage(
    A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}) = physical(A).inner
@inline _compressed_ftlr_inner_storage(
    A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}) = physical(A).inner
@inline _compressed_ftlr_inner_storage(
    A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}) = physical(A).outer
@inline _compressed_ftlr_parent(
    A::LogicalTLROperand{<:Any,<:CompressedFTLRMatrix}) = physical(A)
