# Operand adapters used only by GEMMs whose destination is dense.

"""Zero-copy logical `N/T` view of a standalone dense GEMM operand."""
struct LogicalDenseOperand{Op,A<:AbstractMatrix}
    data::A
end

@inline logical_dense_operand(A::AbstractMatrix, op::Char='N') =
    _logical_dense_operand(A, Val(_normalize_tlr_op(op)))
@inline _logical_dense_operand(A, ::Val{'N'}) = LogicalDenseOperand{:N,typeof(A)}(A)
@inline _logical_dense_operand(A, ::Val{'T'}) = LogicalDenseOperand{:T,typeof(A)}(A)
Base.eltype(A::LogicalDenseOperand) = eltype(A.data)
Base.size(A::LogicalDenseOperand{:N}) = size(A.data)
Base.size(A::LogicalDenseOperand{:T}) = reverse(size(A.data))
Base.size(A::LogicalDenseOperand, d::Int) = size(A)[d]
@inline KernelAbstractions.get_backend(A::LogicalDenseOperand) = get_backend(A.data)

"""Physical dense block and BLAS operation representing logical `rows × cols`."""
@inline _dense_block(A::LogicalDenseOperand{:N}, rows, cols) =
    (view(A.data, rows, cols), 'N')
@inline _dense_block(A::LogicalDenseOperand{:T}, rows, cols) =
    (view(A.data, cols, rows), 'T')

"""A physical dense tile together with the operation needed to make it logical."""
struct LogicalDenseTile{Op,A<:AbstractMatrix}
    data::A
end
@inline _dense_data(A::LogicalDenseTile) = A.data
@inline _dense_op(::LogicalDenseTile{:N}) = 'N'
@inline _dense_op(::LogicalDenseTile{:T}) = 'T'

@inline function _diag_tile_ref(
    A::LogicalTLROperand{Op,<:TLRMatrix}, k::Int) where {Op}
    tile = _diag_tile_view(physical(A), k)
    return LogicalDenseTile{Op,typeof(tile)}(tile)
end
@inline ndiag_tiles(A::LogicalTLROperand{<:Any,<:TLRMatrix}) =
    ndiag_tiles(physical(A))

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
