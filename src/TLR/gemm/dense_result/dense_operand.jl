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

@inline function _diag_tile_ref(A::TLRMatrix, k::Int)
    tile = _diag_tile_view(A, k)
    return LogicalDenseTile{:N,typeof(tile)}(tile)
end

@inline function _diag_tile_ref(
    A::TransposeTLRMatrix{<:Any,<:TLRMatrix}, k::Int)
    tile = _diag_tile_view(parent(A), k)
    return LogicalDenseTile{:T,typeof(tile)}(tile)
end
