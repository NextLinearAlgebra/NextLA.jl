export uncompress!

@kernel function _copy_diag_kernel!(A::AbstractMatrix{T},
    D::AbstractArray{T,3}, tile_m::Int, tile_n::Int) where {T}
    row, col, b = @index(Global, NTuple)
    p0 = (b - 1) * tile_m + 1
    q0 = (b - 1) * tile_n + 1
    @inbounds A[p0+row-1, q0+col-1] = D[row, col, b]
end

function _copy_diagonal_to_dense!(A::AbstractMatrix{T}, A_tlr::TLRMatrix{<:Any,T}) where {T}
    n_full_diag = _nfull_diag_tiles(A_tlr)
    _copy_diag_kernel!(A_tlr.backend)(
        A, A_tlr.D, A_tlr.tile_m, A_tlr.tile_n;
        ndrange=(A_tlr.tile_m, A_tlr.tile_n, n_full_diag),
    )
    if size(A_tlr.D_corner, 3) != 0
        tile_k = ndiag_tiles(A_tlr)
        p0, q0 = tile_origin_coords(A_tlr, tile_k, tile_k)
        tm, tn = tile_size(A_tlr, tile_k, tile_k)
        copyto!(view(A, p0:(p0 + tm - 1), q0:(q0 + tn - 1)), view(A_tlr.D_corner, 1:tm, 1:tn, 1))
    end
    return A
end

function _uncompress_category!(
    A::AbstractMatrix{T},
    A_tlr::TLRMatrix{<:Any,T},
    obs::Vector{Int},
    U::AbstractArray{T,3},
    V::AbstractArray{T,3},
) where {T}
    isempty(obs) && return A

    U_tiles = _batch_views(U)
    V_tiles = _batch_views(V)
    A_tiles = _tile_views(A, A_tlr, obs)
    gemm_batched!('N', _adjoint_blas_char(T), one(T), U_tiles, V_tiles, zero(T), A_tiles)
    return A
end

"""
    uncompress!(A, A_tlr)

Write the dense matrix represented by `A_tlr` into `A` in-place.

Diagonal tiles are copied from the packed diagonal storage and off-diagonal
tiles are reconstructed category-by-category with batched GEMMs.
"""
function uncompress!(A::AbstractMatrix{T}, A_tlr::TLRMatrix{<:Any,T}) where {T}
    size(A, 1) == A_tlr.m && size(A, 2) == A_tlr.n ||
        throw(DimensionMismatch("A dimensions must match A_tlr"))
    A_tlr.m == A_tlr.n && A_tlr.tile_m == A_tlr.tile_n ||
        throw(ArgumentError("uncompress! currently requires square matrices with square tiles"))

    _copy_diagonal_to_dense!(A, A_tlr)

    _uncompress_category!(A, A_tlr, A_tlr.obs_int, A_tlr.int_U, A_tlr.int_V)
    _uncompress_category!(A, A_tlr, A_tlr.obs_right, A_tlr.right_U, A_tlr.right_V)
    _uncompress_category!(A, A_tlr, A_tlr.obs_bottom, A_tlr.bottom_U, A_tlr.bottom_V)

    return A
end
