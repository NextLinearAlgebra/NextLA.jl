export uncompress!

@kernel function _copy_diag_kernel!(A::AbstractMatrix{T},
    D::AbstractArray{T,3}, tile_m::Int, tile_n::Int) where {T}
    row, col, batch = @index(Global, NTuple)
    p0 = (batch - 1) * tile_m + 1
    q0 = (batch - 1) * tile_n + 1
    @inbounds A[p0+row-1, q0+col-1] = D[row, col, batch]
end

function _copy_diagonal_to_dense!(A::AbstractMatrix{T}, A_tlr::TLRMatrix{<:Any,T}) where {T}
    n_full_diag = _nfull_diag_tiles(A_tlr)
    bm, bn = nominal_tile_size(A_tlr)
    _copy_diag_kernel!(A_tlr.backend)(
        A, A_tlr.D, bm, bn;
        ndrange=(bm, bn, n_full_diag),
    )
    if size(A_tlr.D_corner, 3) != 0
        tile_k = ndiag_tiles(A_tlr)
        tm, tn = tile_size(A_tlr, tile_k, tile_k)
        copyto!(_dense_tile_view(A, A_tlr, tile_k, tile_k), view(A_tlr.D_corner, 1:tm, 1:tn, 1))
    end
    return A
end

"""
    uncompress!(A, A_tlr)

Write the dense matrix represented by `A_tlr` into `A` in-place.

The full-grid compressed off-diagonal part is reconstructed first; its rank-zero
diagonal slots leave zeros that are then overwritten from dense diagonal storage.
"""
function uncompress!(A::AbstractMatrix{T}, A_tlr::TLRMatrix{<:Any,T}) where {T}
    size(A, 1) == A_tlr.m && size(A, 2) == A_tlr.n ||
        throw(DimensionMismatch("A dimensions must match A_tlr"))
    uncompress!(A, offdiagonal(A_tlr))
    _copy_diagonal_to_dense!(A, A_tlr)
    return A
end

"""Write an exact-rank CompressedFTLR matrix into dense storage."""
function uncompress!(A::AbstractMatrix{T}, A_tlr::CompressedFTLRMatrix{<:Any,T}) where {T}
    size(A) == size(A_tlr) || throw(DimensionMismatch("A dimensions must match A_tlr"))
    fill!(A, zero(T))
    qm, qn = grid_size(A_tlr)
    mode = default_gemm_compute_mode(T)
    # Exact ranks make the tile reconstructions heterogeneous. On CUDA submit
    # their factor/destination views through GEMMGroupedBatchedEx in one call;
    # the generic path remains an explicit CPU-compatible tile loop.
    if supports_grouped_gemm(get_backend(A_tlr))
        T === Core.BFloat16 && !supports_bfloat16_grouped_gemm(get_backend(A_tlr)) &&
            throw(ArgumentError("CompressedFTLR BF16 grouped GEMMEx requires an NVIDIA SM80 or newer device"))
        tasks = GroupedGemmTask[]
        sizehint!(tasks, qm * qn)
        @inbounds for j in 1:qn, i in 1:qm
            _compressed_ftlr_rank(A_tlr, i, j) == 0 && continue
            U, V = get_factors(A_tlr, i, j)
            push!(tasks, GroupedGemmTask(
                'N', _adjoint_blas_char(T), one(T), U, V, zero(T),
                _dense_tile_view(A, A_tlr, i, j)))
        end
        isempty(tasks) || precision_gemm_grouped!(tasks, mode)
        return A
    end
    @inbounds for j in 1:qn, i in 1:qm
        _compressed_ftlr_rank(A_tlr, i, j) == 0 && continue
        U, V = get_factors(A_tlr, i, j)
        precision_gemm!('N', _adjoint_blas_char(T), one(T), U, V, zero(T),
                        _dense_tile_view(A, A_tlr, i, j), mode)
    end
    return A
end
