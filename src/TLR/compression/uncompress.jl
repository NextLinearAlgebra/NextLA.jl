export uncompress!

@kernel function _copy_diag_kernel!(A::AbstractMatrix{T},
    D::AbstractArray{T,3}, tile_m::Int, tile_n::Int) where {T}
    row, col, batch = @index(Global, NTuple)
    p0 = (batch - 1) * tile_m + 1
    q0 = (batch - 1) * tile_n + 1
    @inbounds A[p0+row-1, q0+col-1] = D[row, col, batch]
end

function _copy_diagonal_to_dense!(A::AbstractMatrix{T}, A_tlr::TLRMatrix{<:Any,T}) where {T}
    # full diagonal tiles
    n_full_diag = size(A_tlr.D, 3)
    bm, bn = nominal_tile_size(A_tlr)
    _copy_diag_kernel!(get_backend(A_tlr))(
        A, A_tlr.D, bm, bn;
        ndrange=(bm, bn, n_full_diag),
    )

    # corner diagonal tile
    if size(A_tlr.D_corner, 3) != 0
        tile_k = ndiag_tiles(A_tlr)
        tm, tn = tile_size(A_tlr, tile_k, tile_k)
        copyto!(dense_tile_view(A, A_tlr, tile_k, tile_k), view(A_tlr.D_corner, 1:tm, 1:tn, 1))
    end
    return A
end

"""
    uncompress!(A, A_tlr)

Write `A_tlr` into `A`. Rank-zero diagonal slots are reconstructed from dense
diagonal storage after the compressed off-diagonal tiles.
"""
function uncompress!(A::AbstractMatrix{T}, A_tlr::TLRMatrix{<:Any,T}) where {T}
    size(A) == size(A_tlr) ||
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

    # grouped exact-rank reconstruction
    # Exact ranks are heterogeneous, so supported backends use one grouped call.
    if supports_grouped_gemm(get_backend(A_tlr))
        T === Core.BFloat16 && !supports_bfloat16_grouped_gemm(get_backend(A_tlr)) &&
            throw(ArgumentError("CompressedFTLR BF16 grouped GEMMEx requires an NVIDIA SM80 or newer device"))
        tasks = GroupedGemmTask[]
        sizehint!(tasks, qm * qn)
        @inbounds for j in 1:qn, i in 1:qm
            compressed_ftlr_rank(A_tlr, i, j) == 0 && continue
            U, V = get_factors(A_tlr, i, j)
            push!(tasks, GroupedGemmTask(
                'N', adjoint_blas_char(T), one(T), U, V, zero(T),
                dense_tile_view(A, A_tlr, i, j)))
        end
        isempty(tasks) || precision_gemm_grouped!(tasks, mode)
        return A
    end

    # generic tile reconstruction
    @inbounds for j in 1:qn, i in 1:qm
        compressed_ftlr_rank(A_tlr, i, j) == 0 && continue
        U, V = get_factors(A_tlr, i, j)
        precision_gemm!('N', adjoint_blas_char(T), one(T), U, V, zero(T),
                        dense_tile_view(A, A_tlr, i, j), mode)
    end
    return A
end
