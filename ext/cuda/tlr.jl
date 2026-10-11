# TLR hooks for CUDA: tile-alignment and rank-multiple requirements of the
# tensor-core GEMM paths, and the batched thin SVD used by `ara_truncate!`.

@inline function NextLA.TLRmodule.validate_compressed_ftlr_tile_alignment(
    ::CUDA.CUDABackend, ::Type{T}, bm::Int, bn::Int) where {T}
    return NextLA.TLRmodule.validate_compressed_ftlr_tile_alignment_cuda(T, bm, bn)
end

@inline NextLA.TLRmodule.required_tlr_gemm_rank_multiple(
    ::CUDA.CUDABackend, ::Type{Float16}, mode) = 8
@inline NextLA.TLRmodule.required_tlr_gemm_rank_multiple(
    ::CUDA.CUDABackend, ::Type{Core.BFloat16}, mode) = 8
@inline NextLA.TLRmodule.required_tlr_gemm_rank_multiple(
    ::CUDA.CUDABackend, ::Type{Float32}, ::NextLA.TF32) = 4


# `gesvdaStridedBatched` factors the tall-skinny panel directly, so it never
# forms a Gram and never squares the condition number. Measured against
# `syrk + syevjBatched` on b=256, nb=128, fp64: 3.4× faster at s=64, 3.2× at
# s=48, and it recovers the true rank at s=16/32 where the Gram route
# over-retains by 30–40%. It is slower below s≈32, which is accepted in exchange
# for a single code path.
#
# The routine is documented as *approximate*, and its left factor is indeed not
# orthonormal when returned untruncated. `ara_truncate!` consumes the right
# factor, measured clean at ~1e-14, and truncates before use.
function NextLA.TLRmodule.batched_thin_svd!(
    A::CUDA.StridedCuArray{T,3},
) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64}}
    size(A, 1) >= size(A, 2) ||
        throw(ArgumentError("batched_thin_svd! requires size(A,1) >= size(A,2)"))
    size(A, 2) == 0 && return (
        similar(A, size(A, 1), 0, size(A, 3)),
        similar(A, real(T), 0, size(A, 3)),
        similar(A, 0, 0, size(A, 3)),
    )
    size(A, 3) == 0 && return (similar(A), similar(A, real(T), size(A, 2), 0),
                               similar(A, size(A, 2), size(A, 2), 0))
    U, S, V = CUDA.CUSOLVER.gesvda!('V', A; rank=size(A, 2))
    return (U, S, V)
end
