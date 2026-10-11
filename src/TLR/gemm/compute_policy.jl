# TLR compute defaults; generic compute-mode machinery lives in NextLA.
# Float16, BFloat16, Float32, and Float64 are supported, with low-precision
# operands accumulated in Float32.

@inline default_gemm_compute_mode(::Type{Float16}) = GEMMCompute{Float32}()
@inline default_gemm_compute_mode(::Type{Core.BFloat16}) = GEMMCompute{Float32}()
@inline default_gemm_compute_mode(::Type{Float32}) = GEMMCompute{Float32}()
@inline default_gemm_compute_mode(::Type{Float64}) = GEMMCompute{Float64}()

function default_gemm_compute_mode(::Type{T}) where {T}
    throw(ArgumentError("TLR GEMM does not support operand type $T; supported operand types are Float16, BFloat16, Float32, and Float64"))
end

function validate_tlr_gemm_precision(backend, ::Type{Tin}, ::Type{Tout}, mode) where {Tin,Tout}
    Tin in (Float16, Core.BFloat16, Float32, Float64) ||
        throw(ArgumentError("TLR GEMM operand type must be Float16, BFloat16, Float32, or Float64; got $Tin"))
    validate_gemm_signature(backend, Tin, Tin, Tin, mode)   # Stage 1/2 workspace
    validate_gemm_signature(backend, Tin, Tin, Tout, mode)  # Stage 3 destination
    return nothing
end

# Backends opt in when a compute mode requires rank-aligned tensor-core panels.
@inline required_tlr_gemm_rank_multiple(backend, ::Type, mode) = 1

# Backends also opt in when grouped tensor-core kernels require aligned tile
# start addresses. Construction stays backend-agnostic; GEMM owns this error.
@inline validate_compressed_ftlr_tile_alignment(
    backend, ::Type, bm::Int, bn::Int) = nothing

function validate_compressed_ftlr_tile_alignment_cuda(
    ::Type{T}, bm::Int, bn::Int) where {T}
    q = gemm_alignment_quantum(T)
    (bm % q == 0 && bn % q == 0) || throw(ArgumentError(
        "CompressedFTLR nominal tile size ($bm, $bn) is not 16-byte aligned for $T: " *
        "both extents must be multiples of $q; use " *
        "($(cld(bm, q) * q), $(cld(bn, q) * q)) instead"))
    return nothing
end

function validate_tlr_gemm_storage(A, mode; name::AbstractString="operand")
    X = A isa TransposeTLRMatrix ? parent(A) : A
    X = X isa TLRMatrix ? offdiagonal(X) : X
    backend = get_backend(X)
    T = eltype(X)
    bm, bn = nominal_tile_size(X)

    # tile alignment
    validate_compressed_ftlr_tile_alignment(backend, T, bm, bn)
    q = required_tlr_gemm_rank_multiple(backend, T, mode)
    q <= 1 && return nothing

    # stored-rank alignment
    qm, qn = grid_size(X)
    @inbounds for j in 1:qn, i in 1:qm
        width = compressed_ftlr_storage_rank(X, i, j)
        (iszero(width) || iszero(width % q)) || throw(ArgumentError(
            "$name stores tile ($i, $j) at rank width $width, but this GEMM " *
            "precision requires widths divisible by $q; construct it with " *
            "rank_multiple=$q (or a multiple of it)"))
    end
    return nothing
end

"""Scale a dense destination once before product terms are accumulated."""
@inline function scale_output!(C, beta)
    T = eltype(C)
    if iszero(beta)
        fill!(C, zero(T))
    elseif !isone(beta)
        C .*= T(beta)
    end
    return C
end
