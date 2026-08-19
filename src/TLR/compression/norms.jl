# Squared magnitude accumulated in Float64. Energy and error decisions are
# intentionally more accurate than factor storage and GEMM compute precision.
@inline abs2_f64(x::Real) = abs2(Float64(x))
@inline abs2_f64(x::Complex) =
    abs2(Float64(real(x))) + abs2(Float64(imag(x)))

@inline function norm_launch(backend, ncols::Int)
    W = unwrap(SUBGROUP_SIZE(typeof(backend)))
    nsub = ncols >= 8 ? 8 : ncols >= 4 ? 4 : ncols >= 2 ? 2 : 1
    return W, nsub, W * nsub
end

# Per-tile squared Frobenius norm for tiles addressed within a dense matrix.
@kernel function tile_norm_sq_kernel!(out::AbstractVector{Tout},
                                       A::AbstractMatrix{T},
                                       p0s,
                                       q0s,
                                       tm::Int,
                                       tn::Int,
                                       ::Val{W},
                                       ::Val{NT},
) where {Tout,T,W,NT}
    ob = @index(Group, Linear)
    tid = @index(Local, Linear)
    partial = @localmem Float64 (NT,)
    p0 = Int(@inbounds p0s[ob]) - 1
    q0 = Int(@inbounds q0s[ob]) - 1
    lane = (tid - 1) % W + 1
    sg = (tid - 1) ÷ W + 1
    nsub = NT ÷ W

    # tile accumulation
    acc = 0.0
    col = sg
    while col <= tn
        row = lane
        while row <= tm
            @inbounds acc += abs2_f64(A[p0+row, q0+col])
            row += W
        end
        col += nsub
    end
    @inbounds partial[tid] = acc

    # workgroup reduction with literal CPU-backend barriers
    @synchronize
    if NT > 512 && tid <= 512; @inbounds partial[tid] += partial[tid+512]; end
    @synchronize
    if NT > 256 && tid <= 256; @inbounds partial[tid] += partial[tid+256]; end
    @synchronize
    if NT > 128 && tid <= 128; @inbounds partial[tid] += partial[tid+128]; end
    @synchronize
    if NT >  64 && tid <=  64; @inbounds partial[tid] += partial[tid+ 64]; end
    @synchronize
    if NT >  32 && tid <=  32; @inbounds partial[tid] += partial[tid+ 32]; end
    @synchronize
    if NT >  16 && tid <=  16; @inbounds partial[tid] += partial[tid+ 16]; end
    @synchronize
    if NT >   8 && tid <=   8; @inbounds partial[tid] += partial[tid+  8]; end
    @synchronize
    if NT >   4 && tid <=   4; @inbounds partial[tid] += partial[tid+  4]; end
    @synchronize
    if NT >   2 && tid <=   2; @inbounds partial[tid] += partial[tid+  2]; end
    @synchronize
    if NT >   1 && tid == 1;   @inbounds partial[1]   += partial[2];       end
    @synchronize

    if tid == 1
        @inbounds out[ob] = Tout(partial[1])
    end
end

# Per-slab squared Frobenius norm of a packed `[m, n, batch]` array.
@kernel function tile_norm_sq_kernel!(out::AbstractVector{Tout},
                                       X::AbstractArray{T,3},
                                       ::Val{W},
                                       ::Val{NT},
) where {Tout,T,W,NT}
    ob = @index(Group, Linear)
    tid = @index(Local, Linear)
    partial = @localmem Float64 (NT,)
    tm = size(X, 1)
    tn = size(X, 2)
    lane = (tid - 1) % W + 1
    sg = (tid - 1) ÷ W + 1
    nsub = NT ÷ W

    # slab accumulation
    acc = 0.0
    col = sg
    while col <= tn
        row = lane
        while row <= tm
            @inbounds acc += abs2_f64(X[row, col, ob])
            row += W
        end
        col += nsub
    end
    @inbounds partial[tid] = acc

    # workgroup reduction with literal CPU-backend barriers
    @synchronize
    if NT > 512 && tid <= 512; @inbounds partial[tid] += partial[tid+512]; end
    @synchronize
    if NT > 256 && tid <= 256; @inbounds partial[tid] += partial[tid+256]; end
    @synchronize
    if NT > 128 && tid <= 128; @inbounds partial[tid] += partial[tid+128]; end
    @synchronize
    if NT >  64 && tid <=  64; @inbounds partial[tid] += partial[tid+ 64]; end
    @synchronize
    if NT >  32 && tid <=  32; @inbounds partial[tid] += partial[tid+ 32]; end
    @synchronize
    if NT >  16 && tid <=  16; @inbounds partial[tid] += partial[tid+ 16]; end
    @synchronize
    if NT >   8 && tid <=   8; @inbounds partial[tid] += partial[tid+  8]; end
    @synchronize
    if NT >   4 && tid <=   4; @inbounds partial[tid] += partial[tid+  4]; end
    @synchronize
    if NT >   2 && tid <=   2; @inbounds partial[tid] += partial[tid+  2]; end
    @synchronize
    if NT >   1 && tid == 1;   @inbounds partial[1]   += partial[2];       end
    @synchronize

    if tid == 1
        @inbounds out[ob] = Tout(partial[1])
    end
end

"""
    batch_frobenius_norms_sq!(out, X) -> out

Compute one squared Frobenius norm for each slab of `X`, accumulating in
`Float64` in one kernel launch.
"""
function batch_frobenius_norms_sq!(out::AbstractVector,
                                   X::AbstractArray{<:Any,3})
    count = size(X, 3)
    length(out) == count ||
        throw(DimensionMismatch("out must have one entry per batch slab"))
    count == 0 && return out

    backend = get_backend(X)
    W, _, NT = norm_launch(backend, size(X, 2))
    tile_norm_sq_kernel!(backend, NT)(
        out, X, Val{W}(), Val{NT}();
        ndrange=(NT * count,), workgroupsize=NT,
    )
    return out
end
