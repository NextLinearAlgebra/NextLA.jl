const RECGEMM_PARALLEL_THRESHOLD = 512

"""
    recgemm!(alpha, A, B, beta, C::FullMixedPrec; parallel=size(C, 1) > RECGEMM_PARALLEL_THRESHOLD)

Computes `C = alpha * A * B + beta * C` in place and returns `C`, where `C` is an `n × n`
`FullMixedPrec`, `A` is `n × k` and `B` is `k × n`. It walks `C`'s block tree as
`vicki-development` did: `A` and `B` are split into quadrants at `C`'s split, and each quadrant
of `C` takes two products, `A11 * B11 + A12 * B21` for `C11`, the first with `beta` and the
second added to it. Off-diagonal blocks are updated through `_gemm_dispatch!` at their own
precision and diagonal blocks recursively. Where `C` is split, `A` and `B` must therefore be
square, `n × n`, else `DimensionMismatch`; a leaf of `C` takes any `k`. A block stored with a
scale factor `s` holds `C / s`, so its update uses `alpha / s`. An off-diagonal block
with less range than `Float32` is updated in a `Float32` copy and stored back with its scale
raised when the update took it past `floatmax` of its type; a leaf keeps its scale.
With `parallel`, the four quadrant updates of the top level run on separate threads.
"""
function recgemm!(alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::FullMixedPrec;
                  parallel::Bool=(size(C, 1) > RECGEMM_PARALLEL_THRESHOLD))
    n = size(C, 1)
    size(A, 1) == n && size(B, 2) == n && size(A, 2) == size(B, 1) ||
        throw(DimensionMismatch("A must be $n×k and B k×$n, for C $n×$n"))
    return _recgemm_mixed!(alpha, A, B, beta, C; parallel=parallel)
end

function _recgemm_mixed!(alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::FullMixedPrec; parallel::Bool)
    if C.Base !== nothing
        _gemm_dispatch!(alpha / C.Base_scale, A, B, beta, C.Base)
        return C
    end

    n, mid = size(C, 1), size(C.A11, 1)
    size(A, 2) == n ||
        throw(DimensionMismatch("recgemm!: where C is split, A and B must be $n×$n, got k = $(size(A, 2))"))
    A11, A12 = view(A, 1:mid, 1:mid), view(A, 1:mid, mid+1:n)
    A21, A22 = view(A, mid+1:n, 1:mid), view(A, mid+1:n, mid+1:n)
    B11, B12 = view(B, 1:mid, 1:mid), view(B, 1:mid, mid+1:n)
    B21, B22 = view(B, mid+1:n, 1:mid), view(B, mid+1:n, mid+1:n)
    a12 = alpha / C.A12_scale
    a21 = alpha / C.A21_scale
    one_ = one(beta)

    # Each off-diagonal block takes its two products through _update_block!, which
    # raises the block's scale if they took it past floatmax of its type.
    function update12()
        C.A12_scale = _update_block!(C.A12, C.A12_scale) do P
            _gemm_dispatch!(a12, A11, B12, beta, P)
            _gemm_dispatch!(a12, A12, B22, one_, P)
        end
    end
    function update21()
        C.A21_scale = _update_block!(C.A21, C.A21_scale) do P
            _gemm_dispatch!(a21, A21, B11, beta, P)
            _gemm_dispatch!(a21, A22, B21, one_, P)
        end
    end

    if parallel
        @sync begin
            Threads.@spawn begin
                _recgemm_mixed!(alpha, A11, B11, beta, C.A11; parallel=false)
                _recgemm_mixed!(alpha, A12, B21, one_, C.A11; parallel=false)
            end
            Threads.@spawn update12()
            Threads.@spawn update21()
            Threads.@spawn begin
                _recgemm_mixed!(alpha, A21, B12, beta, C.A22; parallel=false)
                _recgemm_mixed!(alpha, A22, B22, one_, C.A22; parallel=false)
            end
        end
    else
        _recgemm_mixed!(alpha, A11, B11, beta, C.A11; parallel=false)
        _recgemm_mixed!(alpha, A12, B21, one_, C.A11; parallel=false)
        update12()
        update21()
        _recgemm_mixed!(alpha, A21, B12, beta, C.A22; parallel=false)
        _recgemm_mixed!(alpha, A22, B22, one_, C.A22; parallel=false)
    end
    return C
end
