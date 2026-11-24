const RECGEMM_PARALLEL_THRESHOLD = 512

"""
    recgemm!(alpha, A, B, beta, C::FullMixedPrec;
             parallel=size(C, 1) > RECGEMM_PARALLEL_THRESHOLD) -> C

Computes `C = alpha * A * B + beta * C` in place, where `C` is an `n × n` `FullMixedPrec`, `A`
is `n × k` and `B` is `k × n` for any `k`, else `DimensionMismatch`. It walks `C`'s block tree:
`A` is split by rows and `B` by columns at `C`'s split, so each quadrant of `C` takes one
product over the whole `k`, `A[1:mid, :] * B[:, 1:mid]` for `C11`. Off-diagonal blocks are
updated by a GEMM and diagonal blocks recursively. A block stored with a scale factor `s`
holds `C / s`, so its update uses `alpha / s`. An off-diagonal block with less range than
`Float32` is updated in a `Float32` copy and stored back with its scale raised when the update
took it past `floatmax` of its type; a leaf keeps its scale. With `parallel`, the four
quadrant updates of the top level run on separate threads.

# Examples
```jldoctest
julia> using NextLA.MixedPrecision, LinearAlgebra

julia> A = rand(1024, 512); B = rand(512, 1024); C = rand(1024, 1024);

julia> M = FullMixedPrec(C; precisions = DataType[Float32, Float64]);

julia> recgemm!(1.0, A, B, 0.5, M);   # M = A * B + 0.5 * M

julia> isapprox(reconstruct_matrix(M), A * B + 0.5 * C; rtol = 1e-5)
true
```
"""
function recgemm!(alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::FullMixedPrec;
                  parallel::Bool=(size(C, 1) > RECGEMM_PARALLEL_THRESHOLD))
    n = size(C, 1)
    size(A, 1) == n && size(B, 2) == n && size(A, 2) == size(B, 1) ||
        throw(DimensionMismatch("A must be $n×k and B k×$n, for C $n×$n"))

    if C.Base !== nothing
        _gemm_dispatch!(alpha / C.Base_scale, A, B, beta, C.Base)
        return C
    end

    mid = size(C.A11, 1)
    A1, A2 = view(A, 1:mid, :), view(A, mid+1:n, :)
    B1, B2 = view(B, :, 1:mid), view(B, :, mid+1:n)
    a12 = alpha / C.A12_scale
    a21 = alpha / C.A21_scale

    # Each off-diagonal block takes its product through
    # _update_mixedprec_offdiag_block!, which raises the block's scale if the
    # product took it past floatmax of its type.
    function update12()
        C.A12_scale = _update_mixedprec_offdiag_block!(C.A12, C.A12_scale) do P
            _gemm_dispatch!(a12, A1, B2, beta, P)
        end
    end
    function update21()
        C.A21_scale = _update_mixedprec_offdiag_block!(C.A21, C.A21_scale) do P
            _gemm_dispatch!(a21, A2, B1, beta, P)
        end
    end

    if parallel
        @sync begin
            Threads.@spawn recgemm!(alpha, A1, B1, beta, C.A11; parallel=false)
            Threads.@spawn update12()
            Threads.@spawn update21()
            Threads.@spawn recgemm!(alpha, A2, B2, beta, C.A22; parallel=false)
        end
    else
        recgemm!(alpha, A1, B1, beta, C.A11; parallel=false)
        update12()
        update21()
        recgemm!(alpha, A2, B2, beta, C.A22; parallel=false)
    end
    return C
end
