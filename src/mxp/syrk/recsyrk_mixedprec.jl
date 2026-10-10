# The SymmMixedPrec methods of recsyrk!. Source: vicki-development @ 694d019,
# by Vicki <vickicar@mit.edu>, where they sat in src/recsyrk.jl.

"""
    recsyrk!(alpha, A, beta, C::SymmMixedPrec) -> C

Computes `C = alpha * A * transpose(A) + beta * C` in place on the stored triangle (`C.uplo`)
of the `n × n` `SymmMixedPrec` `C`, where `A` is `n × k`. It walks `C`'s block tree: each
off-diagonal block takes one GEMM through `_syrk_dispatch!` at its own precision, and each
leaf the dense `recsyrk!`. A block stored with a scale factor `s` holds `C / s`, so its update
uses `alpha / s`; an off-diagonal block with less range than `Float32` is updated in a
`Float32` copy and stored back with its scale raised when the update took it past `floatmax`
of its type. When the first diagonal block of the top level
has more than `RECSYRK_PARALLEL_THRESHOLD` rows, the two diagonal halves run on separate threads,
so a task is spawned only when it has that much work.
"""
function recsyrk!(alpha::Number, A::AbstractMatrix, beta::Number, C::SymmMixedPrec)
    size(A, 1) == size(C, 1) || throw(DimensionMismatch("A must have $(size(C, 1)) rows, like C"))
    parallel = C.Base === nothing && size(C.A11, 1) > RECSYRK_PARALLEL_THRESHOLD
    return _recsyrk_mixed!(alpha, A, beta, C; parallel=parallel)
end

function _recsyrk_mixed!(alpha::Number, A::AbstractMatrix, beta::Number, C::SymmMixedPrec;
                        parallel::Bool)
    if C.Base !== nothing
        recsyrk!(alpha / C.Base_scale, A, beta, C.Base; uplo=C.uplo)
        return C
    end

    n1 = size(C.A11, 1)
    A1, A2 = view(A, 1:n1, :), view(A, n1+1:size(A, 1), :)
    a = alpha / C.OffDiag_scale
    C.OffDiag_scale = _update_block!(C.OffDiag, C.OffDiag_scale) do P
        C.uplo == 'L' ? _syrk_dispatch!(:GEMM, a, A2, A1, beta, P) : _syrk_dispatch!(:GEMM, a, A1, A2, beta, P)
    end

    if parallel
        @sync begin
            Threads.@spawn _recsyrk_mixed!(alpha, A1, beta, C.A11; parallel=false)
            Threads.@spawn _recsyrk_mixed!(alpha, A2, beta, C.A22; parallel=false)
        end
    else
        _recsyrk_mixed!(alpha, A1, beta, C.A11; parallel=false)
        _recsyrk_mixed!(alpha, A2, beta, C.A22; parallel=false)
    end
    return C
end
