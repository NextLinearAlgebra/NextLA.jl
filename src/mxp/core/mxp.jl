"""
    MixedPrecision

Mixed-precision matrix containers and the routines that work over them.
Not re-exported by NextLA; opt in with `using NextLA.MixedPrecision`.
"""
module MixedPrecision

using LinearAlgebra: norm, tril!, triu!, Transpose, Adjoint

import ..NextLA: recgemm!, recsyrk!, potrf_recursive!, lu_recursive!, unified_rectrxm!

using ..NextLA: _gemm_dispatch!, _syrk_dispatch!, _rec_split, unified_rec, LU_BLOCK_SIZE, POTRF_BLOCK_SIZE, RECSYRK_PARALLEL_THRESHOLD, BFloat16

abstract type AbstractMixedPrec{T} <: AbstractMatrix{T} end

include("transposedmixedprec.jl")
include("blocks.jl")
include("fullmixedprec.jl")
include("symmmixedprec.jl")
include("trimixedprec.jl")
include("tiledtrimixedprec.jl")
include("adaptive.jl")
include("quantize.jl")

include("../blas/gemm/recgemm_mixedprec.jl")
include("../blas/trsm/rectrxm_mixed.jl")
include("../lapack/lu/reclu_mixedprec.jl")
include("../blas/syrk/recsyrk_mixedprec.jl")
include("../lapack/cholesky/recpotrf_mixedprec.jl")

export FullMixedPrec, SymmMixedPrec, TriMixedPrec, TiledTriMixedPrec
export reconstruct_matrix, adaptive_precisions
export quantize, dequantize
export recgemm!, recsyrk!, potrf_recursive!, lu_recursive!, unified_rectrxm!

end
