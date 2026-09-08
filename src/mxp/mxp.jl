"""
    MixedPrecision

Mixed-precision matrix containers and the routines that work over them.
Not re-exported by NextLA; opt in with `using NextLA.MixedPrecision`.
"""
module MixedPrecision

using LinearAlgebra

import ..NextLA: recgemm!

using ..NextLA: _gemm_dispatch!, _rec_split, unified_rec

abstract type AbstractMixedPrec{T} <: AbstractMatrix{T} end

include("transposedmixedprec.jl")
include("blocks.jl")
include("fullmixedprec.jl")
include("symmmixedprec.jl")
include("trimixedprec.jl")
include("tiledtrimixedprec.jl")
include("adaptive.jl")
include("quantize.jl")

include("gemm/recgemm_mixedprec.jl")
include("trsm/rectrxm_mixed.jl")

export FullMixedPrec, SymmMixedPrec, TriMixedPrec, TiledTriMixedPrec
export reconstruct_matrix, adaptive_precisions, adaptive_precision_LT
export dequantize, quantize, unified_rec_mixed

end
