module NextLAoneAPIExt

using NextLA
using oneAPI
using LinearAlgebra

include("oneapi/common.jl")
include("oneapi/gemm.jl")
include("oneapi/syrk.jl")
include("oneapi/trsm.jl")

end
