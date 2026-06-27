module NextLAMetalExt

using NextLA
using Metal
using LinearAlgebra

include("metal/common.jl")
include("metal/gemm.jl")
include("metal/syrk.jl")
include("metal/trsm.jl")
include("metal/potrf.jl")

end
