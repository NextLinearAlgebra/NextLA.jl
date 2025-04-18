using NextLA
using Test
using LinearAlgebra, Random
using CUDA

#using Aqua
#@testset "Project quality" begin
#    Aqua.test_all(NextLA, ambiguities=false)
#end

include("NextLAMatrix.jl")
include("lu.jl")
include("unified_rectrxm.jl")
include("trsm.jl")

