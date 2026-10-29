# Isolate the branch's standalone source files from the benchmark's imports.
# No package-wide NextLA include: it also loads unrelated development code.
module CatalogSolver
using LinearAlgebra, CUDA, KernelAbstractions
const AnyGPUArray = CUDA.AnyCuArray
const SRC = normpath(joinpath(@__DIR__, "..", "..", "src"))
include(joinpath(SRC, "fullmixedprec.jl"))
include(joinpath(SRC, "cholesky_tree_subdiv.jl"))
end
