# Canonical TLR-output GEMM: produces a compressed TLR matrix via ARA sampling
# over the implicit factor-list operators. See driver.jl for the gemm! entry
# point.
include("tlr_result/workspace.jl")
include("tlr_result/single_tile_coupling.jl")
include("tlr_result/run_coupling.jl")
include("tlr_result/single_tile_sampling.jl")
include("tlr_result/run_sampling.jl")
include("tlr_result/driver.jl")
