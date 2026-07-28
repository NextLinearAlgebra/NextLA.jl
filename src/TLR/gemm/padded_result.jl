# Padded-output GEMM: produces a fixed-capacity PaddedFTLR matrix via ARA sampling
# over the implicit factor-list operators. See driver.jl for the gemm! entry
# point.
include("padded_result/workspace.jl")
include("padded_result/single_tile_coupling.jl")
include("padded_result/run_coupling.jl")
include("padded_result/single_tile_sampling.jl")
include("padded_result/rolling_schedule.jl")
include("padded_result/driver.jl")
