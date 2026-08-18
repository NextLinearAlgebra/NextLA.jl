# Compressed-output GEMM: discovers output ranks with ARA in private fixed-width
# staging, then packs and returns a finalized `CompressedFTLRMatrix`.
include("compressed_result/workspace.jl")
include("compressed_result/run_coupling.jl")
include("compressed_result/rolling_schedule.jl")
include("compressed_result/driver.jl")
