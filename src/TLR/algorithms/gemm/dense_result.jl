# Dense-output TLR GEMM: materializes a dense matrix from TLR/dense-diagonal
# operands via budgeted, region-scheduled low-rank terms. See driver.jl for
# the gemm! entry points.
include("dense_result/precision.jl")
include("dense_result/workspace.jl")
include("dense_result/axis_strategy.jl")
include("dense_result/run_schedule.jl")
include("dense_result/stages.jl")
include("dense_result/low_rank_terms.jl")
include("dense_result/regions/interior.jl")
include("dense_result/regions/corner.jl")
include("dense_result/regions/right.jl")
include("dense_result/regions/bottom.jl")
include("dense_result/dense_products.jl")
include("dense_result/driver.jl")
