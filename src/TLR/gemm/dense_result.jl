# Dense-output TLR GEMM: materializes a dense matrix from TLR/dense-diagonal
# operands via budgeted, region-scheduled low-rank terms. See driver.jl for
# the gemm! entry points. `axis_strategy.jl` is included separately, earlier,
# by TLRmodule.jl (operands.jl needs its GridKind before this hub runs).
include("dense_result/precision.jl")
include("dense_result/workspace.jl")
include("dense_result/run_schedule.jl")
include("dense_result/stages.jl")
include("dense_result/low_rank_terms.jl")
include("dense_result/regions/interior.jl")
include("dense_result/regions/corner.jl")
include("dense_result/regions/right.jl")
include("dense_result/regions/bottom.jl")
include("dense_result/dense_products.jl")
include("dense_result/driver.jl")
