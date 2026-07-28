# Dense-output TLR GEMM: materializes a dense matrix from TLR/dense-diagonal
# operands via budgeted, region-scheduled low-rank terms. See driver.jl for
# the gemm! entry points. `common/axis_strategy.jl` is included separately, earlier,
# by TLRmodule.jl (operands.jl needs its GridKind before this hub runs).
include("common/precision.jl")
include("common/workspace.jl")
include("dense_result/compressed_ftlr/schedule.jl")
include("dense_result/fixed_rank/run_schedule.jl")
include("dense_result/fixed_rank/stages.jl")
include("dense_result/compressed_ftlr/stages.jl")
include("dense_result/fixed_rank/low_rank_terms.jl")
include("dense_result/compressed_ftlr/low_rank_terms.jl")
include("dense_result/fixed_rank/regions/interior.jl")
include("dense_result/fixed_rank/regions/corner.jl")
include("dense_result/fixed_rank/regions/right.jl")
include("dense_result/fixed_rank/regions/bottom.jl")
include("common/dense_products.jl")
include("dense_result/driver.jl")
