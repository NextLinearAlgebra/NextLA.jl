# Dense-output GEMM for exact packed factors and dense-diagonal TLR operands.
include("common/precision.jl")
include("common/workspace.jl")
include("dense_result/compressed_ftlr/rank_metadata.jl")
include("dense_result/compressed_ftlr/fold_cost.jl")
include("dense_result/compressed_ftlr/schedule.jl")
include("dense_result/compressed_ftlr/schedule_dp.jl")   # TEMPORARY: benchmark-only, see file header
include("dense_result/compressed_ftlr/execute.jl")
include("dense_result/compressed_ftlr/mixed_dense.jl")
include("dense_result/compressed_ftlr/low_rank_terms.jl")
include("dense_result/compressed_ftlr/analysis.jl")
include("dense_result/tlr_diagonal.jl")
include("common/dense_products.jl")
include("dense_result/driver.jl")
