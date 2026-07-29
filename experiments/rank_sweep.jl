"""Rank-sweep experiment implementation."""
module RankSweepExperiment

if !isdefined(Main, :DenseGemmCommon)
    include(joinpath(@__DIR__, "common.jl"))
end
using Main.DenseGemmCommon

export RankSweepConfig, rank_sweep

struct RankSweepConfig{B}
    matrix_size::Int
    tile_size::Int
    ranks::Vector{Int}
    cases::Vector{MatrixCase}
    run::RunConfig{B}
end

RankSweepConfig(matrix_size, tile_size, ranks, cases, run::RunConfig) =
    RankSweepConfig(Int(matrix_size), Int(tile_size), Int.(ranks),
                    MatrixCase[cases...], run)

function rank_sweep(config::RankSweepConfig)
    results = GemmResult[]
    for rank in config.ranks
        rank < config.tile_size || throw(ArgumentError("rank must be smaller than tile_size"))
        shape = (config.matrix_size, config.matrix_size, config.matrix_size)
        append!(results, run_cases(:rank_sweep, [shape], config.tile_size,
                                   (rank, rank), config.cases, config.run; square=true))
    end
    results
end

end
