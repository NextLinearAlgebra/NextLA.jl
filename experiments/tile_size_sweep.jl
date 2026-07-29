"""Tile-size sweep experiment implementation."""
module TileSizeSweepExperiment

using Main.DenseGemmCommon

export TileSizeSweepConfig, tile_size_sweep

struct TileSizeSweepConfig{B}
    matrix_size::Int
    tile_sizes::Vector{Int}
    rank_tile_ratios::NTuple{2,Float64}
    cases::Vector{MatrixCase}
    run::RunConfig{B}
end

function TileSizeSweepConfig(matrix_size, tile_sizes, rank_tile_ratios::Tuple,
                             cases, run::RunConfig)
    length(rank_tile_ratios) == 2 ||
        throw(ArgumentError("rank_tile_ratios must contain one ratio for A and one for B"))
    ratios = Tuple(Float64.(rank_tile_ratios))
    return TileSizeSweepConfig(Int(matrix_size), Int.(tile_sizes), ratios,
                               MatrixCase[cases...], run)
end

TileSizeSweepConfig(matrix_size, tile_sizes, rank_tile_ratio::Real, cases,
                    run::RunConfig) = TileSizeSweepConfig(
    matrix_size, tile_sizes, (rank_tile_ratio, rank_tile_ratio), cases, run)

function tile_size_sweep(config::TileSizeSweepConfig; output_path=nothing)
    results = GemmResult[]
    for tile_size in config.tile_sizes
        rank_A = round(Int, config.rank_tile_ratios[1] * tile_size)
        rank_B = round(Int, config.rank_tile_ratios[2] * tile_size)
        0 < rank_A < tile_size && 0 < rank_B < tile_size ||
            throw(ArgumentError("invalid rank-to-tile ratio"))
        shape = (config.matrix_size, config.matrix_size, config.matrix_size)
        cases = MatrixCase[
            MatrixCase(c.name, c.format, c.distribution,
                       c.distribution === :constant && isnothing(c.min_rank) ?
                           min(rank_A, rank_B) : c.min_rank,
                       c.distribution === :constant && isnothing(c.max_rank) ?
                           max(rank_A, rank_B) : c.max_rank)
            for c in config.cases
        ]
        append!(results, run_cases(:tile_size_sweep, [shape], tile_size,
                                   (rank_A, rank_B), cases, config.run;
                                   square=true, output_path))
    end
    results
end

end
