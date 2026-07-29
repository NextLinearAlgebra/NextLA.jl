"""Matrix-shape sweep experiment implementation."""
module MatrixShapeSweepExperiment

using Main.DenseGemmCommon

export MatrixShapeSweepConfig, matrix_shape_sweep

struct MatrixShapeSweepConfig{B}
    base_size::Int
    tile_size::Int
    rank::Int
    ratios::Vector{NTuple{3,Float64}}
    cases::Vector{MatrixCase}
    run::RunConfig{B}
end

function MatrixShapeSweepConfig(base_size, tile_size, rank, ratios, cases, run::RunConfig)
    return MatrixShapeSweepConfig(Int(base_size), Int(tile_size), Int(rank),
        [Tuple(Float64.(x)) for x in ratios], MatrixCase[cases...], run)
end

function matrix_shape_sweep(config::MatrixShapeSweepConfig)
    shapes = NTuple{3,Int}[]
    for ratio in config.ratios
        length(ratio) == 3 || throw(ArgumentError("shape ratios must have length three"))
        scale = config.base_size / cbrt(prod(ratio))
        push!(shapes, ntuple(i -> max(config.tile_size,
            round(Int, scale * ratio[i] / config.tile_size) * config.tile_size), 3))
    end
    cases = MatrixCase[
        MatrixCase(c.name, c.format, c.distribution,
                   c.distribution === :constant ? config.rank : c.min_rank,
                   c.distribution === :constant ? config.rank : c.max_rank)
        for c in config.cases
    ]
    run_cases(:matrix_shape_sweep, shapes, config.tile_size,
              (config.rank, config.rank), cases, config.run; square=false)
end

end
