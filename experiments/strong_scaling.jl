"""Strong-scaling experiment implementation."""
module StrongScalingExperiment

if !isdefined(Main, :DenseGemmCommon)
    include(joinpath(@__DIR__, "common.jl"))
end
using Main.DenseGemmCommon

export StrongScalingConfig, strong_scaling

struct StrongScalingConfig{B}
    sizes::Vector{NTuple{3,Int}}
    tile_size::Int
    ranks::NTuple{2,Int}
    cases::Vector{MatrixCase}
    run::RunConfig{B}
end

function StrongScalingConfig(sizes, tile_size, ranks, cases, run::RunConfig)
    shapes = sizes isa AbstractVector{<:Integer} ?
        [(Int(s), Int(s), Int(s)) for s in sizes] : [Tuple(Int.(x)) for x in sizes]
    return StrongScalingConfig(shapes, Int(tile_size), Int.(ranks),
                               MatrixCase[cases...], run)
end

function strong_scaling(config::StrongScalingConfig)
    run_cases(:strong_scaling, config.sizes, config.tile_size, config.ranks,
              config.cases, config.run; square=true)
end

end
