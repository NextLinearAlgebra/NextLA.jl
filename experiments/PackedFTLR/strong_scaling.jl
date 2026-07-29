module PackedStrongScaling
using Main.DenseGemmCommon
using Main.StrongScalingExperiment
const SIZES = [1024, 2048, 4096, 8192, 16384, 32768]
const TILE_SIZE = 512
const RANKS = (64, 128)
const CASES = [MatrixCase(:padded_constant, :padded, :constant, nothing, nothing)]
run(run_config) = strong_scaling(StrongScalingConfig(SIZES, TILE_SIZE, RANKS, CASES, run_config))
end
