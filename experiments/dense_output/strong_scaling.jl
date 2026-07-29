module DenseOutputStrongScaling
using Main.DenseGemmCommon
using Main.StrongScalingExperiment

const SIZES = [2048, 4096, 8192, 16384, 32768, 65536]
const TILE_SIZE = 512
const RANKS = (64, 64)
const CASES = [
    MatrixCase(:padded_constant, :padded, :constant, 64, 64),
    MatrixCase(:compressed_constant, :compressed, :constant, 64, 64),
    MatrixCase(:compressed_uniform, :compressed, :uniform, 16, 64),
    MatrixCase(:compressed_skewed, :compressed, :skewed, 16, 64),
]

run(run_config, output_path) = strong_scaling(StrongScalingConfig(
    SIZES, TILE_SIZE, RANKS, CASES, run_config); output_path)
end
