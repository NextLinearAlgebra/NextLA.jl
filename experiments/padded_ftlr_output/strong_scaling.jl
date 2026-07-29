module PaddedFTLROutputStrongScaling

using Main.PaddedFTLROutputExperiment

const SIZES = [2048, 4096, 8192, 16384, 32768, 65536]
const TILE_SIZE = 512
const RANKS = (64, 128)
const OUTPUT_RANK = 128

run(run_config) = padded_ftlr_output_strong_scaling(PaddedFTLROutputStrongScalingConfig(
    SIZES, TILE_SIZE, RANKS, OUTPUT_RANK, run_config))

end
