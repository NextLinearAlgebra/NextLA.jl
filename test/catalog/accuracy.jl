# Generic CPU/GPU utilities, kept independent of the solver's reconstruction.
using LinearAlgebra, Random, SparseArrays

block_scale(scale) = scale === nothing ? 1.0 : Float64(scale)

function decode_blocks!(dest, tree)
    if tree.BaseCase !== nothing
        dest .= Float64.(tree.BaseCase) .* block_scale(tree.base_scale)
        return dest
    end
    mid = size(tree.A11, 1)
    n = size(tree, 1)
    decode_blocks!(view(dest, 1:mid, 1:mid), tree.A11)
    decode_blocks!(view(dest, mid+1:n, mid+1:n), tree.A22)
    if tree.uplo == 'L'
        view(dest, mid+1:n, 1:mid) .= Float64.(tree.OffDiag) .* block_scale(tree.offDiag_scale)
        view(dest, 1:mid, mid+1:n) .= transpose(view(dest, mid+1:n, 1:mid))
    else
        view(dest, 1:mid, mid+1:n) .= Float64.(tree.OffDiag) .* block_scale(tree.offDiag_scale)
        view(dest, mid+1:n, 1:mid) .= transpose(view(dest, 1:mid, mid+1:n))
    end
    dest
end

function decode_blocks(tree)
    block = tree.BaseCase === nothing ? tree.OffDiag : tree.BaseCase
    decode_blocks!(similar(block, Float64, size(tree)...), tree)
end

function scale_stats(tree)
    if tree.BaseCase !== nothing
        scale = block_scale(tree.base_scale)
        return (Int(scale != 1), scale)
    end
    left, right = scale_stats(tree.A11), scale_stats(tree.A22)
    scale = block_scale(tree.offDiag_scale)
    (left[1] + right[1] + Int(scale != 1), max(left[2], right[2], scale))
end

function input_diagnostics(A)
    values = nonzeros(A)
    all(isfinite, values) || error("Input contains NaN or Inf")
    diagonal = diag(A)
    off = vec(sum(abs, A; dims=2)) .- abs.(diagonal)
    margin = diagonal .- off
    Dict{String,Any}("nnz" => nnz(A), "density" => nnz(A) / Float64(size(A, 1))^2,
        "max_abs" => maximum(abs, values; init=0.0),
        "min_abs_nonzero" => minimum(x -> iszero(x) ? Inf : abs(x), values; init=Inf),
        "dominant_fraction" => count(>(0), margin) / length(margin),
        "worst_margin" => minimum(margin))
end

function accuracy_inputs(A, config)
    rng = MersenneTwister(config["seed"])
    omega = randn(rng, size(A, 1), config["probes"])
    truth = randn(rng, size(A, 1), config["nrhs"])
    (omega=omega, aomega=A * omega, truth=truth, rhs=A * truth, anorm=norm(A))
end

function solution_metrics(A, solution, inputs)
    residual = A * solution - inputs.rhs
    forward = maximum(norm(view(solution, :, j) - view(inputs.truth, :, j)) /
        norm(view(inputs.truth, :, j)) for j in axes(solution, 2))
    solve = maximum(norm(view(residual, :, j)) / norm(view(inputs.rhs, :, j)) for j in axes(solution, 2))
    backward = maximum(norm(view(residual, :, j)) /
        (inputs.anorm * norm(view(solution, :, j)) + norm(view(inputs.rhs, :, j))) for j in axes(solution, 2))
    Dict{String,Any}("forward_error" => forward, "solve_residual" => solve, "backward_error" => backward)
end

function probe_error(action, inputs)
    norm(action - inputs.aomega) / norm(inputs.aomega)
end
