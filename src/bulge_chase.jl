export bulge_chase!

struct BulgeChaseConvergenceFailure <: Exception
end

function bulge_chase!(N::Integer, H::AbstractMatrix{T}, Q::AbstractMatrix{T};
                      params::Union{DeviceParams{T}, Nothing} = nothing,
                      k::Union{Integer, Nothing} = nothing,
                      max_sweeps::Integer = 30 * N) where {T}
end

function bulge_chase!(H::AbstractMatrix{T}) where {T}
end

@kernel function bulge_window_kernel!(G_accum, H, j_window::Int, k::Int)
end

@kernel function apply_window_left_kernel!(H, G_accum, j_window::Int, k::Int, ::Val{TILE}) where {TILE}
end

@kernel function apply_window_right_kernel!(H, G_accum, j_window::Int, k::Int, ::Val{TILE}) where {TILE}
end

@kernel function bulge_shift_select_kernel!(shifts, H, n_tail::Int, k::Int)
end
