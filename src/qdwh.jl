export qdwh_polar!, qdwh_eig!

struct ConvergenceFailure <: Exception
end

function qdwh_polar!(n::Integer, X::AbstractMatrix{T},
                     work::AbstractMatrix{T};
                     params::Union{DeviceParams{T}, Nothing} = nothing,
                     max_iters::Integer = 8,
                     tol::Union{Real, Nothing} = nothing) where {T}
end

function qdwh_eig!(N::Integer, A::AbstractMatrix{T},
                   eigvals::AbstractVector{<:Complex},
                   Schur::AbstractMatrix{T};
                   params::Union{DeviceParams{T}, Nothing} = nothing,
                   leaf_size::Integer = 256) where {T}
end

@kernel function form_Y_kernel!(Y, X, sqrt_c::T, n::Int) where {T}
end

@kernel function update_X_kernel!(X_next, X, Q1, Q2, a_over_c::T, scale::T, ::Val{TILE}) where {TILE, T}
end

@kernel function kappa2_check_kernel!(ok, X, n::Int)
end

@kernel function convergence_test_kernel!(ok, X_next, X, tol)
end
