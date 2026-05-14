export hessenberg!

function hessenberg!(N::Integer, A::AbstractMatrix{T}, tau::AbstractVector{T},
                     T_mat::AbstractMatrix{T}, V_buf::AbstractMatrix{T},
                     Y_buf::AbstractMatrix{T}, work::AbstractVector{T};
                     params::Union{DeviceParams{T}, Nothing} = nothing,
                     b::Union{Integer, Nothing} = nothing) where {T}
end

function hessenberg!(A::AbstractMatrix{T}) where {T}
end

@kernel function hess_panel_kernel!(A, V, T_mat, tau, k::Int, b::Int, m::Int)
end

@kernel function apply_left_trailing_kernel!(A, V, T_mat, k::Int, b::Int, ::Val{TILE}) where {TILE}
end

@kernel function apply_right_trailing_kernel!(A, V, T_mat, k::Int, b::Int, ::Val{TILE}) where {TILE}
end

@kernel function build_Y_kernel!(Y, A, V, k::Int, b::Int)
end
