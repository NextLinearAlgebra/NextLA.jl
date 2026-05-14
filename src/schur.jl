export schur!

function schur!(N::Integer, A::AbstractMatrix{T},
                Q::AbstractMatrix{T},
                eigvals::AbstractVector{<:Complex};
                params::Union{DeviceParams{T}, Nothing} = nothing) where {T}
end

function schur!(A::AbstractMatrix{T}) where {T}
end
