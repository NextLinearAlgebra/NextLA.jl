# tiled mixed precision matrix

# tile data structure that stores the information about where each tile 
Tile{T_Diag, T_OffDiag} = Union{AbstractMatrix{T_Diag}, AbstractMatrix{T_OffDiag}, Nothing}

struct TiledTriMixedPrec{T_Diag, T_OffDiag} <: AbstractMatrix{T_Diag}
    tiles::Matrix{Tile{T_Diag, T_OffDiag}}
    
    sz::Tuple{Int, Int}
    
    uplo::Char
    
    # to be able to getindex if not a power of 2
    row_sizes::Vector{Int}
    col_sizes::Vector{Int}
end



function TiledTriMixedPrec(
    A::AbstractMatrix, 
    uplo::Char; 
    T_Diag::Type, 
    T_OffDiag::Type, 
    threshold::Int
)
    n = size(A, 1)
    @assert n == size(A, 2) "Matrix must be square"

    num_leaves = _count_leaves(n, threshold)
    
    tiles = Matrix{Tile{T_Diag, T_OffDiag}}(nothing, num_leaves, num_leaves)
    row_sizes = zeros(Int, num_leaves)
    col_sizes = zeros(Int, num_leaves)

    _recursive_tile_split!(
        tiles, row_sizes, col_sizes,
        A,                          
        1, 1,                       
        uplo, T_Diag, T_OffDiag, threshold
    )

    return TiledTriMixedPrec{T_Diag, T_OffDiag}(tiles, (n, n), uplo, row_sizes, col_sizes)
end


function _recursive_tile_split!(
    tiles, row_sizes, col_sizes,
    A_view,
    tile_r::Int, tile_c::Int,
    uplo, T_Diag, T_OffDiag, threshold
)
    n = size(A_view, 1)

    
    if n <= threshold
        tiles[tile_r, tile_c] = T_Diag.(A_view) 
        row_sizes[tile_r] = n
        col_sizes[tile_c] = n
        return
    end

    
    mid = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))

    A11_view = view(A_view, 1:mid, 1:mid)
    A22_view = view(A_view, mid+1:n, mid+1:n)
    
    A11_leaves = _count_leaves(mid, threshold)

    if uplo == 'L'
        A21_view = view(A_view, mid+1:n, 1:mid)
        
        tiles[tile_r + A11_leaves, tile_c] = T_OffDiag.(A21_view)
        
        _recursive_tile_split!(tiles, row_sizes, col_sizes, A11_view, tile_r, tile_c, uplo, T_Diag, T_OffDiag, threshold)
        _recursive_tile_split!(tiles, row_sizes, col_sizes, A22_view, tile_r + A11_leaves, tile_c + A11_leaves, uplo, T_Diag, T_OffDiag, threshold)

    else # uplo == 'U'
        A12_view = view(A_view, 1:mid, mid+1:n)
        
        tiles[tile_r, tile_c + A11_leaves] = T_OffDiag.(A12_view)
        
        _recursive_tile_split!(tiles, row_sizes, col_sizes, A11_view, tile_r, tile_c, uplo, T_Diag, T_OffDiag, threshold)
        _recursive_tile_split!(tiles, row_sizes, col_sizes, A22_view, tile_r + A11_leaves, tile_c + A11_leaves, uplo, T_Diag, T_OffDiag, threshold)
    end
end

function _count_leaves(n::Int, threshold::Int)
    if n <= threshold
        return 1
    end
    
    mid = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))
    
    leaves_A11 = _count_leaves(mid, threshold)
    leaves_A22 = _count_leaves(n - mid, threshold)
    
    return leaves_A11 + leaves_A22
end

function Base.size(A::TiledTriMixedPrec)
    return A.sz
end

function Base.size(A::TiledTriMixedPrec, dim::Integer)
    return A.sz[dim]
end


function find_tile_and_local_indices(sizes::Vector{Int}, global_idx::Int)
    local_idx = global_idx
    cumulative_size = 0
    
    for (tile_idx, tile_size) in enumerate(sizes)
        if local_idx <= tile_size
            # We found the correct tile!
            return tile_idx, local_idx
        end
        # Otherwise, subtract this tile's size and move to the next.
        local_idx -= tile_size
        cumulative_size += tile_size # This line is not strictly needed for the logic, but good for debugging
    end
    
    # If we get here, the index is out of bounds
    error("Global index $global_idx is out of bounds.")
end


function Base.getindex(A::TiledTriMixedPrec{T_Diag, T_OffDiag}, i::Int, j::Int) where {T_Diag, T_OffDiag}
    # 1. Check for out-of-bounds access on the global matrix
    rows, cols = A.sz
    if !(1 <= i <= rows && 1 <= j <= cols)
        throw(BoundsError(A, (i, j)))
    end

    # 2. Check if the index is in the zero part of the triangular matrix
    #    before doing any complex calculations. This is a fast path.
    if A.uplo == 'L' && j > i
        return zero(T_Diag)
    elseif A.uplo == 'U' && i > j
        return zero(T_Diag)
    end

    # 3. Find the tile coordinates and local indices
    tile_r, local_i = find_tile_and_local_indices(A.row_sizes, i)
    tile_c, local_j = find_tile_and_local_indices(A.col_sizes, j)
    
    # 4. Retrieve the tile from the grid
    tile = A.tiles[tile_r, tile_c]

    # 5. Get the value
    if tile === nothing
        # This can happen if we access an index on the main diagonal
        # that falls between tiles in the explicit zero part. For example,
        # in a lower-triangular matrix, A.tiles[1, 2] would be nothing.
        return zero(T_Diag)
    else
        # Access the element within the tile using local indices
        return tile[local_i, local_j]
    end
end