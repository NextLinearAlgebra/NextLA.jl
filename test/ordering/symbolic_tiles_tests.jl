using Test, Random, LinearAlgebra, SparseArrays
include("methods.jl")
include("symbolic_tiles.jl")

@testset "Symbolic probes match explicit graph elimination" begin
    rng = MersenneTwister(112)
    for n in (3, 9, 32), density in (0.0, 0.15, 0.5, 1.0)
        graph = triu(rand(rng, n, n) .< density, 1)
        graph = graph .| transpose(graph)
        expected = Set{Tuple{Int,Int}}()
        evolving = copy(graph)
        for j in 1:n
            neighbors = [i for i in j+1:n if evolving[i,j]]
            union!(expected, ((i,j) for i in neighbors))
            evolving[neighbors,neighbors] .= true
        end
        factor = Set{Tuple{Int,Int}}()
        count = symbolic_lower!((i,j) -> push!(factor,(i,j)), sparse(graph))
        @test factor == expected
        @test count == n + length(expected)
        @test_throws SymbolicLimit symbolic_lower!((i,j) -> nothing, sparse(graph); limit=count-1)
    end
end

@testset "Recursive tile counts include fill and handle edge tiles" begin
    # Vertices 3 and 5 both attach to 1. Eliminating 1 fills (5,3), in
    # the root panel, while the root's row 6 stays inactive.
    graph = sparse([1,3,1,5], [3,1,5,1], ones(4), 6, 6)
    count, panels = observe_panels(graph, regions(6, 3), [1, 2, 3]; limit=100)
    @test count == 9
    root = first(panels)
    a, b, c = [panel_metrics(root, w, w) for w in (1, 2, 3)]
    @test (a.active_panel_rows, a.active_panel_cols) == (1, 2)
    @test (a.input_entries, a.factor_entries, a.fill_only_tiles) == (1, 2, 1)
    @test (a.tiles, a.empty_tiles) == (8, 6)
    @test (b.tiles, b.empty_tiles, b.kept_cells) == (2, 0, 8)
    @test (b.row_packed_cells, b.col_packed_cells, b.both_packed_cells) == (4, 4, 2)
    @test (c.tiles, c.empty_tiles, c.kept_cells) == (2, 1, 6)
end

@testset "An occupied pair can share no inner columns" begin
    A = sparse([5,1,7,2], [1,5,2,7], ones(4), 8, 8)
    _, panels = observe_panels(A, regions(8, 1), [2]; limit=100)
    counts = panel_metrics(only(panels), 2, 1)
    @test (counts.dense_schur_products, counts.source_schur_products,
           counts.shared_col_schur_products) == (6, 3, 2)
end
