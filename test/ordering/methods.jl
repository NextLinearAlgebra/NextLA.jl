# A symmetric matrix's off-diagonal pattern is an undirected graph.
function adjacency(A)
    G = copy(A)
    G[diagind(G)] .= 0
    dropzeros!(G)
    G
end

function rcm(A, seed)
    G = adjacency(A)
    n = size(G, 1)
    degree = diff(G.colptr)
    roots = sortperm(1:n; by=i -> (degree[i], i))
    seen = falses(n)
    result = Int[]
    for root in roots
        seen[root] && continue
        queue = [root]
        seen[root] = true
        head = 1
        while head <= length(queue)
            v = queue[head]
            head += 1
            push!(result, v)
            neighbors = [G.rowval[k] for k in nzrange(G, v) if !seen[G.rowval[k]]]
            sort!(neighbors; by=i -> (degree[i], i))
            for w in neighbors
                seen[w] && continue
                seen[w] = true
                push!(queue, w)
            end
        end
    end
    reverse!(result)
end

function scotch_order(A, seed)
    G = adjacency(A)
    T = Scotch.SCOTCH_Num
    basegraph = Scotch.graph_build(T.(G.colptr), T.(G.rowval); index_start=1)
    context = Scotch.context_alloc()
    Scotch.context_option!(context, :deterministic, true)
    Scotch.context_option!(context, :fixed_seed, true)
    Scotch.random_clone(context)
    Scotch.random_seed(context, T(seed))
    Scotch.random_reset(context)
    graph = Scotch.bind_graph(context, basegraph)
    strategy = Scotch.strat_alloc() # SCOTCH's default graph ordering strategy
    n = size(A, 1)
    direct = Vector{T}(undef, n) # original vertex -> new position
    inverse = similar(direct)   # new position -> original vertex, for A[p,p]
    # Scotch.jl currently has no high-level graph_order wrapper. Its generated
    # binding mirrors the native SCOTCH_graphOrder API. Keep this adapter here.
    code = GC.@preserve basegraph context graph strategy direct inverse begin
        Scotch.LibScotch.SCOTCH_graphOrder(graph, strategy, direct, inverse,
                                         C_NULL, C_NULL, C_NULL)
    end
    code == 0 || error("SCOTCH_graphOrder returned $code")
    p = Int.(inverse)
    Int.(direct)[p] == collect(1:n) || error("SCOTCH permutation direction check failed")
    p
end

function stiles_order(A, seed)
    if isdefined(sTiles, :ordering)
        return Int.(sTiles.ordering(A; cores=setting("STILES_CORES", 1)).iperm)
    end
    F = sTiles.analyze(A; cores=setting("STILES_CORES", 1), inverse=false)
    try
        Int.(sTiles.permutation(F))
    finally
        close(F)
    end
end

const ORDERERS = Dict{String,Function}(
    "natural" => (A, seed) -> collect(1:size(A, 1)),
    "random" => (A, seed) -> randperm(MersenneTwister(seed), size(A, 1)),
    "amd" => (A, seed) -> Int.(AMD.amd(A)),
    "rcm" => rcm,
    "scotch" => scotch_order,
    "cholmod" => (A, seed) -> Int.(cholesky(Symmetric(A)).p),
    "stiles" => stiles_order,
)

# Same split rule as src/symmmixedprec.jl, recursively on diagonal blocks.
function regions(n, depth; offset=0, level=1)
    (depth == 0 || n <= 1) && return NamedTuple[]
    power = prevpow(2, n)
    mid = n == power ? n ÷ 2 : power
    here = (lo=offset+1, cut=offset+mid, hi=offset+n, level=level)
    vcat([here], regions(mid, depth-1; offset=offset, level=level+1),
         regions(n-mid, depth-1; offset=offset+mid, level=level+1))
end
