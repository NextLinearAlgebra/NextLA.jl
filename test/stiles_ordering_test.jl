# CPU-only ordering check; deliberately does not import NextLA (which loads CUDA).
# MATRIX_DIR=. ORDERING_MATRIX=bodyy5 STILES_CORES=8 \
#   julia --project=test/catalog test/stiles_ordering_test.jl
# Use --example-only to run just Esmail's example.
using Test, LinearAlgebra, SparseArrays, Random, Printf

try
    @eval import sTiles
catch err
    println(stderr, "sTiles import failed in active project $(Base.active_project()): ",
            sprint(showerror, err))
    rethrow()
end
println("Julia $(VERSION); sTiles version=$(Base.pkgversion(sTiles)); source=$(pathof(sTiles))")
println("sTiles API: ", isdefined(sTiles, :ordering) ? "standalone ordering" :
        "analyze + permutation (ordering bake-off and tile layout)")

function positive_setting(name, default)
    value = parse(Int, get(ENV, name, string(default)))
    value > 0 || error("$name must be positive")
    value
end

function permutation(raw, n)
    values = collect(raw)
    @test values isa AbstractVector
    @test eltype(values) <: Integer
    p = Int.(values)
    valid = length(p) == n && sort(p) == collect(1:n)
    @test valid
    valid || error("Expected a 1-based permutation of 1:$n; got $p")
    p
end

natural_order(A, cores) = collect(1:size(A, 1))
# Extract CHOLMOD's default permutation, then use the same fixed-order
# factorization as every other candidate below. This setup includes a factorization.
cholmod_order(A, cores) = Int.(cholesky(Symmetric(A)).p)
function stiles_order(A, cores)
    try
        if isdefined(sTiles, :ordering)
            return sTiles.ordering(A; cores=cores)
        end
        F = sTiles.analyze(A; cores=cores, inverse=false)
        try
            # Current public API; no numeric sTiles factorization is needed.
            return invperm(Int.(sTiles.permutation(F)))
        finally
            close(F)
        end
    catch err
        println(stderr, "sTiles.ordering failed; version=$(Base.pkgversion(sTiles)), ",
                "size=$(size(A)), cores=$cores: ", sprint(showerror, err))
        rethrow()
    end
end

function inspect_order(o, n)
    println("typeof(o) = ", typeof(o))
    println("propertynames(o) = ", propertynames(o))
    println("o.iperm = ", o.iperm)
    p = permutation(o.iperm, n)
    if hasproperty(o, :perm)
        println("o.perm = ", o.perm)
        q = permutation(o.perm, n)
        inverses = q[p] == p[q] == collect(1:n)
        println("perm and iperm are inverses = ", inverses)
        @test inverses
    end
    p
end

function check_case(label, A, order, cores, W, x_true; inspect=false)
    n = size(A, 1)
    # Only the ordering call is timed, excluding conversion and validation.
    ordering_seconds = @elapsed result = order(A, cores)
    p = result isa AbstractVector ? permutation(result, n) :
        (inspect ? inspect_order(result, n) : permutation(result.iperm, n))
    Ap = A[p, p]
    @test size(Ap) == size(A)
    @test issymmetric(Ap)
    @test nnz(Ap) == nnz(A)
    inverse = invperm(p)
    @test Ap[inverse, inverse] == A

    # CHOLMOD's explicit perm overrides its default fill-reducing ordering.
    # Includes symbolic and numeric factorization, excludes extracting L.
    # Warm this path, then compare repeated complete factorizations for the
    # same fixed permutation. No factor extraction or residual checks are timed.
    fixed = collect(1:n)
    cholesky(Symmetric(Ap); perm=fixed, check=true)
    times = Float64[]
    F = nothing
    for _ in 1:positive_setting("ORDERING_REPEATS", 3)
        push!(times, @elapsed F = cholesky(Symmetric(Ap); perm=fixed, check=true))
    end
    factor_seconds = minimum(times)
    @test issuccess(F)
    @test collect(F.p) == collect(1:n)
    L = sparse(F.L)
    Wp = W[p, :]
    AW = Ap * Wp
    factor_residual = norm(L * (L' * Wp) - AW) / norm(AW)
    b = A * x_true
    y = F \ b[p]
    x = similar(y)
    x[p] = y
    forward_error = norm(x - x_true) / norm(x_true)
    solve_residual = norm(A * x - b) / norm(b)
    @test all(isfinite, (factor_residual, forward_error, solve_residual))
    @test factor_residual < 1e-10
    @test forward_error < 1e-7
    @test solve_residual < 1e-10
    @printf("%-8s order_s=%.6g nnz(L)=%d fill=%.6g factor_s=%.6g factor_res=%.3e forward_err=%.3e solve_res=%.3e\n",
            label, ordering_seconds, nnz(L), nnz(L) / nnz(tril(Ap)),
            factor_seconds, factor_residual, forward_error, solve_residual)
    (label=label, fill=nnz(L), factor_seconds=factor_seconds)
end

function compare_orderings(A, cores, probes; inspect=false)
    rng = MersenneTwister(20261029)
    W = randn(rng, size(A, 1), probes)
    x_true = randn(rng, size(A, 1))
    results = NamedTuple[]
    for (label, order) in (("natural", natural_order), ("cholmod", cholmod_order),
                           ("stiles", stiles_order))
        @testset "$label" begin
            push!(results, check_case(label, A, order, cores, W, x_true; inspect=inspect))
        end
    end
    println("Least factor storage: ", results[argmin([r.fill for r in results])].label,
            "; fastest measured factorization: ",
            results[argmin([r.factor_seconds for r in results])].label)
end

function main()
    all(arg -> arg == "--example-only", ARGS) || error("Only --example-only is supported")
    cores = positive_setting("STILES_CORES", get(ENV, "SLURM_CPUS_PER_TASK", "1"))
    probes = positive_setting("ORDERING_PROBES", 8)
    println("cores=$cores probes=$probes seed=20261029; Float64 CPU Cholesky")
    println("order_s is one setup call (includes first-call overhead); factor_s is the minimum of warmed complete factorizations")
    println("Setup scopes differ: natural=identity, cholmod=default factorization, stiles=ordering or analysis+layout")
    @testset "sTiles ordering: Esmail's exact example" begin
        n = 6
        Q = sparse(Symmetric(sparse(
            [1,1,1,1,1,1, 2,2, 3,3, 4,4, 5,5, 6],
            [1,2,3,4,5,6, 2,3, 3,4, 4,5, 5,6, 6],
            [9.,1,1,1,1,1, 4,1, 4,1, 4,1, 4,1, 4], n, n)))
        compare_orderings(Q, cores, probes; inspect=true)
    end
    println("Verified on 6x6: Ap=Q[p,p], bp=b[p], x[p]=y (legacy API uses iperm).")
    "--example-only" in ARGS && return
    matrix_dir = get(ENV, "MATRIX_DIR", dirname(@__DIR__))
    name = get(ENV, "ORDERING_MATRIX", "bodyy5")
    path = joinpath(matrix_dir, name * ".mtx")
    if !isfile(path)
        println("SKIP Matrix Market test: file not found: $path")
        @testset "Matrix Market input available" begin
            @test_skip isfile(path)
        end
        return
    end
    # Same conventions as catalog/matrix_io.jl; no solver/benchmark includes.
    # Import lazily so the 6x6 example needs no MatrixMarket package.
    @eval import MatrixMarket
    raw = Base.invokelatest(MatrixMarket.mmread, path)
    size(raw, 1) == size(raw, 2) || error("Matrix must be square: $(size(raw))")
    A = sparse(Float64.(raw))
    if !issymmetric(A)
        if istril(A)
            A = sparse(Symmetric(A, :L))
        elseif istriu(A)
            A = sparse(Symmetric(A, :U))
        else
            error("Input has inconsistent triangles; refusing to average or shift it")
        end
    end
    dropzeros!(A)
    all(isfinite, nonzeros(A)) || error("Nonfinite input")
    println("matrix=$name path=$path size=$(size(A)) nnz=$(nnz(A))")
    @testset "sTiles ordering: $name" begin
        compare_orderings(A, cores, probes)
    end
    println("Verified: Ap=A[p,p], bp=b[p], and x[p]=Ap\\bp recover the original solution.")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
