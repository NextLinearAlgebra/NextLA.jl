# CPU Float64 ordering exploration; no NextLA/CUDA or mixed precision.
using LinearAlgebra, SparseArrays, Random, Printf, Dates, SHA
import MatrixMarket
setting(name, default) = parse(Int, get(ENV, name, string(default)))
include("methods.jl")
esc(x) = replace(string(x), '&'=>"&amp;", '<'=>"&lt;", '>'=>"&gt;", '"'=>"&quot;")
csv(x) = "\"" * replace(string(x), "\""=>"\"\"") * "\""

function loadmatrix(path)
    A = sparse(Float64.(MatrixMarket.mmread(path)))
    size(A, 1) == size(A, 2) || error("Input must be square")
    if !issymmetric(A)
        A = istril(A) ? sparse(Symmetric(A, :L)) :
            istriu(A) ? sparse(Symmetric(A, :U)) : error("Inconsistent triangles")
    end
    dropzeros!(A)
    all(isfinite, nonzeros(A)) || error("Nonfinite values")
    A
end

function demo(seed)
    # Strictly diagonally dominant grid Laplacian + identity, hence SPD.
    side = 24
    n = side^2
    I = Int[]; J = Int[]; V = Float64[]
    for j in 1:side, i in 1:side
        v = i + (j-1)*side
        for w in (i < side ? v+1 : 0, j < side ? v+side : 0)
            w == 0 && continue
            append!(I, [v,w]); append!(J, [w,v]); append!(V, [-1.,-1.])
        end
    end
    A = sparse(I, J, V, n, n)
    A += spdiagm(0 => vec(sum(abs, A; dims=1)) .+ 1)
    p = randperm(MersenneTwister(seed), n)
    A[p,p]
end

# Exact occupancy aggregation: every nonzero contributes to a pixel. No random
# sampling and no n-by-n dense conversion. Bin counts are intentionally binary.
function spy_svg(path, A, depth; original=nothing)
    n = size(A, 1)
    pixels = min(n, 600)
    bins = zeros(UInt8, pixels, pixels)
    for j in 1:n, k in nzrange(A, j)
        i = A.rowval[k]
        iszero(A.nzval[k]) && continue
        color = original === nothing || !iszero(original[i,j]) ? UInt8(1) : UInt8(2)
        r = min(pixels, fld((i-1)*pixels,n)+1)
        c = min(pixels, fld((j-1)*pixels,n)+1)
        bins[r,c] = max(bins[r,c], color)
    end
    open(path, "w") do io
        println(io, "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 $pixels $pixels'><rect width='100%' height='100%' fill='white'/>")
        for c in 1:pixels, r in 1:pixels
            bins[r,c] == 0 && continue
            color = bins[r,c] == 1 ? "#164e63" : "#e87924"
            println(io, "<rect x='$(c-1)' y='$(r-1)' width='1' height='1' fill='$color'/>")
        end
        for region in regions(n, depth)
            lo = (region.lo-1)*pixels/n; hi = region.hi*pixels/n
            cut = region.cut*pixels/n
            println(io, "<path d='M $lo $cut H $hi M $cut $lo V $hi' fill='none' stroke='#e11d48' stroke-width='0.7'/>")
        end
        println(io, "</svg>")
    end
end

const COLUMNS = [:matrix,:method,:status,:detail,:n,:nnz_A,:setup_s,:factor_min_s,
    :factor_median_s,:solve_min_s,:nnz_L,:fill_ratio,:bandwidth,:profile,
    :column_count_square_sum,:factor_residual,:solve_residual,:forward_error]

function measure(A, method, seed, repeats, probes, folder, depth)
    n = size(A, 1)
    row = Dict{Symbol,Any}(:method=>method, :n=>n, :nnz_A=>nnz(A), :status=>"FAILED")
    try
        if method in ("amd", "scotch", "stiles")
            package = method == "amd" ? :AMD : method == "scotch" ? :Scotch : :sTiles
            Base.eval(@__MODULE__, Expr(:import, Expr(:., package)))
        end
        order = ORDERERS[method]
        # Warm on this matrix, then time one fresh setup. CHOLMOD setup is a
        # complete factorization; sTiles setup includes tile layout.
        Base.invokelatest(order, A, seed)
        row[:setup_s] = @elapsed raw = Base.invokelatest(order, A, seed)
        p = Int.(raw)
        sort(p) == collect(1:n) || error("Invalid permutation")
        Ap = A[p,p]
        nnz(Ap) == nnz(A) || error("Permutation changed nonzero count")
        Ap[invperm(p),invperm(p)] == A || error("Permutation round trip failed")
        open(joinpath(folder, method*"-permutation.csv"), "w") do io
            println(io, "new_index,original_index")
            for (i,j) in enumerate(p); println(io, "$i,$j"); end
        end
        spy_svg(joinpath(folder, method*"-matrix.svg"), Ap, depth)
        fixed = collect(1:n)
        cholesky(Symmetric(Ap); perm=fixed, check=true)
        times = Float64[]
        F = nothing
        for _ in 1:repeats
            F = nothing
            GC.gc()
            push!(times, @elapsed F = cholesky(Symmetric(Ap); perm=fixed, check=true))
        end
        collect(F.p) == fixed || error("Unexpected internal CHOLMOD permutation")
        L = sparse(F.L)
        dropzeros!(L)
        row[:factor_min_s] = minimum(times)
        sorted_times = sort(times)
        row[:factor_median_s] = (sorted_times[cld(length(times),2)] + sorted_times[fld(length(times),2)+1])/2
        row[:nnz_L] = nnz(L)
        row[:fill_ratio] = nnz(L)/nnz(tril(Ap))
        row[:column_count_square_sum] = sum(x -> Float64(x)^2, diff(L.colptr))
        bandwidth = 0; profile = 0
        for j in 1:n
            firstrow = j
            for k in nzrange(Ap,j)
                i = Ap.rowval[k]
                bandwidth = max(bandwidth, abs(i-j))
                firstrow = min(firstrow,i)
            end
            profile += j-firstrow
        end
        row[:bandwidth] = bandwidth; row[:profile] = profile
        rng = MersenneTwister(seed)
        W = randn(rng,n,probes)
        AW = Ap*W
        row[:factor_residual] = norm(L*(L'*W)-AW)/norm(AW)
        truth = randn(rng,n)
        b = A*truth; bp = b[p]
        F \ bp
        stimes = Float64[]; y = nothing
        for _ in 1:repeats
            push!(stimes, @elapsed y = F \ bp)
        end
        row[:solve_min_s] = minimum(stimes)
        open(joinpath(folder,method*"-timings.csv"),"w") do io
            println(io,"repetition,complete_factor_s,solve_s")
            for i in eachindex(times)
                println(io,"$i,$(times[i]),$(stimes[i])")
            end
        end
        x = similar(y); x[p] = y
        row[:solve_residual] = norm(A*x-b)/norm(b)
        row[:forward_error] = norm(x-truth)/norm(truth)
        for key in (:factor_residual,:solve_residual,:forward_error)
            isfinite(row[key]) || error("Nonfinite $key")
        end
        row[:status] = row[:factor_residual] < 1e-10 && row[:solve_residual] < 1e-10 ? "OK" : "INACCURATE"
        spy_svg(joinpath(folder,method*"-factor.svg"), L, depth; original=Ap)
        # Geometry/value diagnostics for each lower off-diagonal recursion region.
        open(joinpath(folder,method*"-blocks.csv"),"w") do io
            println(io,"level,lo,cut,hi,rows,cols,nnz,density,frobenius_norm,max_abs")
            for r in regions(n,depth)
                B = Ap[r.cut+1:r.hi,r.lo:r.cut]
                vals = nonzeros(B)
                println(io,join((r.level,r.lo,r.cut,r.hi,size(B,1),size(B,2),nnz(B),
                    nnz(B)/prod(size(B)),norm(vals),isempty(vals) ? 0.0 : maximum(abs,vals)),','))
            end
        end
    catch err
        row[:status] = "FAILED"
        row[:detail] = sprint(showerror,err)
        println(stderr, method, ": ", row[:detail])
    end
    row
end

function render(out, rows, metadata)
    open(joinpath(out,"metrics.csv"),"w") do io
        println(io,join(string.(COLUMNS),','))
        for row in rows
            println(io,join([csv(get(row,c,"")) for c in COLUMNS],','))
        end
    end
    open(joinpath(out,"index.html"),"w") do io
        print(io,"""<!doctype html><meta charset="utf-8"><title>Cholesky ordering laboratory</title>
        <style>body{font:16px system-ui;margin:32px auto;max-width:1400px;padding:0 20px;color:#18313b}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}.pictures{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}img{width:100%;border:1px solid #ccc}figure{margin:0}small{color:#52636b}code{background:#eef4f5}article{margin:40px 0}a{color:#096c87}</style>
        <h1>Cholesky ordering laboratory · Float64</h1>
        <p>Permuting rows and columns together preserves the problem and input nonzero count. Elimination can create new nonzeros (fill). Ordering changes that fill and where values fall in recursive blocks.</p>
        <p>All numerical results use CPU sparse CHOLMOD with the supplied order fixed. These are not NextLA GPU timings, native sTiles factor timings, or mixed-precision error predictions.</p>
        <p>Blue = occupied input/factor pixel; orange = factor fill absent from the reordered input; red = recursive diagonal-block splits matching <code>src/symmmixedprec.jl</code>. For n &gt; 600, each pixel aggregates a region: any nonzero lights it up; orange takes priority. Pictures are occupancy maps, not density or magnitude maps.</p>
        <p>Smaller nnz(L) means less factor storage. Bandwidth/profile measure locality; they do not determine fill. The sum of squared factor column counts is a structural work proxy, not a measured flop count. Timing winners can differ from storage winners.</p>
        <p>Setup is warmed but scopes differ: CHOLMOD includes a full factorization; sTiles includes ordering and tile layout. Factor timings include fresh symbolic + numeric factorization, exclude permutation/plotting/validation. Solve timings include allocation for one RHS. GC runs before each factor timing.</p>
        <p><a href="metrics.csv">Download all metrics</a></p>""")
        println(io,"<pre>",esc(metadata),"</pre>")
        for name in unique([r[:matrix] for r in rows])
            group = filter(r -> r[:matrix] == name, rows)
            println(io,"<h2>",esc(name),"</h2><table><tr>")
            fields = [:method,:status,:setup_s,:factor_min_s,:factor_median_s,:solve_min_s,:nnz_L,:fill_ratio,:bandwidth,:profile,:factor_residual,:solve_residual,:forward_error]
            for c in fields; print(io,"<th>",esc(c),"</th>"); end
            println(io,"</tr>")
            for r in group
                println(io,"<tr>")
                for c in fields
                    v = get(r,c,""); v = v isa AbstractFloat ? @sprintf("%.4g",v) : v
                    print(io,"<td>",esc(v),"</td>")
                end
                println(io,"</tr>")
            end
            println(io,"</table>")
            good = filter(r -> r[:status] == "OK",group)
            if !isempty(good)
                storage = good[argmin([r[:nnz_L] for r in good])][:method]
                speed = good[argmin([r[:factor_min_s] for r in good])][:method]
                println(io,"<p>Among successful candidates: smallest factor = ",esc(storage),"; fastest minimum factor time = ",esc(speed),". Check repeated runs before interpreting small differences.</p>")
            end
            for r in group
                method = r[:method]; prefix = name*"/"*method
                println(io,"<article><h3>",esc(method)," · ",esc(r[:status]),"</h3>")
                haskey(r,:detail) && println(io,"<pre>",esc(r[:detail]),"</pre>")
                if isfile(joinpath(out,prefix*"-factor.svg"))
                    println(io,"<div class='pictures'>")
                    for (label,file) in (("Before: original matrix",name*"/original.svg"),("After: reordered matrix",prefix*"-matrix.svg"),("Cholesky L with fill",prefix*"-factor.svg"))
                        println(io,"<figure><figcaption>",label,"</figcaption><a href='",esc(file),"'><img loading='lazy' src='",esc(file),"'></a></figure>")
                    end
                    println(io,"</div><p><a href='",esc(prefix),"-permutation.csv'>Permutation</a> · <a href='",esc(prefix),"-blocks.csv'>Recursive block statistics</a></p>")
                end
                println(io,"</article>")
            end
        end
        print(io,"<h2>What this tells us about recursion</h2><p>The split sizes stay fixed for a given n and depth; ordering changes the values in those regions. Block CSVs show each lower off-diagonal region's occupancy, norm, and maximum magnitude. Later Schur updates can fill or change these blocks. Neither these plots nor input block statistics establish a safe precision schedule: that requires a later controlled factorization/error experiment.</p>")
    end
end

function main()
    all(a -> a == "--demo",ARGS) || error("Supported flag: --demo")
    seed = setting("ORDERING_SEED",20261105)
    repeats = setting("ORDERING_REPEATS",5)
    probes = setting("ORDERING_PROBES",4)
    depth = setting("ORDERING_DEPTH",3)
    repeats > 0 && probes > 0 && 0 <= depth <= 12 || error("Invalid repeats, probes, or depth")
    BLAS.set_num_threads(setting("ORDERING_BLAS_THREADS",1))
    methods = strip.(split(get(ENV,"ORDERING_METHODS","natural,random,amd,rcm,scotch,cholmod,stiles"),','))
    all(m -> haskey(ORDERERS,m),methods) || error("Unknown method")
    out = abspath(get(ENV,"ORDERING_OUTPUT",joinpath(@__DIR__,"results",Dates.format(now(),"yyyymmdd-HHMMSS"))))
    mkpath(out)
    names = "--demo" in ARGS ? ["demo-grid-scrambled"] : strip.(split(get(ENV,"ORDERING_MATRICES",get(ENV,"ORDERING_MATRIX","bodyy5")),','))
    all(n -> occursin(r"^[A-Za-z0-9_-]+$",n),names) || error("Matrix names must be simple basenames")
    metadata = "Julia=$(VERSION)\nseed=$seed repeats=$repeats probes=$probes depth=$depth\nBLAS threads=$(BLAS.get_num_threads()) config=$(BLAS.get_config())\nCPU=$(Sys.CPU_NAME) hardware_threads=$(Sys.CPU_THREADS)\nSTILES_CORES=$(get(ENV,"STILES_CORES","1"))\nproject=$(Base.active_project())\n"
    rows = Dict{Symbol,Any}[]
    for name in names
        folder = joinpath(out,name); mkpath(folder)
        path = joinpath(get(ENV,"MATRIX_DIR",dirname(dirname(@__DIR__))),name*".mtx")
        try
            A = "--demo" in ARGS ? demo(seed) : loadmatrix(path)
            metadata *= "$name: n=$(size(A,1)) nnz=$(nnz(A)) source=$("--demo" in ARGS ? "synthetic grid" : path)\n"
            if !("--demo" in ARGS)
                metadata *= "sha256=" * bytes2hex(open(sha256,path)) * "\n"
            end
            spy_svg(joinpath(folder,"original.svg"),A,depth)
            for method in methods
                println("$name / $method")
                r = measure(A,method,seed,repeats,probes,folder,depth)
                r[:matrix] = name; push!(rows,r)
                render(out,rows,metadata) # checkpoint after every candidate
            end
        catch err
            push!(rows,Dict(:matrix=>name,:method=>"input",:status=>"FAILED",:detail=>sprint(showerror,err)))
            render(out,rows,metadata)
        end
    end
    # Record exact resolved dependencies for reproducibility when available.
    for file in ("Project.toml","Manifest.toml")
        source = joinpath(dirname(Base.active_project()),file)
        isfile(source) && cp(source,joinpath(out,file); force=true)
    end
    render(out,rows,metadata)
    println("Report: ",joinpath(out,"index.html"))
    any(r -> r[:status] != "OK",rows) && println("Some candidates failed or were inaccurate; inspect the report details.")
end
abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
