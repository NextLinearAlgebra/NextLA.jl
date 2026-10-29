using Test, LinearAlgebra, SparseArrays
include("run.jl")
include("accuracy.jl")

struct TestTree
    BaseCase
    base_scale
    A11
    A22
    OffDiag
    offDiag_scale
    uplo::Char
    n::Int
end
Base.size(tree::TestTree) = (tree.n, tree.n)
Base.size(tree::TestTree, dim::Integer) = dim <= 2 ? tree.n : 1

@testset "Catalog selection and CSV checkpoints" begin
    withenv("REAL_MATRICES" => nothing, "MATRIX_MANIFEST" => nothing) do
        config = CatalogSweep.settings()
        entries = CatalogSweep.selection(config)
        @test length(entries) == 20
        @test length(unique(getindex.(entries, "group"))) == 6
        @test minimum(parse.(Int, getindex.(entries, "n"))) == 3948
        @test maximum(parse.(Int, getindex.(entries, "n"))) == 36000
        unavailable = CatalogSweep.read_csv(joinpath(@__DIR__, "matrices-unavailable.csv"))
        @test length(unavailable) == 10
        @test all(row -> row["http_status"] == "404", unavailable)
        @test isempty(intersect(getindex.(entries,"name"), getindex.(unavailable,"name")))
        @test length(union(getindex.(entries,"name"), getindex.(unavailable,"name"))) == 30
        @test isempty(intersect(getindex.(entries, "name"), ["nasasrb", "crankseg_2", "bodyy4", "bodyy5", "bodyy6", "inla_graph_animal2"]))
        withenv("REAL_MATRICES" => "nd3k,bcsstk15") do
            @test length(CatalogSweep.selection(config)) == 2
        end
        withenv("REAL_MATRICES" => "not_in_catalog") do
            @test_throws ErrorException CatalogSweep.selection(config)
        end
        withenv("REAL_MATRICES" => "bcsstk08") do
            @test_throws ErrorException CatalogSweep.selection(config)
        end
    end
    @test_throws ErrorException withenv("GPU_TIMEOUT_SECONDS" => "0") do
        CatalogSweep.settings()
    end
    mktempdir() do dir
        path = joinpath(dir, "rows.csv")
        original = Dict("matrix" => "case", "detail" => "a comma, and a \"quote\"", "status" => "TIMEOUT")
        CatalogSweep.append_row(path, original)
        @test only(CatalogSweep.read_csv(path))["detail"] == original["detail"]
        CatalogSweep.write_csv(path, [merge(original, Dict("status" => "OK"))])
        @test only(CatalogSweep.read_csv(path))["status"] == "OK"
    end
end

@testset "Scale-aware decoding and error metrics" begin
    leaf = TestTree([2.0 0; 0 3.0], 2f0, nothing, nothing, nothing, nothing, 'L', 2)
    @test decode_blocks(leaf) == [4 0; 0 6]
    for uplo in ('L', 'U')
        tree = TestTree(nothing, nothing, leaf, leaf, [0.375 -0.375; 1 2], 2f0, uplo, 4)
        decoded = decode_blocks(tree)
        @test issymmetric(decoded)
        @test (uplo == 'L' ? decoded[3:4,1:2] : decoded[1:2,3:4]) == [0.75 -0.75; 2 4]
        @test scale_stats(tree) == (3, 2.0)
    end
    A = spdiagm(0 => [2.0, 3, 4])
    config = Dict("seed" => 17, "probes" => 8, "nrhs" => 4)
    inputs = accuracy_inputs(A, config)
    @test inputs.omega == accuracy_inputs(A, config).omega
    @test probe_error(Matrix(A) * inputs.omega, inputs) == 0
    @test all(iszero, values(solution_metrics(A, inputs.truth, inputs)))
    @test solution_metrics(A, 2inputs.truth, inputs)["forward_error"] ≈ 1
    @test input_diagnostics(A)["dominant_fraction"] == 1
    @test input_diagnostics(A)["density"] ≈ 1/3
end

@testset "External process timeouts and errors" begin
    mktempdir() do dir
        command = `$(Base.julia_cmd()) --startup-file=no -e $("println(123)")`
        status, code, _ = CatalogSweep.run_bounded(command, 15.0, joinpath(dir, "ok.log"))
        @test status == "DONE" && code == 0
        @test occursin("123", read(joinpath(dir, "ok.log"), String))
        failing = `$(Base.julia_cmd()) --startup-file=no -e $("error(\"expected failure\")")`
        status, code, _ = CatalogSweep.run_bounded(failing, 15.0, joinpath(dir, "fail.log"))
        @test status == "ERROR" && code != 0
        # Block in a native call, rather than a cooperative Julia task.
        native = Sys.iswindows() ? "ccall(:Sleep, stdcall, Cvoid, (UInt32,), 120000)" : "ccall(:sleep, Cuint, (Cuint,), 120)"
        blocked = `$(Base.julia_cmd()) --startup-file=no -e $("println(\"entered native call\"); flush(stdout); " * native)`
        status, _, elapsed = CatalogSweep.run_bounded(blocked, 3.0, joinpath(dir, "timeout.log"))
        @test status == "TIMEOUT"
        @test elapsed < 12.0
        @test occursin("entered native call", read(joinpath(dir, "timeout.log"), String))
    end
end

@testset "Julia source syntax" begin
    paths = vcat(filter(p -> endswith(p, ".jl"), readdir(@__DIR__; join=true)),
        [joinpath(CatalogSweep.ROOT, "src", "rectrxm.jl"), joinpath(CatalogSweep.ROOT, "src", "wrappers.jl")])
    for path in paths
        parsed = Meta.parseall(read(path, String); filename=path)
        @test !any(ex -> ex isa Expr && ex.head in (:error, :incomplete), parsed.args)
        lowered = Meta.lower(Main, parsed)
        @test !(lowered isa Expr && lowered.head == :error)
    end
end
